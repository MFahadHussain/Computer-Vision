#!/usr/bin/env python3
"""
AI Security System with Face and License Plate Recognition
Supports RTSP cameras and provides a web dashboard
"""

import os
import sys
import time
import threading
import logging
import argparse
import signal
from datetime import datetime
import sqlite3
import json
import cv2
import numpy as np
import psutil
from flask import Flask, Response, jsonify, render_template_string
from werkzeug.serving import make_server

# Import our modules
from RTSPStreamHandler import RTSPStreamHandler
from FaceRecognizer import FaceRecognizer
from PlateRecognizer import PlateRecognizer
from HardwareController import HardwareController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecuritySystem:
    """Main security system controller"""
    
    def __init__(self, config):
        self.config = config
        self.running = False
        self.db_connection = None
        self.hardware = None
        self.face_recognizer = None
        self.plate_recognizer = None
        self.threads = []
        
        # Dashboard data
        self.dashboard_data = {
            'known_faces_count': 0,
            'today_count': 0,
            'authorized_count': 0,
            'unauthorized_count': 0,
            'face_feed': None,
            'plate_feed': None,
            'logs': [],
            'attendance': [],
            'system_status': {
                'face_camera': False,
                'plate_camera': False,
                'cpu_usage': 0,
                'memory_usage': 0,
                'uptime': 0
            }
        }
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            # Setup database
            self._setup_database()
            
            # Initialize hardware controller
            self.hardware = HardwareController(
                barrier_pin=self.config.BARRIER_GPIO_PIN,
                green_led_pin=self.config.GREEN_LED_GPIO_PIN,
                red_led_pin=self.config.RED_LED_GPIO_PIN
            )
            
            # Initialize face recognizer
            self.face_recognizer = FaceRecognizer(
                rtsp_url=self.config.FACE_CAMERA_RTSP,
                model_name=self.config.FACE_RECOGNITION_MODEL,
                threshold=self.config.FACE_RECOGNITION_THRESHOLD,
                cooldown=self.config.FACE_RECOGNITION_COOLDOWN
            )
            
            # Initialize plate recognizer
            self.plate_recognizer = PlateRecognizer(
                rtsp_url=self.config.PLATE_CAMERA_RTSP,
                threshold=self.config.PLATE_CONFIDENCE_THRESHOLD,
                cooldown=self.config.PLATE_RECOGNITION_COOLDOWN
            )
            
            # Update dashboard data
            self.dashboard_data['known_faces_count'] = len(self.face_recognizer.known_names)
            
            logger.info("System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}")
            raise
    
    def _setup_database(self):
        """Initialize database tables"""
        try:
            self.db_connection = sqlite3.connect(self.config.DATABASE_PATH, check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            # Create logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    type TEXT,
                    data TEXT,
                    authorized INTEGER,
                    snapshot_path TEXT
                )
            ''')
            
            # Create attendance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    person_name TEXT,
                    plate_number TEXT
                )
            ''')
            
            self.db_connection.commit()
            logger.info("Database setup completed")
            
        except Exception as e:
            logger.error(f"Database setup failed: {str(e)}")
            raise
    
    def start(self):
        """Start the security system"""
        if self.running:
            logger.warning("System is already running")
            return
        
        self.running = True
        self.start_time = time.time()
        
        # Start processing threads
        self.threads = [
            threading.Thread(target=self._process_face_camera, daemon=True),
            threading.Thread(target=self._process_plate_camera, daemon=True),
            threading.Thread(target=self._monitor_system, daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
        
        logger.info("Security system started")
    
    def stop(self):
        """Stop the security system"""
        if not self.running:
            logger.warning("System is not running")
            return
        
        logger.info("Stopping security system...")
        self.running = False
        
        # Stop recognizers
        if self.face_recognizer:
            self.face_recognizer.release()
        
        if self.plate_recognizer:
            self.plate_recognizer.release()
        
        # Close hardware
        if self.hardware:
            self.hardware.cleanup()
        
        # Close database
        if self.db_connection:
            self.db_connection.close()
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1)
        
        logger.info("Security system stopped")
    
    def _process_face_camera(self):
        """Process face camera feed"""
        while self.running:
            try:
                frame = self.face_recognizer.stream_handler.get_frame()
                if frame is not None:
                    results, processed_frame = self.face_recognizer.recognize_frame(frame)
                    
                    for result in results:
                        if result['name'] == "Unknown":
                            snapshot = self.face_recognizer.capture_snapshot(frame)
                            self._log_event("face", {
                                'name': result['name'],
                                'confidence': result['confidence'],
                                'snapshot': snapshot
                            }, authorized=False)
                        else:
                            self._log_event("face", {
                                'name': result['name'],
                                'confidence': result['confidence']
                            }, authorized=True)
                            self._mark_attendance(person_name=result['name'])
                    
                    # Update dashboard
                    self.dashboard_data['face_feed'] = processed_frame
                    self.dashboard_data['system_status']['face_camera'] = self.face_recognizer.stream_handler.is_alive()
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error processing face camera: {str(e)}")
                time.sleep(1)
    
    def _process_plate_camera(self):
        """Process plate camera feed"""
        while self.running:
            try:
                frame = self.plate_recognizer.stream_handler.get_frame()
                if frame is not None:
                    results, processed_frame = self.plate_recognizer.recognize_plate(frame)
                    
                    for result in results:
                        if not result['is_known']:
                            snapshot = self.plate_recognizer.capture_snapshot(frame, result['plate'])
                            self._log_event("plate", {
                                'plate': result['plate'],
                                'confidence': result['confidence'],
                                'snapshot': snapshot
                            }, authorized=False)
                        else:
                            self._log_event("plate", {
                                'plate': result['plate'],
                                'confidence': result['confidence']
                            }, authorized=True)
                            self._mark_attendance(plate=result['plate'])
                    
                    # Update dashboard
                    self.dashboard_data['plate_feed'] = processed_frame
                    self.dashboard_data['system_status']['plate_camera'] = self.plate_recognizer.stream_handler.is_alive()
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error processing plate camera: {str(e)}")
                time.sleep(1)
    
    def _monitor_system(self):
        """Monitor system resources"""
        while self.running:
            try:
                # Update system status
                self.dashboard_data['system_status']['cpu_usage'] = psutil.cpu_percent()
                self.dashboard_data['system_status']['memory_usage'] = psutil.virtual_memory().percent
                self.dashboard_data['system_status']['uptime'] = time.time() - self.start_time
                
                # Update counts
                self.dashboard_data['authorized_count'] = sum(1 for log in self.dashboard_data['logs'] if log.get('authorized'))
                self.dashboard_data['unauthorized_count'] = sum(1 for log in self.dashboard_data['logs'] if not log.get('authorized'))
                
                # Sleep for 5 seconds
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error monitoring system: {str(e)}")
                time.sleep(5)
    
    def _log_event(self, event_type, data, authorized):
        """Log security event"""
        try:
            timestamp = datetime.now().isoformat()
            
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO logs (timestamp, type, data, authorized, snapshot_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                timestamp,
                event_type,
                json.dumps(data),
                1 if authorized else 0,
                data.get('snapshot', None)
            ))
            self.db_connection.commit()
            
            # Update dashboard
            log_entry = {
                'timestamp': timestamp,
                'type': event_type,
                'data': data,
                'authorized': authorized
            }
            self.dashboard_data['logs'].append(log_entry)
            
            # Keep only last 100 logs in memory
            if len(self.dashboard_data['logs']) > 100:
                self.dashboard_data['logs'] = self.dashboard_data['logs'][-100:]
            
            # Control barrier
            if authorized:
                self.hardware.open_barrier()
            else:
                self.hardware.close_barrier()
            
        except Exception as e:
            logger.error(f"Error logging event: {str(e)}")
    
    def _mark_attendance(self, person_name=None, plate=None):
        """Mark attendance for recognized person/plate"""
        try:
            timestamp = datetime.now().isoformat()
            
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO attendance (timestamp, person_name, plate_number)
                VALUES (?, ?, ?)
            ''', (timestamp, person_name, plate))
            self.db_connection.commit()
            
            # Update dashboard
            attendance_entry = {
                'timestamp': timestamp,
                'person_name': person_name,
                'plate': plate
            }
            self.dashboard_data['attendance'].append(attendance_entry)
            self.dashboard_data['today_count'] += 1
            
            # Keep only last 50 attendance records in memory
            if len(self.dashboard_data['attendance']) > 50:
                self.dashboard_data['attendance'] = self.dashboard_data['attendance'][-50:]
            
        except Exception as e:
            logger.error(f"Error marking attendance: {str(e)}")
    
    def get_dashboard_data(self):
        """Get current dashboard data"""
        return self.dashboard_data

class WebDashboard:
    """Web dashboard for the security system"""
    
    def __init__(self, security_system, host='0.0.0.0', port=5000):
        self.security_system = security_system
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.server = None
        self.server_thread = None
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template_string(DASHBOARD_TEMPLATE)
        
        @self.app.route('/dashboard_data')
        def dashboard_data():
            """Get dashboard data as JSON"""
            return jsonify(self.security_system.get_dashboard_data())
        
        @self.app.route('/face_feed')
        def face_feed():
            """Face camera video feed"""
            return self._generate_video_feed('face')
        
        @self.app.route('/plate_feed')
        def plate_feed():
            """Plate camera video feed"""
            return self._generate_video_feed('plate')
    
    def _generate_video_feed(self, camera_type):
        """Generate video feed for specified camera"""
        def generate():
            while True:
                frame = None
                if camera_type == 'face':
                    frame = self.security_system.dashboard_data['face_feed']
                elif camera_type == 'plate':
                    frame = self.security_system.dashboard_data['plate_feed']
                
                if frame is not None:
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                time.sleep(0.03)
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def start(self):
        """Start the web dashboard"""
        if self.server_thread and self.server_thread.is_alive():
            logger.warning("Web dashboard is already running")
            return
        
        logger.info(f"Starting web dashboard at http://{self.host}:{self.port}")
        self.server = make_server(self.host, self.port, self.app, threaded=True)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
    
    def stop(self):
        """Stop the web dashboard"""
        if self.server:
            logger.info("Stopping web dashboard...")
            self.server.shutdown()
            self.server_thread.join(timeout=1)
            self.server = None
            self.server_thread = None

# Dashboard HTML Template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Security System Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <style>
        :root {
            --primary-color: #2c3e50;
            --success-color: #27ae60;
            --danger-color: #e74c3c;
            --warning-color: #f39c12;
            --light-bg: #ecf0f1;
            --dark-bg: #34495e;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--light-bg);
            color: var(--primary-color);
        }
        
        .dashboard-header {
            background: linear-gradient(135deg, var(--primary-color), var(--dark-bg));
            color: white;
            padding: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .stat-card {
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
            margin-bottom: 20px;
            overflow: hidden;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-card-header {
            padding: 15px;
            background-color: var(--primary-color);
            color: white;
        }
        
        .stat-card-body {
            padding: 20px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .camera-feed {
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            margin-bottom: 20px;
            position: relative;
        }
        
        .camera-feed img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .camera-label {
            position: absolute;
            top: 10px;
            left: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        
        .barrier-status {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        .barrier-open {
            background-color: var(--success-color);
            color: white;
        }
        
        .barrier-closed {
            background-color: var(--danger-color);
            color: white;
        }
        
        .log-entry {
            padding: 10px;
            border-bottom: 1px solid #ddd;
            transition: background-color 0.3s ease;
        }
        
        .log-entry:hover {
            background-color: #f8f9fa;
        }
        
        .log-entry.authorized {
            border-left: 4px solid var(--success-color);
        }
        
        .log-entry.unauthorized {
            border-left: 4px solid var(--danger-color);
        }
        
        .attendance-entry {
            padding: 10px;
            border-bottom: 1px solid #ddd;
            display: flex;
            align-items: center;
        }
        
        .attendance-photo {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            margin-right: 15px;
            object-fit: cover;
        }
        
        .system-status {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        
        .status-item {
            text-align: center;
            margin: 10px;
        }
        
        .status-value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .status-label {
            color: #666;
        }
        
        .online {
            color: var(--success-color);
        }
        
        .offline {
            color: var(--danger-color);
        }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-md-6">
                    <h1 class="mb-0"><i class="bi bi-shield-lock-fill me-2"></i>AI Security System</h1>
                </div>
                <div class="col-md-6 text-end">
                    <div id="current-time" class="h4 mb-0"></div>
                </div>
            </div>
        </div>
    </div>

    <div class="container mt-4">
        <!-- Stats Row -->
        <div class="row">
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-card-header">
                        <h5 class="mb-0">Known Faces</h5>
                    </div>
                    <div class="stat-card-body">
                        <div class="stat-value" id="known-faces-count">0</div>
                        <i class="bi bi-people-fill fs-1 text-primary"></i>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-card-header">
                        <h5 class="mb-0">Today's Entries</h5>
                    </div>
                    <div class="stat-card-body">
                        <div class="stat-value" id="today-count">0</div>
                        <i class="bi bi-calendar-check-fill fs-1 text-success"></i>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-card-header">
                        <h5 class="mb-0">Authorized</h5>
                    </div>
                    <div class="stat-card-body">
                        <div class="stat-value" id="authorized-count">0</div>
                        <i class="bi bi-check-circle-fill fs-1 text-success"></i>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-card-header">
                        <h5 class="mb-0">Unauthorized</h5>
                    </div>
                    <div class="stat-card-body">
                        <div class="stat-value" id="unauthorized-count">0</div>
                        <i class="bi bi-x-circle-fill fs-1 text-danger"></i>
                    </div>
                </div>
            </div>
        </div>

        <!-- Camera Feeds Row -->
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="camera-feed">
                    <div class="camera-label">Face Recognition Camera</div>
                    <img id="face-feed" src="/face_feed" alt="Face Camera Feed">
                </div>
            </div>
            <div class="col-md-6">
                <div class="camera-feed">
                    <div class="camera-label">License Plate Camera</div>
                    <img id="plate-feed" src="/plate_feed" alt="Plate Camera Feed">
                </div>
            </div>
        </div>

        <!-- System Status Row -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">System Status</h5>
                    </div>
                    <div class="card-body">
                        <div class="system-status">
                            <div class="status-item">
                                <div class="status-value" id="face-camera-status">
                                    <i class="bi bi-circle-fill offline"></i> Offline
                                </div>
                                <div class="status-label">Face Camera</div>
                            </div>
                            <div class="status-item">
                                <div class="status-value" id="plate-camera-status">
                                    <i class="bi bi-circle-fill offline"></i> Offline
                                </div>
                                <div class="status-label">Plate Camera</div>
                            </div>
                            <div class="status-item">
                                <div class="status-value" id="cpu-usage">0%</div>
                                <div class="status-label">CPU Usage</div>
                            </div>
                            <div class="status-item">
                                <div class="status-value" id="memory-usage">0%</div>
                                <div class="status-label">Memory Usage</div>
                            </div>
                            <div class="status-item">
                                <div class="status-value" id="uptime">0s</div>
                                <div class="status-label">Uptime</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Barrier Status Row -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="barrier-status barrier-closed" id="barrier-status">
                    <i class="bi bi-shield-lock-fill fs-1 me-3"></i>
                    <h3 class="mb-0">Barrier: CLOSED</h3>
                </div>
            </div>
        </div>

        <!-- Logs and Attendance Tabs -->
        <div class="row mt-4">
            <div class="col-12">
                <ul class="nav nav-tabs" id="logsTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="logs-tab" data-bs-toggle="tab" data-bs-target="#logs" type="button" role="tab" aria-controls="logs" aria-selected="true">Activity Logs</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="attendance-tab" data-bs-toggle="tab" data-bs-target="#attendance" type="button" role="tab" aria-controls="attendance" aria-selected="false">Attendance</button>
                    </li>
                </ul>
                <div class="tab-content" id="logsTabsContent">
                    <div class="tab-pane fade show active" id="logs" role="tabpanel" aria-labelledby="logs-tab">
                        <div class="card mt-3">
                            <div class="card-body">
                                <div id="logs-container" style="max-height: 400px; overflow-y: auto;">
                                    <!-- Logs will be populated here -->
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="tab-pane fade" id="attendance" role="tabpanel" aria-labelledby="attendance-tab">
                        <div class="card mt-3">
                            <div class="card-body">
                                <div id="attendance-container" style="max-height: 400px; overflow-y: auto;">
                                    <!-- Attendance will be populated here -->
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Update current time
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleString();
        }
        updateTime();
        setInterval(updateTime, 1000);
        
        // Format uptime
        function formatUptime(seconds) {
            const days = Math.floor(seconds / (3600 * 24));
            const hours = Math.floor((seconds % (3600 * 24)) / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            
            if (days > 0) {
                return `${days}d ${hours}h ${minutes}m`;
            } else if (hours > 0) {
                return `${hours}h ${minutes}m ${secs}s`;
            } else if (minutes > 0) {
                return `${minutes}m ${secs}s`;
            } else {
                return `${secs}s`;
            }
        }
        
        // Update dashboard data
        function updateDashboard() {
            fetch('/dashboard_data')
                .then(response => response.json())
                .then(data => {
                    // Update stats
                    document.getElementById('known-faces-count').textContent = data.known_faces_count;
                    document.getElementById('today-count').textContent = data.today_count;
                    document.getElementById('authorized-count').textContent = data.authorized_count;
                    document.getElementById('unauthorized-count').textContent = data.unauthorized_count;
                    
                    // Update system status
                    const faceStatus = document.getElementById('face-camera-status');
                    faceStatus.innerHTML = data.system_status.face_camera ? 
                        '<i class="bi bi-circle-fill online"></i> Online' : 
                        '<i class="bi bi-circle-fill offline"></i> Offline';
                    
                    const plateStatus = document.getElementById('plate-camera-status');
                    plateStatus.innerHTML = data.system_status.plate_camera ? 
                        '<i class="bi bi-circle-fill online"></i> Online' : 
                        '<i class="bi bi-circle-fill offline"></i> Offline';
                    
                    document.getElementById('cpu-usage').textContent = data.system_status.cpu_usage + '%';
                    document.getElementById('memory-usage').textContent = data.system_status.memory_usage + '%';
                    document.getElementById('uptime').textContent = formatUptime(data.system_status.uptime);
                    
                    // Update barrier status
                    const barrierStatus = document.getElementById('barrier-status');
                    if (data.authorized_count > data.unauthorized_count) {
                        barrierStatus.className = 'barrier-status barrier-open';
                        barrierStatus.innerHTML = '<i class="bi bi-shield-check-fill fs-1 me-3"></i><h3 class="mb-0">Barrier: OPEN</h3>';
                    } else {
                        barrierStatus.className = 'barrier-status barrier-closed';
                        barrierStatus.innerHTML = '<i class="bi bi-shield-lock-fill fs-1 me-3"></i><h3 class="mb-0">Barrier: CLOSED</h3>';
                    }
                    
                    // Update logs
                    const logsContainer = document.getElementById('logs-container');
                    logsContainer.innerHTML = '';
                    data.logs.slice(-20).reverse().forEach(log => {
                        const logEntry = document.createElement('div');
                        logEntry.className = `log-entry ${log.authorized ? 'authorized' : 'unauthorized'}`;
                        
                        const time = new Date(log.timestamp).toLocaleTimeString();
                        const details = log.type === 'face' ? 
                            `Face: ${log.data.name} (${(log.data.confidence * 100).toFixed(1)}%)` : 
                            `Plate: ${log.data.plate} (${(log.data.confidence * 100).toFixed(1)}%)`;
                        
                        logEntry.innerHTML = `
                            <div class="d-flex justify-content-between">
                                <div>
                                    <strong>${log.type.toUpperCase()}</strong>: ${details}
                                </div>
                                <div class="text-muted">${time}</div>
                            </div>
                        `;
                        logsContainer.appendChild(logEntry);
                    });
                    
                    // Update attendance
                    const attendanceContainer = document.getElementById('attendance-container');
                    attendanceContainer.innerHTML = '';
                    data.attendance.slice(-10).reverse().forEach(entry => {
                        const attendanceEntry = document.createElement('div');
                        attendanceEntry.className = 'attendance-entry';
                        
                        const time = new Date(entry.timestamp).toLocaleTimeString();
                        const name = entry.person_name || 'Unknown';
                        const plate = entry.plate || 'N/A';
                        
                        attendanceEntry.innerHTML = `
                            <img src="https://picsum.photos/seed/${name}/50/50.jpg" alt="${name}" class="attendance-photo">
                            <div>
                                <strong>${name}</strong>
                                <div class="text-muted">Plate: ${plate} | ${time}</div>
                            </div>
                        `;
                        attendanceContainer.appendChild(attendanceEntry);
                    });
                })
                .catch(error => {
                    console.error('Error fetching dashboard data:', error);
                });
        }
        
        // Initial update
        updateDashboard();
        
        // Update every 2 seconds
        setInterval(updateDashboard, 2000);
    </script>
</body>
</html>
"""

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received")
    if security_system:
        security_system.stop()
    if web_dashboard:
        web_dashboard.stop()
    sys.exit(0)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AI Security System with Face and License Plate Recognition')
    parser.add_argument('--config', type=str, default='config.py', help='Path to configuration file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web dashboard host')
    parser.add_argument('--port', type=int, default=5000, help='Web dashboard port')
    return parser.parse_args()

def main():
    """Main entry point"""
    global security_system, web_dashboard
    
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)
    
    # Initialize security system
    try:
        security_system = SecuritySystem(config)
        security_system.start()
    except Exception as e:
        logger.error(f"Failed to start security system: {str(e)}")
        sys.exit(1)
    
    # Initialize web dashboard
    try:
        web_dashboard = WebDashboard(security_system, host=args.host, port=args.port)
        web_dashboard.start()
    except Exception as e:
        logger.error(f"Failed to start web dashboard: {str(e)}")
        security_system.stop()
        sys.exit(1)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("AI Security System is running. Press Ctrl+C to stop.")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        security_system.stop()
        web_dashboard.stop()

if __name__ == "__main__":
    import importlib.util
    security_system = None
    web_dashboard = None
    main()