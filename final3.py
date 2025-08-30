import sys
import cv2
import numpy as np
import time
import threading
import queue
import os
from datetime import datetime
from ultralytics import YOLO
import pygame
import json
import signal
from typing import List, Tuple, Dict, Optional, Any

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QGridLayout, QScrollArea, QFrame, 
                            QTabWidget, QSplitter, QTextEdit, QStatusBar, QMenuBar, 
                            QAction, QFileDialog, QMessageBox, QComboBox, QSpinBox, 
                            QCheckBox, QGroupBox, QFormLayout, QSlider, QProgressBar,
                            QDialog, QLineEdit, QListWidget, QListWidgetItem, QDialogButtonBox,
                            QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont, QIcon, QCursor
from PyQt5.QtGui import QPalette


# Import the existing classes from the previous implementation
# (We'll keep the camera processing logic but replace the GUI with PyQt5)

class VideoInputHandler:
    def __init__(self, source: str = None, max_reconnect_attempts=5, reconnect_delay=5):
        """Initialize video input handler with RTSP or webcam source."""
        self.source = source if source else 0
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.is_opened = False
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.reconnect_attempts = 0
        
    def initialize(self) -> bool:
        """Initialize video capture."""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                print(f"Error: Could not open video source {self.source}")
                return False
                
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps == 0:
                self.fps = 30  # Default if unable to get FPS
                
            self.is_opened = True
            self.reconnect_attempts = 0
            print(f"Video source initialized: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS")
            return True
        except Exception as e:
            print(f"Error initializing video source: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video source with reconnection logic."""
        if not self.is_opened or not self.cap:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            print("Frame read failed, attempting to reconnect...")
            self.reconnect_attempts += 1
            
            if self.reconnect_attempts <= self.max_reconnect_attempts:
                print(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
                self.cap.release()
                self.is_opened = False
                
                # Wait before reconnecting
                time.sleep(self.reconnect_delay)
                
                if self.initialize():
                    ret, frame = self.cap.read()
                    if ret:
                        print("Reconnected successfully")
                        self.reconnect_attempts = 0
                        return ret, frame
                    else:
                        print("Reconnected but frame read failed")
                else:
                    print("Reconnection failed")
            else:
                print(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
                return False, None
        
        return ret, frame if ret else None
    
    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            self.is_opened = False

class DetectionEngine:
    def __init__(self, model_path: str = "yolov8n.pt", detection_size: Tuple[int, int] = (640, 360), 
                 frame_skip: int = 2, use_threading: bool = True):
        """Initialize detection engine with YOLOv8."""
        self.model_path = model_path
        self.detection_size = detection_size
        self.frame_skip = frame_skip
        self.use_threading = use_threading
        self.model = None
        self.frame_count = 0
        self.detection_queue = queue.Queue(maxsize=10)
        self.detection_thread = None
        self.stop_thread = False
        self.latest_detections = []
        self.detection_lock = threading.Lock()
        
    def initialize(self) -> bool:
        """Initialize YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            print(f"YOLO model loaded: {self.model_path}")
            
            if self.use_threading:
                self.detection_thread = threading.Thread(target=self._detection_worker)
                self.detection_thread.daemon = True
                self.detection_thread.start()
                print("Detection thread started")
                
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False
    
    def _detection_worker(self):
        """Worker function for threaded detection."""
        while not self.stop_thread:
            try:
                if not self.detection_queue.empty():
                    frame, frame_id = self.detection_queue.get(timeout=0.1)
                    detections = self._detect_objects(frame)
                    with self.detection_lock:
                        self.latest_detections = detections
                    self.detection_queue.task_done()
                else:
                    time.sleep(0.01)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in detection thread: {e}")
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in frame using YOLO model."""
        try:
            # Resize frame for detection
            resized_frame = cv2.resize(frame, self.detection_size)
            
            # Run detection
            results = self.model(resized_frame, verbose=False)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID (0 for person in COCO dataset)
                        cls_id = int(box.cls[0])
                        if cls_id == 0:  # Only detect persons
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Scale coordinates back to original frame size
                            x1 = int(x1 * frame.shape[1] / self.detection_size[0])
                            y1 = int(y1 * frame.shape[0] / self.detection_size[1])
                            x2 = int(x2 * frame.shape[1] / self.detection_size[0])
                            y2 = int(y2 * frame.shape[0] / self.detection_size[1])
                            
                            # Calculate centroid
                            centroid_x = (x1 + x2) // 2
                            centroid_y = (y1 + y2) // 2
                            
                            # Get confidence
                            conf = float(box.conf[0])
                            
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'centroid': (centroid_x, centroid_y),
                                'confidence': conf
                            })
            
            return detections
        except Exception as e:
            print(f"Error in detection: {e}")
            return []
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect persons in frame, with frame skipping."""
        self.frame_count += 1
        
        # Skip frames based on frame_skip setting
        if self.frame_count % self.frame_skip != 0:
            return self.latest_detections  # Return latest detections while skipping
        
        if self.use_threading:
            # Add frame to detection queue
            if self.detection_queue.full():
                # If queue is full, skip this frame
                return self.latest_detections
            
            frame_id = self.frame_count
            self.detection_queue.put((frame.copy(), frame_id))
            return self.latest_detections  # Return latest detections while processing new one
        else:
            # Direct detection
            detections = self._detect_objects(frame)
            with self.detection_lock:
                self.latest_detections = detections
            return detections
    
    def release(self):
        """Release detection resources."""
        if self.use_threading and self.detection_thread:
            self.stop_thread = True
            self.detection_thread.join(timeout=1.0)

class PolygonZoneDetector:
    def __init__(self, config_file: str = "zone_config.json", cooldown: int = 3):
        """Initialize polygon zone detector."""
        self.config_file = config_file
        self.zones = []  # List of polygon zones
        self.current_zone = []  # Points of the zone being drawn
        self.drawing = False  # Flag to indicate if we're currently drawing a zone
        self.cooldown = cooldown
        self.last_alert_time = 0
        self.person_in_zone = False  # Track if someone is currently in any zone
        
    def load_zones(self) -> bool:
        """Load zones from configuration file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.zones = data.get('zones', [])
                print(f"Loaded {len(self.zones)} zones from {self.config_file}")
                return True
            else:
                print(f"Configuration file {self.config_file} not found. Starting with empty zones.")
                return False
        except Exception as e:
            print(f"Error loading zones: {e}")
            return False
    
    def save_zones(self) -> bool:
        """Save zones to configuration file."""
        try:
            data = {'zones': self.zones}
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(self.zones)} zones to {self.config_file}")
            return True
        except Exception as e:
            print(f"Error saving zones: {e}")
            return False
    
    def start_drawing(self, point: Tuple[int, int]):
        """Start drawing a new zone."""
        self.current_zone = [point]
        self.drawing = True
    
    def add_point(self, point: Tuple[int, int]):
        """Add a point to the current zone being drawn."""
        if self.drawing:
            self.current_zone.append(point)
    
    def finish_drawing(self) -> bool:
        """Finish drawing the current zone and add it to the zones list."""
        if self.drawing and len(self.current_zone) >= 3:  # Need at least 3 points for a polygon
            self.zones.append(self.current_zone)
            self.current_zone = []
            self.drawing = False
            return True
        else:
            self.current_zone = []
            self.drawing = False
            return False
    
    def cancel_drawing(self):
        """Cancel drawing the current zone."""
        self.current_zone = []
        self.drawing = False
    
    def clear_zones(self):
        """Clear all zones."""
        self.zones = []
        self.current_zone = []
        self.drawing = False
    
    def add_zone_from_coordinates(self, coordinates: List[Tuple[int, int]]) -> bool:
        """Add a zone from a list of coordinates."""
        if len(coordinates) >= 3:  # Need at least 3 points for a polygon
            self.zones.append(coordinates)
            return self.save_zones()
        return False
    
    def is_point_in_zone(self, point: Tuple[int, int]) -> int:
        """Check if a point is inside any zone. Returns zone index or -1 if not in any zone."""
        for i, zone in enumerate(self.zones):
            if self._point_in_polygon(point, zone):
                return i
        return -1
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def check_intrusion(self, detections: List[Dict]) -> Tuple[bool, int]:
        """Check if any person is in any danger zone. Returns (alert_triggered, zone_index)."""
        current_time = time.time()
        if current_time - self.last_alert_time < self.cooldown:
            return False, -1
        
        person_in_zone = False
        zone_idx = -1
        
        for detection in detections:
            centroid = detection['centroid']
            zone_idx = self.is_point_in_zone(centroid)
            if zone_idx != -1:
                person_in_zone = True
                break
        
        # If someone is in the zone and we haven't recently alerted
        if person_in_zone and not self.person_in_zone:
            self.last_alert_time = current_time
            self.person_in_zone = True
            return True, zone_idx
        
        # Update the zone status
        self.person_in_zone = person_in_zone
        return False, -1

class AlertSystem:
    def __init__(self, alert_dir: str = "alerts", camera_id: str = ""):
        """Initialize alert system for sounds and logging."""
        self.camera_id = camera_id
        self.alert_dir = os.path.join(alert_dir, f"camera_{camera_id}")
        pygame.mixer.init()
        self.siren_sound = None
        self.siren_playing = False
        self.siren_thread = None
        self.stop_siren_event = threading.Event()
        self.log_file = None
        
        # Create alert directory if it doesn't exist
        os.makedirs(self.alert_dir, exist_ok=True)
        
        # Initialize log file
        log_path = os.path.join(self.alert_dir, "alert_log.txt")
        self.log_file = open(log_path, 'a')
        
        # Load sounds (create siren if files don't exist)
        self._load_sounds()
    
    def _load_sounds(self):
        """Load alert sounds or create siren."""
        try:
            # Create a siren sound
            self.siren_sound = self._create_siren()
                
        except Exception as e:
            print(f"Error loading sounds: {e}")
    
    def _create_siren(self) -> pygame.mixer.Sound:
        """Create a siren sound."""
        sample_rate = 22050
        duration = 2.0  # 2 seconds for a full siren cycle
        
        # Create a siren with rising and falling pitch
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create a frequency sweep from 600Hz to 1200Hz and back
        frequency = 600 + 300 * np.sin(2 * np.pi * t / duration)
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope to make it sound more natural
        envelope = np.ones_like(t)
        attack = int(0.1 * sample_rate)  # 100ms attack
        release = int(0.1 * sample_rate)   # 100ms release
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        wave = wave * envelope
        
        # Convert to 16-bit PCM
        wave = (wave * 32767).astype(np.int16)
        
        # Create stereo sound
        stereo_wave = np.zeros((len(wave), 2), dtype=np.int16)
        stereo_wave[:, 0] = wave
        stereo_wave[:, 1] = wave
        
        return pygame.sndarray.make_sound(stereo_wave)
    
    def start_siren(self):
        """Start playing the siren in a separate thread."""
        if not self.siren_playing:
            self.stop_siren_event.clear()
            self.siren_thread = threading.Thread(target=self._play_siren_loop)
            self.siren_thread.daemon = True
            self.siren_thread.start()
            self.siren_playing = True
            print(f"Camera {self.camera_id}: Siren started")
    
    def stop_siren(self):
        """Stop playing the siren."""
        if self.siren_playing:
            self.stop_siren_event.set()
            self.siren_thread.join(timeout=1.0)
            self.siren_playing = False
            print(f"Camera {self.camera_id}: Siren stopped")
    
    def _play_siren_loop(self):
        """Loop to play the siren continuously."""
        while not self.stop_siren_event.is_set():
            self.siren_sound.play()
            # Wait for the sound to finish playing
            pygame.time.wait(int(self.siren_sound.get_length() * 1000))
            # Check if we should stop
            if self.stop_siren_event.is_set():
                break
    
    def trigger_zone_alert(self, frame: np.ndarray, zone_idx: int):
        """Trigger alert for person in danger zone."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start siren
        self.start_siren()
        
        # Save snapshot
        snapshot_path = os.path.join(self.alert_dir, f"zone_alert_{timestamp.replace(':', '-')}.jpg")
        cv2.imwrite(snapshot_path, frame)
        
        # Log alert
        log_entry = f"[{timestamp}] Camera {self.camera_id}: Person detected in zone {zone_idx}. Siren activated. Snapshot saved to {snapshot_path}\n"
        self.log_file.write(log_entry)
        self.log_file.flush()
        
        print(f"Camera {self.camera_id}: Zone Alert: Person detected in zone {zone_idx} - Siren activated")
        
        return {
            'timestamp': timestamp,
            'camera_id': self.camera_id,
            'zone_idx': zone_idx,
            'snapshot_path': snapshot_path
        }
    
    def release(self):
        """Release alert system resources."""
        if self.siren_playing:
            self.stop_siren()
        if self.log_file:
            self.log_file.close()
        pygame.mixer.quit()

class Camera:
    def __init__(self, camera_id: str, source: str, config_file: str = None, 
                 detection_size: Tuple[int, int] = (640, 360), frame_skip: int = 2, 
                 use_threading: bool = True):
        """Initialize camera with its own components."""
        self.camera_id = camera_id
        self.running = False
        self.siren_active = False
        self.zone_clear_time = 0
        self.siren_stop_delay = 5  # seconds to wait after zone clear before stopping siren
        
        # Initialize components
        self.video_handler = VideoInputHandler(source)
        self.detection_engine = DetectionEngine("yolov8n.pt", detection_size, frame_skip, use_threading)
        
        # Use camera-specific config file if not provided
        if config_file is None:
            config_file = f"zone_config_camera_{camera_id}.json"
        self.zone_detector = PolygonZoneDetector(config_file)
        self.alert_system = AlertSystem(camera_id=camera_id)
        
        # FPS calculation
        self.frame_times = []
        self.fps_update_interval = 1.0  # Update FPS display every second
        self.last_fps_update = time.time()
        self.current_fps = 0
        
        # Thread for this camera
        self.thread = None
        
        # Queue for communication with GUI thread
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Current frame and data for GUI
        self.current_frame = None
        self.current_detections = []
        self.current_fps = 0
        self.siren_active = False
        
    def initialize(self) -> bool:
        """Initialize all components for this camera."""
        print(f"Initializing Camera {self.camera_id}...")
        
        # Initialize video handler
        if not self.video_handler.initialize():
            return False
        
        # Initialize detection engine
        if not self.detection_engine.initialize():
            return False
        
        # Initialize zone detector and load existing zones
        self.zone_detector.load_zones()
        
        print(f"Camera {self.camera_id} initialization complete")
        return True
    
    def run(self):
        """Run the main application loop for this camera."""
        self.running = True
        print(f"Camera {self.camera_id}: Starting application loop.")
        print(f"Camera {self.camera_id}: System will run 24/7. Siren will activate when someone enters a danger zone.")
        
        while self.running:
            start_time = time.time()
            
            # Read frame
            ret, frame = self.video_handler.read_frame()
            if not ret:
                print(f"Camera {self.camera_id}: Error reading frame. Retrying...")
                time.sleep(1)  # Wait before retrying
                continue
            
            # Run detection
            detections = self.detection_engine.detect(frame)
            
            # Check for zone intrusion
            alert_triggered, zone_idx = self.zone_detector.check_intrusion(detections)
            alert_data = None
            if alert_triggered:
                alert_data = self.alert_system.trigger_zone_alert(frame, zone_idx)
                self.siren_active = True
                self.zone_clear_time = 0  # Reset zone clear time
            
            # Check if zone is clear and siren should be stopped
            if self.siren_active and not self.zone_detector.person_in_zone:
                if self.zone_clear_time == 0:
                    # Zone just became clear, record the time
                    self.zone_clear_time = time.time()
                elif time.time() - self.zone_clear_time > self.siren_stop_delay:
                    # Zone has been clear for the delay period, stop siren
                    self.alert_system.stop_siren()
                    self.siren_active = False
                    self.zone_clear_time = 0
            
            # Calculate FPS
            self._update_fps(start_time)
            
            # Update current frame data for GUI
            self.current_frame = frame
            self.current_detections = detections
            self.current_fps = self.current_fps
            self.siren_active = self.siren_active
            
            # Put frame data in queue for GUI
            frame_data = {
                'camera_id': self.camera_id,
                'frame': frame,
                'detections': detections,
                'fps': self.current_fps,
                'siren_active': self.siren_active,
                'zones': self.zone_detector.zones,
                'current_zone': self.zone_detector.current_zone,
                'drawing': self.zone_detector.drawing,
                'alert_data': alert_data
            }
            
            # Put frame in queue for GUI thread
            try:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_data, block=False)
            except queue.Full:
                # If queue is full, skip this frame
                pass
        
        print(f"Camera {self.camera_id}: Application loop ended")
    
    def _update_fps(self, frame_start_time: float):
        """Update FPS calculation."""
        current_time = time.time()
        self.frame_times.append(current_time - frame_start_time)
        
        # Remove old frame times
        while self.frame_times and self.frame_times[0] < current_time - self.fps_update_interval:
            self.frame_times.pop(0)
        
        # Update FPS display at intervals
        if current_time - self.last_fps_update >= self.fps_update_interval:
            if self.frame_times:
                self.current_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            else:
                self.current_fps = 0.0
            self.last_fps_update = current_time
    
    def start_thread(self):
        """Start the camera in a separate thread."""
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        print(f"Camera {self.camera_id}: Thread started")
    
    def release(self):
        """Release all resources for this camera."""
        print(f"Camera {self.camera_id}: Releasing resources...")
        
        # Stop siren if playing
        if self.siren_active:
            self.alert_system.stop_siren()
        
        self.running = False
        self.video_handler.release()
        self.detection_engine.release()
        self.alert_system.release()
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        print(f"Camera {self.camera_id}: Resources released")

# New Dialog for editing Camera 1 zones
class CameraZoneEditorDialog(QDialog):
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.current_frame = None
        self.drawing = False
        self.current_zone = []
        self.zones = []
        self.setWindowTitle(f"Edit Camera {camera.camera_id} Danger Zones")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: white;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QTableWidget {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                gridline-color: #555;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #4a4a4a;
                color: white;
                padding: 5px;
                border: 1px solid #555;
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "Edit danger zones for Camera 1. Click on the video to add points to a zone. "
            "Double-click to finish the zone. Right-click to cancel."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #aaaaaa; font-size: 12px; padding: 10px;")
        layout.addWidget(instructions)
        
        # Create splitter for video and zones list
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Video widget
        self.video_widget = QWidget()
        self.video_widget.setMinimumSize(400, 300)
        self.video_widget.setMouseTracking(True)
        self.video_layout = QVBoxLayout(self.video_widget)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(400, 300)
        self.video_label.setMouseTracking(True)
        self.video_layout.addWidget(self.video_label)
        
        # Zone list
        self.zone_widget = QWidget()
        self.zone_layout = QVBoxLayout(self.zone_widget)
        
        self.zone_table = QTableWidget()
        self.zone_table.setColumnCount(3)
        self.zone_table.setHorizontalHeaderLabels(["Zone ID", "Points", "Actions"])
        self.zone_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.zone_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.zone_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.zone_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.zone_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.zone_layout.addWidget(self.zone_table)
        
        # Zone buttons
        zone_button_layout = QHBoxLayout()
        
        self.add_coord_button = QPushButton("Add with Coordinates")
        self.add_coord_button.clicked.connect(self.add_with_coordinates)
        zone_button_layout.addWidget(self.add_coord_button)
        
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.remove_zone)
        zone_button_layout.addWidget(self.remove_button)
        
        self.zone_layout.addLayout(zone_button_layout)
        
        # Add widgets to splitter
        splitter.addWidget(self.video_widget)
        splitter.addWidget(self.zone_widget)
        
        # Set splitter sizes
        splitter.setSizes([500, 300])
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Set up mouse events
        self.video_label.mousePressEvent = self.video_mouse_press_event
        self.video_label.mouseDoubleClickEvent = self.video_mouse_double_click_event
        
        # Load zones
        self.load_zones()
        
        # Set up timer to update video
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)  # Update at ~33 FPS
    
    def load_zones(self):
        # Clear table
        self.zone_table.setRowCount(0)
        self.zones = self.camera.zone_detector.zones.copy()
        
        # Add zones to table
        for i, zone in enumerate(self.zones):
            row_position = self.zone_table.rowCount()
            self.zone_table.insertRow(row_position)
            
            # Zone ID
            self.zone_table.setItem(row_position, 0, QTableWidgetItem(str(i + 1)))
            
            # Points count
            self.zone_table.setItem(row_position, 1, QTableWidgetItem(f"{len(zone)} points"))
            
            # Actions button
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            
            edit_button = QPushButton("Edit")
            edit_button.setProperty("zone_index", i)
            edit_button.clicked.connect(self.edit_zone)
            actions_layout.addWidget(edit_button)
            
            actions_layout.addStretch()
            
            self.zone_table.setCellWidget(row_position, 2, actions_widget)
    
    def update_video(self):
        # Get frame from camera
        try:
            frame_data = self.camera.frame_queue.get(block=False)
            self.current_frame = frame_data['frame']
            
            # Process frame for display
            display_frame = self.current_frame.copy()
            
            # Draw danger zones
            for zone in self.zones:
                # Draw filled polygon with transparency
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [np.array(zone, dtype=np.int32)], (0, 0, 255))
                alpha = 0.2  # Transparency factor
                display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)
                
                # Draw polygon outline
                cv2.polylines(display_frame, [np.array(zone, dtype=np.int32)], True, (0, 0, 255), 2)
                
                # Add zone label
                if len(zone) > 0:
                    # Calculate centroid of the polygon for label placement
                    M = cv2.moments(np.array(zone, dtype=np.int32))
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(display_frame, "DANGER ZONE", (cX - 60, cY), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw current zone being drawn
            if self.drawing and len(self.current_zone) > 0:
                # Draw lines between points
                for i in range(len(self.current_zone) - 1):
                    cv2.line(display_frame, self.current_zone[i], 
                            self.current_zone[i+1], (0, 255, 255), 2)
                
                # Draw points
                for point in self.current_zone:
                    cv2.circle(display_frame, point, 3, (0, 255, 255), -1)
            
            # Convert to RGB for Qt
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale image to fit the label
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Draw on the pixmap if needed
            if self.drawing and len(self.current_zone) > 0:
                painter = QPainter(scaled_pixmap)
                painter.setPen(QPen(QColor(0, 255, 255), 2))
                
                # Scale the current zone points to the pixmap size
                label_w, label_h = self.video_label.width(), self.video_label.height()
                frame_h, frame_w = self.current_frame.shape[:2]
                
                scale_x = label_w / frame_w
                scale_y = label_h / frame_h
                
                scaled_zone = [(int(x * scale_x), int(y * scale_y)) for x, y in self.current_zone]
                
                # Draw lines between points
                for i in range(len(scaled_zone) - 1):
                    painter.drawLine(scaled_zone[i][0], scaled_zone[i][1], 
                                   scaled_zone[i+1][0], scaled_zone[i+1][1])
                
                # Draw points
                for point in scaled_zone:
                    painter.drawEllipse(point[0] - 3, point[1] - 3, 6, 6)
                
                painter.end()
            
            self.video_label.setPixmap(scaled_pixmap)
            
        except queue.Empty:
            # No frame available
            pass
    
    def video_mouse_press_event(self, event):
        if self.current_frame is None:
            return
            
        # Calculate the position in the original frame
        frame_h, frame_w = self.current_frame.shape[:2]
        label_w, label_h = self.video_label.width(), self.video_label.height()
        
        # Calculate scaling factors
        scale_x = frame_w / label_w
        scale_y = frame_h / label_h
        
        # Calculate position in original frame
        frame_x = int(event.x() * scale_x)
        frame_y = int(event.y() * scale_y)
        
        # Handle mouse events
        if event.button() == Qt.LeftButton:
            if not self.drawing:
                self.drawing = True
                self.current_zone = [(frame_x, frame_y)]
                print(f"Started drawing zone at ({frame_x}, {frame_y})")
            else:
                self.current_zone.append((frame_x, frame_y))
                print(f"Added point ({frame_x}, {frame_y}) to current zone")
                
        elif event.button() == Qt.RightButton:
            if self.drawing:
                self.drawing = False
                self.current_zone = []
                print("Zone drawing cancelled")
    
    def video_mouse_double_click_event(self, event):
        if not self.drawing:
            return
            
        # Finish drawing the zone
        if len(self.current_zone) >= 3:
            self.zones.append(self.current_zone)
            self.load_zones()
            print(f"Zone added with {len(self.current_zone)} points")
        
        self.drawing = False
        self.current_zone = []
    
    def add_with_coordinates(self):
        # Open coordinate input dialog
        dialog = CoordinateInputDialog(self.camera.camera_id, self)
        if dialog.exec_() == QDialog.Accepted:
            coordinates = dialog.get_coordinates()
            if len(coordinates) >= 3:  # Need at least 3 points for a polygon
                self.zones.append(coordinates)
                self.load_zones()
    
    def edit_zone(self):
        # Get the zone index
        button = self.sender()
        zone_index = button.property("zone_index")
        
        # Get the zone coordinates
        zone = self.zones[zone_index]
        
        # Open coordinate input dialog with existing coordinates
        dialog = CoordinateInputDialog(self.camera.camera_id, self)
        dialog.coordinates = zone.copy()
        
        # Populate the table
        for x, y in zone:
            row_position = dialog.coord_table.rowCount()
            dialog.coord_table.insertRow(row_position)
            dialog.coord_table.setItem(row_position, 0, QTableWidgetItem(str(x)))
            dialog.coord_table.setItem(row_position, 1, QTableWidgetItem(str(y)))
        
        if dialog.exec_() == QDialog.Accepted:
            coordinates = dialog.get_coordinates()
            if len(coordinates) >= 3:  # Need at least 3 points for a polygon
                # Update zone
                self.zones[zone_index] = coordinates
                self.load_zones()
    
    def remove_zone(self):
        selected_items = self.zone_table.selectedItems()
        if not selected_items:
            return
        
        row = selected_items[0].row()
        zone_index = row
        
        # Remove zone
        self.zones.pop(zone_index)
        self.load_zones()
    
    def accept(self):
        # Save zones to camera
        self.camera.zone_detector.zones = self.zones
        self.camera.zone_detector.save_zones()
        
        # Stop timer
        self.timer.stop()
        
        super().accept()
    
    def reject(self):
        # Stop timer
        self.timer.stop()
        
        super().reject()

# New Dialog for coordinate input
class CoordinateInputDialog(QDialog):
    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.coordinates = []
        self.setWindowTitle(f"Enter Coordinates - Camera {camera_id}")
        self.setMinimumSize(400, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: white;
            }
            QLabel {
                color: white;
            }
            QLineEdit {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QListWidget {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
            }
            QTableWidget {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                gridline-color: #555;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #4a4a4a;
                color: white;
                padding: 5px;
                border: 1px solid #555;
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Enter coordinates for the danger zone polygon. Each point requires X and Y coordinates.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #aaaaaa; font-size: 12px; padding: 10px;")
        layout.addWidget(instructions)
        
        # Coordinate input form
        form_layout = QFormLayout()
        
        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("X coordinate")
        form_layout.addRow("X:", self.x_input)
        
        self.y_input = QLineEdit()
        self.y_input.setPlaceholderText("Y coordinate")
        form_layout.addRow("Y:", self.y_input)
        
        layout.addLayout(form_layout)
        
        # Buttons for adding and removing points
        button_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add Point")
        self.add_button.clicked.connect(self.add_point)
        button_layout.addWidget(self.add_button)
        
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.remove_point)
        button_layout.addWidget(self.remove_button)
        
        layout.addLayout(button_layout)
        
        # Table to display coordinates
        self.coord_table = QTableWidget()
        self.coord_table.setColumnCount(2)
        self.coord_table.setHorizontalHeaderLabels(["X", "Y"])
        self.coord_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.coord_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.coord_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.coord_table)
        
        # Buttons for dialog
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def add_point(self):
        try:
            x = int(self.x_input.text())
            y = int(self.y_input.text())
            self.coordinates.append((x, y))
            
            # Add to table
            row_position = self.coord_table.rowCount()
            self.coord_table.insertRow(row_position)
            self.coord_table.setItem(row_position, 0, QTableWidgetItem(str(x)))
            self.coord_table.setItem(row_position, 1, QTableWidgetItem(str(y)))
            
            # Clear inputs
            self.x_input.clear()
            self.y_input.clear()
            
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integer coordinates.")
    
    def remove_point(self):
        selected_rows = self.coord_table.selectedItems()
        if selected_rows:
            row = selected_rows[0].row()
            self.coord_table.removeRow(row)
            self.coordinates.pop(row)
    
    def get_coordinates(self):
        return self.coordinates

# New Dialog for managing zones
class ZoneManagerDialog(QDialog):
    zone_added = pyqtSignal(str, list)  # camera_id, coordinates
    
    def __init__(self, cameras, parent=None):
        super().__init__(parent)
        self.cameras = cameras
        self.setWindowTitle("Zone Manager")
        self.setMinimumSize(600, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: white;
            }
            QLabel {
                color: white;
            }
            QComboBox {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QListWidget {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
            }
            QTableWidget {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                gridline-color: #555;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #4a4a4a;
                color: white;
                padding: 5px;
                border: 1px solid #555;
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Camera selection
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        
        self.camera_combo = QComboBox()
        for camera in self.cameras:
            self.camera_combo.addItem(f"Camera {camera.camera_id}", camera.camera_id)
        camera_layout.addWidget(self.camera_combo)
        
        layout.addLayout(camera_layout)
        
        # Zone list
        self.zone_table = QTableWidget()
        self.zone_table.setColumnCount(3)
        self.zone_table.setHorizontalHeaderLabels(["Zone ID", "Points", "Actions"])
        self.zone_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.zone_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.zone_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.zone_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.zone_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.zone_table)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.add_coord_button = QPushButton("Add with Coordinates")
        self.add_coord_button.clicked.connect(self.add_with_coordinates)
        button_layout.addWidget(self.add_coord_button)
        
        self.add_draw_button = QPushButton("Add by Drawing")
        self.add_draw_button.clicked.connect(self.add_by_drawing)
        button_layout.addWidget(self.add_draw_button)
        
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.remove_zone)
        button_layout.addWidget(self.remove_button)
        
        layout.addLayout(button_layout)
        
        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        layout.addWidget(self.close_button)
        
        # Load zones for the selected camera
        self.camera_combo.currentIndexChanged.connect(self.load_zones)
        self.load_zones()
    
    def load_zones(self):
        camera_id = self.camera_combo.currentData()
        if not camera_id:
            return
        
        # Find the camera
        camera = None
        for cam in self.cameras:
            if cam.camera_id == camera_id:
                camera = cam
                break
        
        if not camera:
            return
        
        # Clear table
        self.zone_table.setRowCount(0)
        
        # Add zones to table
        for i, zone in enumerate(camera.zone_detector.zones):
            row_position = self.zone_table.rowCount()
            self.zone_table.insertRow(row_position)
            
            # Zone ID
            self.zone_table.setItem(row_position, 0, QTableWidgetItem(str(i + 1)))
            
            # Points count
            self.zone_table.setItem(row_position, 1, QTableWidgetItem(f"{len(zone)} points"))
            
            # Actions button
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            
            edit_button = QPushButton("Edit")
            edit_button.setProperty("zone_index", i)
            edit_button.clicked.connect(self.edit_zone)
            actions_layout.addWidget(edit_button)
            
            actions_layout.addStretch()
            
            self.zone_table.setCellWidget(row_position, 2, actions_widget)
    
    def add_with_coordinates(self):
        camera_id = self.camera_combo.currentData()
        if not camera_id:
            return
        
        # Open coordinate input dialog
        dialog = CoordinateInputDialog(camera_id, self)
        if dialog.exec_() == QDialog.Accepted:
            coordinates = dialog.get_coordinates()
            if len(coordinates) >= 3:  # Need at least 3 points for a polygon
                # Find the camera
                camera = None
                for cam in self.cameras:
                    if cam.camera_id == camera_id:
                        camera = cam
                        break
                
                if camera:
                    # Add zone
                    camera.zone_detector.add_zone_from_coordinates(coordinates)
                    self.load_zones()
                    self.zone_added.emit(camera_id, coordinates)
    
    def add_by_drawing(self):
        camera_id = self.camera_combo.currentData()
        if not camera_id:
            return
        
        QMessageBox.information(self, "Draw Zone", 
                              f"Switch to Camera {camera_id} view and draw the zone using the mouse.\n"
                              f"Left-click to add points, double-click to finish.")
    
    def edit_zone(self):
        camera_id = self.camera_combo.currentData()
        if not camera_id:
            return
        
        # Get the zone index
        button = self.sender()
        zone_index = button.property("zone_index")
        
        # Find the camera
        camera = None
        for cam in self.cameras:
            if cam.camera_id == camera_id:
                camera = cam
                break
        
        if not camera:
            return
        
        # Get the zone coordinates
        zone = camera.zone_detector.zones[zone_index]
        
        # Open coordinate input dialog with existing coordinates
        dialog = CoordinateInputDialog(camera_id, self)
        dialog.coordinates = zone.copy()
        
        # Populate the table
        for x, y in zone:
            row_position = dialog.coord_table.rowCount()
            dialog.coord_table.insertRow(row_position)
            dialog.coord_table.setItem(row_position, 0, QTableWidgetItem(str(x)))
            dialog.coord_table.setItem(row_position, 1, QTableWidgetItem(str(y)))
        
        if dialog.exec_() == QDialog.Accepted:
            coordinates = dialog.get_coordinates()
            if len(coordinates) >= 3:  # Need at least 3 points for a polygon
                # Update zone
                camera.zone_detector.zones[zone_index] = coordinates
                camera.zone_detector.save_zones()
                self.load_zones()
                self.zone_added.emit(camera_id, coordinates)
    
    def remove_zone(self):
        camera_id = self.camera_combo.currentData()
        if not camera_id:
            return
        
        selected_items = self.zone_table.selectedItems()
        if not selected_items:
            return
        
        row = selected_items[0].row()
        zone_index = row
        
        # Find the camera
        camera = None
        for cam in self.cameras:
            if cam.camera_id == camera_id:
                camera = cam
                break
        
        if camera:
            # Remove zone
            camera.zone_detector.zones.pop(zone_index)
            camera.zone_detector.save_zones()
            self.load_zones()

# PyQt5 Dashboard Classes

class CameraWidget(QFrame):
    alert_signal = pyqtSignal(dict)
    edit_zone_signal = pyqtSignal(str)
    
    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera = None
        self.current_frame = None
        self.drawing = False
        self.current_zone = []
        self.zones = []
        self.setMinimumSize(320, 240)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("CameraWidget { background-color: #2b2b2b; border: 1px solid #555; }")
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Camera label
        self.camera_label = QLabel(f"Camera {camera_id}")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.camera_label)
        
        # Video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setMouseTracking(True)
        layout.addWidget(self.video_label)
        
        # Status label
        self.status_label = QLabel("Disconnected")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #ff5555; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # FPS label
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setAlignment(Qt.AlignCenter)
        self.fps_label.setStyleSheet("color: #55ff55; font-size: 12px;")
        layout.addWidget(self.fps_label)
        
        # Siren indicator
        self.siren_indicator = QLabel()
        self.siren_indicator.setFixedSize(16, 16)
        self.siren_indicator.setStyleSheet("background-color: #333; border-radius: 8px;")
        layout.addWidget(self.siren_indicator, 0, Qt.AlignCenter)
        
        # Set up mouse tracking
        self.setMouseTracking(True)
        
    def set_camera(self, camera):
        self.camera = camera
        
    def mousePressEvent(self, event):
        if self.camera is None:
            return
            
        # Get the position relative to the video label
        pos = self.video_label.mapFrom(self, event.pos())
        
        # Check if the position is within the video label
        if 0 <= pos.x() < self.video_label.width() and 0 <= pos.y() < self.video_label.height():
            # Calculate the position in the original frame
            if self.current_frame is not None:
                frame_h, frame_w = self.current_frame.shape[:2]
                label_w, label_h = self.video_label.width(), self.video_label.height()
                
                # Calculate scaling factors
                scale_x = frame_w / label_w
                scale_y = frame_h / label_h
                
                # Calculate position in original frame
                frame_x = int(pos.x() * scale_x)
                frame_y = int(pos.y() * scale_y)
                
                # Handle mouse events
                if event.button() == Qt.LeftButton:
                    if not self.drawing:
                        self.drawing = True
                        self.current_zone = [(frame_x, frame_y)]
                        print(f"Camera {self.camera_id}: Started drawing zone at ({frame_x}, {frame_y})")
                    else:
                        self.current_zone.append((frame_x, frame_y))
                        print(f"Camera {self.camera_id}: Added point ({frame_x}, {frame_y}) to current zone")
                        
                elif event.button() == Qt.RightButton:
                    if self.drawing:
                        self.drawing = False
                        self.current_zone = []
                        print(f"Camera {self.camera_id}: Zone drawing cancelled")
    
    def mouseDoubleClickEvent(self, event):
        if self.camera is None or not self.drawing:
            return
            
        # Finish drawing the zone
        if len(self.current_zone) >= 3:
            self.camera.zone_detector.zones.append(self.current_zone)
            self.camera.zone_detector.save_zones()
            self.zones = self.camera.zone_detector.zones
            print(f"Camera {self.camera_id}: Zone added with {len(self.current_zone)} points")
        
        self.drawing = False
        self.current_zone = []
    
    def update_frame(self, frame_data):
        self.current_frame = frame_data['frame']
        self.detections = frame_data['detections']
        self.fps = frame_data['fps']
        self.siren_active = frame_data['siren_active']
        self.zones = frame_data['zones']
        self.current_zone = frame_data['current_zone']
        self.drawing = frame_data['drawing']
        
        # Update status
        self.status_label.setText("Connected")
        self.status_label.setStyleSheet("color: #55ff55; font-size: 12px;")
        
        # Update FPS
        self.fps_label.setText(f"FPS: {self.fps:.1f}")
        
        # Update siren indicator
        if self.siren_active:
            self.siren_indicator.setStyleSheet("background-color: #ff0000; border-radius: 8px;")
        else:
            self.siren_indicator.setStyleSheet("background-color: #333; border-radius: 8px;")
        
        # Process frame for display
        display_frame = self.current_frame.copy()
        
        # Draw danger zones
        for zone in self.zones:
            # Draw filled polygon with transparency
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [np.array(zone, dtype=np.int32)], (0, 0, 255))
            alpha = 0.2  # Transparency factor
            display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)
            
            # Draw polygon outline
            cv2.polylines(display_frame, [np.array(zone, dtype=np.int32)], True, (0, 0, 255), 2)
            
            # Add zone label
            if len(zone) > 0:
                # Calculate centroid of the polygon for label placement
                M = cv2.moments(np.array(zone, dtype=np.int32))
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(display_frame, "DANGER ZONE", (cX - 60, cY), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw current zone being drawn
        if self.drawing and len(self.current_zone) > 0:
            # Draw lines between points
            for i in range(len(self.current_zone) - 1):
                cv2.line(display_frame, self.current_zone[i], 
                        self.current_zone[i+1], (0, 255, 255), 2)
            
            # Draw points
            for point in self.current_zone:
                cv2.circle(display_frame, point, 3, (0, 255, 255), -1)
        
        # Draw person detections
        for detection in self.detections:
            bbox = detection['bbox']
            centroid = detection['centroid']
            
            # Draw bounding box
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            
            # Draw centroid
            cv2.circle(display_frame, centroid, 4, (255, 0, 0), -1)
            
            # Check if person is in a danger zone
            zone_idx = -1
            for i, zone in enumerate(self.zones):
                if self._point_in_polygon(centroid, zone):
                    zone_idx = i
                    break
            
            if zone_idx != -1:
                # Draw a warning indicator
                cv2.putText(display_frame, "ALERT!", (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Convert to RGB for Qt
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale image to fit the label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Draw on the pixmap if needed
        if self.drawing and len(self.current_zone) > 0:
            painter = QPainter(scaled_pixmap)
            painter.setPen(QPen(QColor(0, 255, 255), 2))
            
            # Scale the current zone points to the pixmap size
            label_w, label_h = self.video_label.width(), self.video_label.height()
            frame_h, frame_w = self.current_frame.shape[:2]
            
            scale_x = label_w / frame_w
            scale_y = label_h / frame_h
            
            scaled_zone = [(int(x * scale_x), int(y * scale_y)) for x, y in self.current_zone]
            
            # Draw lines between points
            for i in range(len(scaled_zone) - 1):
                painter.drawLine(scaled_zone[i][0], scaled_zone[i][1], 
                               scaled_zone[i+1][0], scaled_zone[i+1][1])
            
            # Draw points
            for point in scaled_zone:
                painter.drawEllipse(point[0] - 3, point[1] - 3, 6, 6)
            
            painter.end()
        
        self.video_label.setPixmap(scaled_pixmap)
        
        # Emit alert signal if there's an alert
        if frame_data.get('alert_data'):
            self.alert_signal.emit(frame_data['alert_data'])
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def clear_zones(self):
        if self.camera:
            self.camera.zone_detector.clear_zones()
            self.camera.zone_detector.save_zones()
            self.zones = []

class AlertPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(300)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("AlertPanel { background-color: #2b2b2b; border: 1px solid #555; }")
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Recent Alerts")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white; font-weight: bold; font-size: 16px; padding: 10px;")
        layout.addWidget(title)
        
        # Alert list
        self.alert_list = QTextEdit()
        self.alert_list.setReadOnly(True)
        self.alert_list.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #555;
                font-family: monospace;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.alert_list)
        
        # Clear button
        self.clear_button = QPushButton("Clear Alerts")
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
        """)
        self.clear_button.clicked.connect(self.clear_alerts)
        layout.addWidget(self.clear_button)
    
    def add_alert(self, alert_data):
        timestamp = alert_data['timestamp']
        camera_id = alert_data['camera_id']
        zone_idx = alert_data['zone_idx']
        
        alert_text = f"[{timestamp}] Camera {camera_id}: Person detected in zone {zone_idx}\n"
        
        # Add to the text edit
        cursor = self.alert_list.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(alert_text)
        
        # Scroll to the bottom
        scrollbar = self.alert_list.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_alerts(self):
        self.alert_list.clear()

class ControlPanel(QFrame):
    start_signal = pyqtSignal()
    stop_signal = pyqtSignal()
    clear_zones_signal = pyqtSignal()
    manage_zones_signal = pyqtSignal()
    edit_camera1_zones_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(300)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("ControlPanel { background-color: #2b2b2b; border: 1px solid #555; }")
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("System Controls")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white; font-weight: bold; font-size: 16px; padding: 10px;")
        layout.addWidget(title)
        
        # Start/Stop buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5CBF60;
            }
            QPushButton:pressed {
                background-color: #3A8F3E;
            }
        """)
        self.start_button.clicked.connect(self.start_signal.emit)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F55346;
            }
            QPushButton:pressed {
                background-color: #D43326;
            }
        """)
        self.stop_button.clicked.connect(self.stop_signal.emit)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        # Zone management buttons
        self.clear_zones_button = QPushButton("Clear All Zones")
        self.clear_zones_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FFA810;
            }
            QPushButton:pressed {
                background-color: #DF7800;
            }
        """)
        self.clear_zones_button.clicked.connect(self.clear_zones_signal.emit)
        layout.addWidget(self.clear_zones_button)
        
        self.manage_zones_button = QPushButton("Manage Zones")
        self.manage_zones_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #31A6F3;
            }
            QPushButton:pressed {
                background-color: #1186E3;
            }
        """)
        self.manage_zones_button.clicked.connect(self.manage_zones_signal.emit)
        layout.addWidget(self.manage_zones_button)
        
        # Edit Camera 1 zones button
        self.edit_camera1_button = QPushButton("Edit Camera 1 Zones")
        self.edit_camera1_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #AC37C0;
            }
            QPushButton:pressed {
                background-color: #8C17A0;
            }
        """)
        self.edit_camera1_button.clicked.connect(self.edit_camera1_zones_signal.emit)
        layout.addWidget(self.edit_camera1_button)
        
        # Settings group
        settings_group = QGroupBox("Settings")
        settings_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        settings_layout = QFormLayout(settings_group)
        
        # Detection size
        self.detection_size_combo = QComboBox()
        self.detection_size_combo.addItems(["320x180", "640x360", "1280x720"])
        self.detection_size_combo.setCurrentIndex(1)  # 640x360
        self.detection_size_combo.setStyleSheet("""
            QComboBox {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                padding: 3px;
                border-radius: 3px;
            }
        """)
        settings_layout.addRow("Detection Size:", self.detection_size_combo)
        
        # Frame skip
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 10)
        self.frame_skip_spin.setValue(2)
        self.frame_skip_spin.setStyleSheet("""
            QSpinBox {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                padding: 3px;
                border-radius: 3px;
            }
        """)
        settings_layout.addRow("Frame Skip:", self.frame_skip_spin)
        
        # Siren volume
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #3a3a3a;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        settings_layout.addRow("Siren Volume:", self.volume_slider)
        
        layout.addWidget(settings_group)
        
        # Status group
        status_group = QGroupBox("System Status")
        status_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        status_layout = QFormLayout(status_group)
        
        # System uptime
        self.uptime_label = QLabel("00:00:00")
        self.uptime_label.setStyleSheet("color: #55ff55;")
        status_layout.addRow("Uptime:", self.uptime_label)
        
        # Active cameras
        self.active_cameras_label = QLabel("0/0")
        self.active_cameras_label.setStyleSheet("color: #55ff55;")
        status_layout.addRow("Active Cameras:", self.active_cameras_label)
        
        # Total alerts
        self.total_alerts_label = QLabel("0")
        self.total_alerts_label.setStyleSheet("color: #ff5555;")
        status_layout.addRow("Total Alerts:", self.total_alerts_label)
        
        layout.addWidget(status_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()

class Dashboard(QMainWindow):
    def __init__(self, camera_configs):
        super().__init__()
        self.camera_configs = camera_configs
        self.cameras = []
        self.camera_widgets = []
        self.running = False
        self.start_time = time.time()
        self.total_alerts = 0
        
        # Set up the UI
        self.init_ui()
        
        # Set up timer for updating status
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second
        
        # Set up timer for processing camera frames
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.process_frames)
        self.frame_timer.start(30)  # Process frames at ~33 FPS
    
    def init_ui(self):
        # Set window properties
        self.setWindowTitle("Security Camera Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: white;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: white;
            }
            QMenuBar::item:selected {
                background-color: #3d3d3d;
            }
            QMenu {
                background-color: #2d2d2d;
                color: white;
            }
            QMenu::item:selected {
                background-color: #3d3d3d;
            }
            QStatusBar {
                background-color: #2d2d2d;
                color: white;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Create camera grid widget
        self.camera_grid = QWidget()
        camera_layout = QGridLayout(self.camera_grid)
        camera_layout.setSpacing(10)
        camera_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create camera widgets
        for i, config in enumerate(self.camera_configs):
            camera_id = config['camera_id']
            camera_widget = CameraWidget(camera_id)
            camera_widget.alert_signal.connect(self.add_alert)
            camera_widget.edit_zone_signal.connect(self.edit_zone)
            
            # Add to grid
            row = i // 2
            col = i % 2
            camera_layout.addWidget(camera_widget, row, col)
            
            self.camera_widgets.append(camera_widget)
        
        # Add camera grid to splitter
        splitter.addWidget(self.camera_grid)
        
        # Create right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create alert panel
        self.alert_panel = AlertPanel()
        right_layout.addWidget(self.alert_panel)
        
        # Create control panel
        self.control_panel = ControlPanel()
        self.control_panel.start_signal.connect(self.start_system)
        self.control_panel.stop_signal.connect(self.stop_system)
        self.control_panel.clear_zones_signal.connect(self.clear_all_zones)
        self.control_panel.manage_zones_signal.connect(self.manage_zones)
        self.control_panel.edit_camera1_zones_signal.connect(self.edit_camera1_zones)
        right_layout.addWidget(self.control_panel)
        
        # Add right panel to splitter
        splitter.addWidget(right_panel)
        
        # Set splitter sizes
        splitter.setSizes([800, 300])
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Camera layout actions
        grid_1x1_action = QAction("1x1 Grid", self)
        grid_1x1_action.triggered.connect(lambda: self.set_camera_grid(1, 1))
        view_menu.addAction(grid_1x1_action)
        
        grid_2x2_action = QAction("2x2 Grid", self)
        grid_2x2_action.triggered.connect(lambda: self.set_camera_grid(2, 2))
        view_menu.addAction(grid_2x2_action)
        
        grid_3x3_action = QAction("3x3 Grid", self)
        grid_3x3_action.triggered.connect(lambda: self.set_camera_grid(3, 3))
        view_menu.addAction(grid_3x3_action)
        
        # Zones menu
        zones_menu = menubar.addMenu("Zones")
        
        manage_zones_action = QAction("Manage Zones", self)
        manage_zones_action.triggered.connect(self.manage_zones)
        zones_menu.addAction(manage_zones_action)
        
        edit_camera1_zones_action = QAction("Edit Camera 1 Zones", self)
        edit_camera1_zones_action.triggered.connect(self.edit_camera1_zones)
        zones_menu.addAction(edit_camera1_zones_action)
        
        clear_zones_action = QAction("Clear All Zones", self)
        clear_zones_action.triggered.connect(self.clear_all_zones)
        zones_menu.addAction(clear_zones_action)
        
        export_zones_action = QAction("Export Zones", self)
        export_zones_action.triggered.connect(self.export_zones)
        zones_menu.addAction(export_zones_action)
        
        import_zones_action = QAction("Import Zones", self)
        import_zones_action.triggered.connect(self.import_zones)
        zones_menu.addAction(import_zones_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        # Add coordinate input action
        add_coord_action = QAction("Add Zone by Coordinates", self)
        add_coord_action.triggered.connect(self.add_zone_by_coordinates)
        tools_menu.addAction(add_coord_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def set_camera_grid(self, rows, cols):
        # Clear the grid
        for i in reversed(range(self.camera_grid.layout().count())):
            self.camera_grid.layout().itemAt(i).widget().setParent(None)
        
        # Re-add camera widgets
        for i, camera_widget in enumerate(self.camera_widgets):
            if i < rows * cols:
                row = i // cols
                col = i % cols
                self.camera_grid.layout().addWidget(camera_widget, row, col)
                camera_widget.setVisible(True)
            else:
                camera_widget.setVisible(False)
    
    def start_system(self):
        if self.running:
            return
        
        self.running = True
        self.start_time = time.time()
        
        # Create and start cameras
        for i, config in enumerate(self.camera_configs):
            camera = Camera(
                camera_id=config['camera_id'],
                source=config['source'],
                config_file=config.get('config_file'),
                detection_size=self.get_detection_size(),
                frame_skip=self.control_panel.frame_skip_spin.value(),
                use_threading=True
            )
            
            if camera.initialize():
                camera.start_thread()
                self.cameras.append(camera)
                self.camera_widgets[i].set_camera(camera)
                self.status_bar.showMessage(f"Camera {config['camera_id']} started")
            else:
                self.status_bar.showMessage(f"Failed to start Camera {config['camera_id']}")
        
        # Update control panel
        self.control_panel.start_button.setEnabled(False)
        self.control_panel.stop_button.setEnabled(True)
    
    def stop_system(self):
        if not self.running:
            return
        
        self.running = False
        
        # Stop cameras
        for camera in self.cameras:
            camera.release()
        
        self.cameras = []
        
        # Update camera widgets
        for camera_widget in self.camera_widgets:
            camera_widget.set_camera(None)
            camera_widget.status_label.setText("Disconnected")
            camera_widget.status_label.setStyleSheet("color: #ff5555; font-size: 12px;")
            camera_widget.fps_label.setText("FPS: 0.0")
            camera_widget.siren_indicator.setStyleSheet("background-color: #333; border-radius: 8px;")
        
        # Update control panel
        self.control_panel.start_button.setEnabled(True)
        self.control_panel.stop_button.setEnabled(False)
        
        self.status_bar.showMessage("System stopped")
    
    def clear_all_zones(self):
        reply = QMessageBox.question(
            self, 'Clear All Zones',
            'Are you sure you want to clear all danger zones?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for camera_widget in self.camera_widgets:
                camera_widget.clear_zones()
            
            self.status_bar.showMessage("All zones cleared")
    
    def manage_zones(self):
        if not self.cameras:
            QMessageBox.warning(self, "No Cameras", "Please start the system first.")
            return
        
        # Open zone manager dialog
        dialog = ZoneManagerDialog(self.cameras, self)
        dialog.zone_added.connect(self.on_zone_added)
        dialog.exec_()
    
    def edit_camera1_zones(self):
        if not self.cameras:
            QMessageBox.warning(self, "No Cameras", "Please start the system first.")
            return
        
        # Find Camera 1
        camera1 = None
        for camera in self.cameras:
            if camera.camera_id == "1":
                camera1 = camera
                break
        
        if not camera1:
            QMessageBox.warning(self, "Camera 1 Not Found", "Camera 1 is not available.")
            return
        
        # Open Camera 1 zone editor dialog
        dialog = CameraZoneEditorDialog(camera1, self)
        dialog.exec_()
        
        # Update the corresponding camera widget
        for camera_widget in self.camera_widgets:
            if camera_widget.camera_id == "1":
                camera_widget.zones = camera1.zone_detector.zones
                break
    
    def on_zone_added(self, camera_id, coordinates):
        self.status_bar.showMessage(f"Zone added for Camera {camera_id}")
    
    def edit_zone(self, camera_id):
        # Find the camera widget
        camera_widget = None
        for widget in self.camera_widgets:
            if widget.camera_id == camera_id:
                camera_widget = widget
                break
        
        if camera_widget and camera_widget.camera:
            # Switch to the camera's grid position
            for i in range(self.camera_grid.layout().count()):
                item = self.camera_grid.layout().itemAt(i)
                if item.widget() == camera_widget:
                    # Highlight the camera widget
                    camera_widget.setStyleSheet("CameraWidget { background-color: #3b3b3b; border: 2px solid #4a90e2; }")
                    # Reset style after 2 seconds
                    QTimer.singleShot(2000, lambda: camera_widget.setStyleSheet("CameraWidget { background-color: #2b2b2b; border: 1px solid #555; }"))
                    break
    
    def add_zone_by_coordinates(self):
        if not self.cameras:
            QMessageBox.warning(self, "No Cameras", "Please start the system first.")
            return
        
        # Create a dialog to select camera
        camera_dialog = QDialog(self)
        camera_dialog.setWindowTitle("Select Camera")
        camera_dialog.setMinimumSize(300, 150)
        camera_dialog.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: white;
            }
            QLabel {
                color: white;
            }
            QComboBox {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #666;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
        """)
        
        layout = QVBoxLayout(camera_dialog)
        
        layout.addWidget(QLabel("Select a camera:"))
        
        camera_combo = QComboBox()
        for camera in self.cameras:
            camera_combo.addItem(f"Camera {camera.camera_id}", camera.camera_id)
        layout.addWidget(camera_combo)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(camera_dialog.accept)
        button_box.rejected.connect(camera_dialog.reject)
        layout.addWidget(button_box)
        
        if camera_dialog.exec_() == QDialog.Accepted:
            camera_id = camera_combo.currentData()
            
            # Open coordinate input dialog
            dialog = CoordinateInputDialog(camera_id, self)
            if dialog.exec_() == QDialog.Accepted:
                coordinates = dialog.get_coordinates()
                if len(coordinates) >= 3:  # Need at least 3 points for a polygon
                    # Find the camera
                    camera = None
                    for cam in self.cameras:
                        if cam.camera_id == camera_id:
                            camera = cam
                            break
                    
                    if camera:
                        # Add zone
                        camera.zone_detector.add_zone_from_coordinates(coordinates)
                        self.status_bar.showMessage(f"Zone added for Camera {camera_id}")
    
    def export_zones(self):
        # Create a dictionary with all zones
        all_zones = {}
        for camera in self.cameras:
            all_zones[camera.camera_id] = camera.zone_detector.zones
        
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Zones", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(all_zones, f, indent=2)
                self.status_bar.showMessage(f"Zones exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export zones: {str(e)}")
    
    def import_zones(self):
        # Ask for file location
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Zones", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    all_zones = json.load(f)
                
                # Import zones for each camera
                for camera in self.cameras:
                    if camera.camera_id in all_zones:
                        camera.zone_detector.zones = all_zones[camera.camera_id]
                        camera.zone_detector.save_zones()
                
                self.status_bar.showMessage(f"Zones imported from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import zones: {str(e)}")
    
    def show_about(self):
        QMessageBox.about(
            self,
            "About Security Camera Dashboard",
            "<h1>Security Camera Dashboard</h1>"
            "<p>A modern dashboard for multi-camera security monitoring with danger zone detection.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>Real-time camera monitoring</li>"
            "<li>Interactive danger zone drawing</li>"
            "<li>Precise coordinate input for zones</li>"
            "<li>Alert system with siren</li>"
            "<li>Zone management and export/import</li>"
            "<li>Dedicated Camera 1 zone editor</li>"
            "</ul>"
            "<p>Version 1.0</p>"
            "<p>Created with PyQt5 and OpenCV</p>"
        )
    
    def add_alert(self, alert_data):
        self.alert_panel.add_alert(alert_data)
        self.total_alerts += 1
        self.control_panel.total_alerts_label.setText(str(self.total_alerts))
    
    def update_status(self):
        # Update uptime
        if self.running:
            uptime = time.time() - self.start_time
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.control_panel.uptime_label.setText(f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        # Update active cameras
        active_cameras = sum(1 for camera in self.cameras if camera.running)
        total_cameras = len(self.cameras)
        self.control_panel.active_cameras_label.setText(f"{active_cameras}/{total_cameras}")
    
    def process_frames(self):
        # Process frames from all cameras
        for camera in self.cameras:
            try:
                # Get frame data from camera queue (non-blocking)
                frame_data = camera.frame_queue.get(block=False)
                
                # Find the corresponding camera widget
                for camera_widget in self.camera_widgets:
                    if camera_widget.camera_id == camera.camera_id:
                        camera_widget.update_frame(frame_data)
                        break
            except queue.Empty:
                # No frame available, continue to next camera
                pass
    
    def get_detection_size(self):
        size_str = self.control_panel.detection_size_combo.currentText()
        width, height = map(int, size_str.split('x'))
        return (width, height)
    
    def closeEvent(self, event):
        # Stop the system before closing
        if self.running:
            self.stop_system()
        
        # Accept the close event
        event.accept()

def main():
    # Define camera configurations
    camera_configs = [
        {
            'camera_id': '1',
            'source': 'rtsp://admin:afaqkhan-11@192.168.18.150:554/cam/realmonitor?channel=1&subtype=0',
            'config_file': 'zone_config1 copy.json'
        },
        {
            'camera_id': '2',
            'source': 'rtsp://admin:afaqkhan-11@192.168.18.149:554/cam/realmonitor?channel=1&subtype=0',
            'config_file': 'zone_config2 copy.json'
        },
        {
            'camera_id': '3',
            'source': 'rtsp://admin:afaqkhan-11@192.168.18.149:554/cam/realmonitor?channel=1&subtype=0',
            'config_file': 'zone_config3 copy.json'
        }
    ]
    
    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application palette for dark theme
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    # Create and show dashboard
    dashboard = Dashboard(camera_configs)
    dashboard.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()