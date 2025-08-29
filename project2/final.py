import cv2
import numpy as np
import time
import threading
import queue
import os
from datetime import datetime
from ultralytics import YOLO
import pygame
from typing import List, Tuple, Dict, Optional, Any
import signal
import sys
import json

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
        self.active_camera = False  # Flag to indicate if this camera is active for drawing
        
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
            if alert_triggered:
                self.alert_system.trigger_zone_alert(frame, zone_idx)
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
            
            # Prepare frame data for GUI
            frame_data = {
                'camera_id': self.camera_id,
                'frame': frame,
                'detections': detections,
                'fps': self.current_fps,
                'siren_active': self.siren_active,
                'zones': self.zone_detector.zones,
                'current_zone': self.zone_detector.current_zone,
                'drawing': self.zone_detector.drawing
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

class GUIManager:
    def __init__(self, camera_ids: List[str]):
        """Initialize GUI manager for display and user interactions."""
        self.camera_ids = camera_ids
        self.window_names = [f"Camera {camera_id}" for camera_id in camera_ids]
        self.colors = {
            'zone': (0, 0, 255),        # Red for danger zones
            'current_zone': (0, 255, 255),  # Yellow for zone being drawn
            'person': (255, 0, 0),       # Blue for person bounding box
            'text': (255, 255, 255)      # White for text
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.mouse_callback_data = {}
        self.active_camera_id = None  # Which camera is active for drawing
        
    def initialize(self):
        """Initialize OpenCV windows."""
        for window_name in self.window_names:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for OpenCV window."""
        # Extract callback data
        cameras = self.mouse_callback_data.get('cameras')
        if cameras is None:
            return
        
        # Determine which window the mouse is in
        window_name = cv2.getWindowProperty(param, cv2.WND_PROP_FULLSCREEN)
        if window_name < 0:  # Window not found
            return
        
        # Find the camera ID from the window name
        camera_id = None
        for cam_id in self.camera_ids:
            if f"Camera {cam_id}" == param:
                camera_id = cam_id
                break
        
        if camera_id is None:
            return
        
        # Set this as the active camera for drawing
        self.active_camera_id = camera_id
        
        # Get the zone detector for this camera
        zone_detector = None
        for camera in cameras:
            if camera.camera_id == camera_id:
                zone_detector = camera.zone_detector
                break
        
        if zone_detector is None:
            return
        
        # Left click - add point to current zone
        if event == cv2.EVENT_LBUTTONDOWN:
            if not zone_detector.drawing:
                zone_detector.start_drawing((x, y))
                print(f"Camera {camera_id}: Started drawing zone at ({x}, {y})")
            else:
                zone_detector.add_point((x, y))
                print(f"Camera {camera_id}: Added point ({x}, {y}) to current zone")
        
        # Double left click - finish drawing zone
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            if zone_detector.drawing:
                if zone_detector.finish_drawing():
                    print(f"Camera {camera_id}: Zone added with {len(zone_detector.zones[-1])} points")
                    zone_detector.save_zones()
                else:
                    print("Camera {camera_id}: Failed to add zone - need at least 3 points")
        
        # Right click - cancel drawing zone
        elif event == cv2.EVENT_RBUTTONDOWN:
            if zone_detector.drawing:
                zone_detector.cancel_drawing()
                print(f"Camera {camera_id}: Zone drawing cancelled")
    
    def set_callback_data(self, cameras):
        """Set the data for mouse callback."""
        self.mouse_callback_data['cameras'] = cameras
    
    def draw_frame(self, frame_data: Dict) -> np.ndarray:
        """Draw frame with overlays."""
        frame = frame_data['frame']
        detections = frame_data['detections']
        fps = frame_data['fps']
        siren_active = frame_data['siren_active']
        zones = frame_data['zones']
        current_zone = frame_data['current_zone']
        drawing = frame_data['drawing']
        camera_id = frame_data['camera_id']
        
        # Make a copy of the frame to avoid modifying the original
        display_frame = frame.copy()
        
        # Draw danger zones
        for zone in zones:
            # Draw filled polygon with transparency
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [np.array(zone, dtype=np.int32)], self.colors['zone'])
            alpha = 0.2  # Transparency factor
            display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)
            
            # Draw polygon outline
            cv2.polylines(display_frame, [np.array(zone, dtype=np.int32)], True, 
                         self.colors['zone'], 2, cv2.LINE_AA)
            
            # Add zone label
            if len(zone) > 0:
                # Calculate centroid of the polygon for label placement
                M = cv2.moments(np.array(zone, dtype=np.int32))
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(display_frame, "DANGER ZONE", (cX - 60, cY), 
                               self.font, self.font_scale, self.colors['zone'], self.font_thickness)
        
        # Draw current zone being drawn
        if drawing and len(current_zone) > 0:
            # Draw lines between points
            for i in range(len(current_zone) - 1):
                cv2.line(display_frame, current_zone[i], 
                        current_zone[i+1], self.colors['current_zone'], 2)
            
            # Draw points
            for point in current_zone:
                cv2.circle(display_frame, point, 3, self.colors['current_zone'], -1)
            
            # Add instruction text
            if len(current_zone) >= 3:
                cv2.putText(display_frame, "Double-click to finish zone", 
                           (current_zone[-1][0] - 100, current_zone[-1][1] - 10), 
                           self.font, self.font_scale, self.colors['current_zone'], self.font_thickness)
            else:
                cv2.putText(display_frame, "Add more points", 
                           (current_zone[-1][0] - 60, current_zone[-1][1] - 10), 
                           self.font, self.font_scale, self.colors['current_zone'], self.font_thickness)
        
        # Draw person detections
        for detection in detections:
            bbox = detection['bbox']
            centroid = detection['centroid']
            
            # Draw bounding box
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         self.colors['person'], 2)
            
            # Draw centroid
            cv2.circle(display_frame, centroid, 4, self.colors['person'], -1)
            
            # Check if person is in a danger zone
            zone_idx = -1
            for i, zone in enumerate(zones):
                if self._point_in_polygon(centroid, zone):
                    zone_idx = i
                    break
            
            if zone_idx != -1:
                # Draw a warning indicator
                cv2.putText(display_frame, "ALERT!", (bbox[0], bbox[1] - 10), 
                           self.font, self.font_scale, (0, 0, 255), self.font_thickness)
        
        # Draw information text
        self._draw_info(display_frame, fps, len(detections), siren_active, drawing, len(current_zone))
        
        return display_frame
    
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
    
    def _draw_info(self, frame: np.ndarray, fps: float, person_count: int, 
                  siren_active: bool, drawing: bool, current_zone_points: int):
        """Draw information text on frame."""
        h, w = frame.shape[:2]
        line_height = 25
        margin = 10
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (margin, line_height), 
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        
        # Person count
        cv2.putText(frame, f"People: {person_count}", (margin, 2 * line_height), 
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        
        # Siren status
        siren_status = "Siren: ON" if siren_active else "Siren: OFF"
        siren_color = (0, 0, 255) if siren_active else self.colors['text']
        cv2.putText(frame, siren_status, (margin, 3 * line_height), 
                   self.font, self.font_scale, siren_color, self.font_thickness)
        
        # Drawing mode status
        if drawing:
            mode_text = f"Drawing zone: {current_zone_points} points"
        else:
            mode_text = "Mode: Monitoring"
        
        cv2.putText(frame, mode_text, (margin, 4 * line_height), 
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        
        # Instructions
        instructions = [
            "Left Click: Add zone point",
            "Double Click: Finish zone",
            "Right Click: Cancel zone",
            "Press 'c': Clear all zones",
            "Press 'q': Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (margin, (i + 5) * line_height), 
                       self.font, self.font_scale, self.colors['text'], self.font_thickness)
    
    def show_frame(self, camera_id: str, frame: np.ndarray):
        """Show frame in OpenCV window."""
        window_name = f"Camera {camera_id}"
        cv2.imshow(window_name, frame)
    
    def check_key(self) -> str:
        """Check for key press and return the key character."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return 'q'
        elif key == ord('c'):
            return 'c'
        return None
    
    def release(self):
        """Release GUI resources."""
        cv2.destroyAllWindows()

class MultiCameraApp:
    def __init__(self, camera_configs: List[Dict]):
        """Initialize multi-camera application."""
        self.cameras = []
        self.running = False
        
        # Create camera objects
        for config in camera_configs:
            camera = Camera(
                camera_id=config['camera_id'],
                source=config['source'],
                config_file=config.get('config_file'),
                detection_size=config.get('detection_size', (640, 360)),
                frame_skip=config.get('frame_skip', 2),
                use_threading=config.get('use_threading', True)
            )
            self.cameras.append(camera)
        
        # Create GUI manager
        camera_ids = [config['camera_id'] for config in camera_configs]
        self.gui_manager = GUIManager(camera_ids)
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize all cameras."""
        print("Initializing Multi-Camera System...")
        
        for camera in self.cameras:
            if not camera.initialize():
                print(f"Failed to initialize Camera {camera.camera_id}")
                return False
        
        # Initialize GUI
        self.gui_manager.initialize()
        self.gui_manager.set_callback_data(self.cameras)
        
        print("Multi-Camera System initialization complete")
        return True
    
    def run(self):
        """Run the multi-camera application."""
        self.running = True
        print("Starting Multi-Camera System...")
        
        # Start each camera in a separate thread
        for camera in self.cameras:
            camera.start_thread()
        
        # Main GUI loop
        try:
            while self.running:
                # Check if all camera threads are still alive
                all_alive = True
                for camera in self.cameras:
                    if camera.thread and not camera.thread.is_alive():
                        all_alive = False
                        break
                
                if not all_alive:
                    print("One or more camera threads have stopped. Shutting down...")
                    self.running = False
                
                # Process frames from all cameras
                for camera in self.cameras:
                    try:
                        # Get frame from camera queue (non-blocking)
                        frame_data = camera.frame_queue.get(block=False)
                        
                        # Draw frame with overlays
                        display_frame = self.gui_manager.draw_frame(frame_data)
                        
                        # Show frame
                        self.gui_manager.show_frame(camera.camera_id, display_frame)
                    except queue.Empty:
                        # No frame available, continue to next camera
                        pass
                
                # Check for key presses
                key = self.gui_manager.check_key()
                if key == 'q':
                    self.running = False
                elif key == 'c':
                    # Clear all zones for the active camera
                    if self.gui_manager.active_camera_id:
                        for camera in self.cameras:
                            if camera.camera_id == self.gui_manager.active_camera_id:
                                camera.zone_detector.clear_zones()
                                camera.zone_detector.save_zones()
                                print(f"Camera {camera.camera_id}: All zones cleared")
                                break
                
                # Small delay to prevent high CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self.running = False
        
        # Wait for threads to finish
        for camera in self.cameras:
            if camera.thread and camera.thread.is_alive():
                camera.thread.join(timeout=1.0)
        
        print("Multi-Camera System ended")
    
    def release(self):
        """Release all resources."""
        print("Releasing Multi-Camera System resources...")
        
        for camera in self.cameras:
            camera.release()
        
        self.gui_manager.release()
        
        print("Multi-Camera System resources released")

def main():
    # Define camera configurations
    camera_configs = [
        {
            'camera_id': '1',
            'source': 'rtsp://admin:afaqkhan-11@192.168.18.124:554/cam/realmonitor?channel=1&subtype=0',
            'config_file': 'zone_config1.json'
        },
        {
            'camera_id': '2',
            'source': 'rtsp://admin:afaqkhan-11@192.168.18.125:554/cam/realmonitor?channel=1&subtype=0',
            'config_file': 'zone_config2.json'
        },
        {
            'camera_id': '3',
            'source': 'rtsp://admin:afaqkhan-11@192.168.18.126:554/cam/realmonitor?channel=1&subtype=0',
            'config_file': 'zone_config3.json'
        }
    ]
    
    # Create and run multi-camera application
    app = MultiCameraApp(camera_configs)
    
    if app.initialize():
        try:
            app.run()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            app.release()
    else:
        print("Failed to initialize Multi-Camera System")

if __name__ == "__main__":
    main()