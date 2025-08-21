import easyocr
import re

class PlateRecognizer:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.stream_handler = RTSPStreamHandler(rtsp_url)
        self.stream_handler.start()
        
        # Initialize EasyOCR
        self.reader = easyocr.Reader(['en'])
        
        # Known plates database
        self.known_plates = []
        self.load_known_plates()
        
        # Cooldown settings
        self.last_recognition = {}
        self.cooldown_period = 5  # seconds
        
        # Performance monitoring
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
    
    def load_known_plates(self):
        """Load known plates from database"""
        # Implementation would fetch from DB
        # For demo, we'll use a list
        self.known_plates = [
            "ABC123", "XYZ789", "DEF456", "GHI101"
        ]
    
    def recognize_plate(self, frame):
        """Recognize license plates in frame"""
        # Update FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Add FPS to frame
        cv2.putText(frame, f"FPS: {self.fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add stream status
        status = "STREAM: ONLINE" if self.stream_handler.is_alive() else "STREAM: OFFLINE"
        color = (0, 255, 0) if self.stream_handler.is_alive() else (0, 0, 255)
        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use EasyOCR to detect text
        detections = self.reader.readtext(gray)
        
        for (bbox, text, confidence) in detections:
            # Clean text
            text = text.upper().replace(" ", "")
            text = re.sub(r'[^A-Z0-9]', '', text)
            
            # Check if it's a valid plate format (example: 3 letters followed by 3 digits)
            if re.match(r'^[A-Z]{3}[0-9]{3}$', text):
                # Check cooldown
                current_time = time.time()
                if text in self.last_recognition:
                    if current_time - self.last_recognition[text] < self.cooldown_period:
                        continue
                
                self.last_recognition[text] = current_time
                
                # Check if plate is known
                is_known = text in self.known_plates
                
                # Draw bounding box
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                
                box_color = (0, 255, 0) if is_known else (0, 0, 255)
                cv2.rectangle(frame, top_left, bottom_right, box_color, 2)
                
                # Draw plate text and confidence
                label = f"{text} ({confidence:.2f})"
                cv2.putText(frame, label, (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                
                results.append({
                    'plate': text,
                    'confidence': confidence,
                    'is_known': is_known,
                    'bbox': bbox
                })
        
        return results, frame
    
    def capture_snapshot(self, frame, plate="unknown"):
        """Capture snapshot of unknown plate"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshots/plate_{plate}_{timestamp}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, frame)
        return filename
    
    def release(self):
        self.stream_handler.stop()