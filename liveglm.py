import cv2
import threading
import queue
import time
import os
import csv
import json
import base64
import logging
import sqlite3
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Tuple
from flask import Flask, Response, jsonify, send_from_directory, request
from ultralytics import YOLO
import easyocr
import insightface
from insightface.app import FaceAnalysis

# ---------------- Configuration ----------------
RTSP_URL = "rtsp://admin:afaqkhan-1@192.168.18.116:554/Streaming/channels/101"
AUTHORIZED_PLATES = {
    "EW080": "Fahad Bangash",
    "KHT9090": "Ali Akbar",
    "PES5566": "Talal Syed",
    "MN144480": "Aitazaz"
}
AUTHORIZED_FACES = ["Fahad Bangash", "Ali Akbar", "Talal Syed", "Aitazaz"]  # Names matching known faces

# Directories
SNAPSHOT_DIR = "snapshots"
UNKNOWN_DIR = "unknown_faces"
KNOWN_DIR = "known_faces"
LOGS_DIR = "logs"
DB_FILE = "access_control.db"
CSV_FILE = "access_log.csv"

# Create directories
for d in [SNAPSHOT_DIR, UNKNOWN_DIR, KNOWN_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Recognition thresholds
PLATE_SIM_THRESHOLD = 0.7
FACE_SIM_THRESHOLD = 0.45
MIN_FACE_SIZE = 50
UNKNOWN_DEDUP_SECONDS = 30
BARRIER_COOLDOWN = 10  # seconds between triggers

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "access_control.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("access_control")

# ---------------- Database Setup ----------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS access_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        access_type TEXT NOT NULL,
        identifier TEXT NOT NULL,
        authorized BOOLEAN NOT NULL,
        snapshot_path TEXT,
        barrier_triggered BOOLEAN
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS known_faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        embedding BLOB NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize CSV logging
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Access Type', 'Identifier', 'Authorized', 'Snapshot', 'Barrier Triggered'])

# ---------------- Models Initialization ----------------
# License plate models
plate_detector = YOLO("license_plate_detector.pt")  # Replace with your trained model
ocr_reader = easyocr.Reader(['en'])

# Face recognition model
def init_face_model():
    try:
        face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        # Tighten detector threshold to reduce false-positives
        try:
            face_app.models["detection"].threshold = 0.5
        except Exception:
            pass
        return face_app
    except Exception as e:
        logger.error(f"Failed to initialize face model: {e}")
        return None

face_app = init_face_model()

# ---------------- Face Index Management ----------------
class FaceIndex:
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.labels: List[str] = []
    
    def clear(self):
        self.embeddings = []
        self.labels = []
    
    def add(self, label: str, emb: np.ndarray):
        if emb is None:
            return
        self.embeddings.append(emb.astype(np.float32))
        self.labels.append(label)
    
    def build_from_dir(self, folder: str):
        self.clear()
        logger.info(f"Building face index from: {folder}")
        for name in os.listdir(folder):
            p = os.path.join(folder, name)
            if not os.path.isfile(p):
                continue
            base, ext = os.path.splitext(name)
            if ext.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Failed to read known face: {p}")
                continue
            persons = face_app.get(img)
            if not persons:
                logger.warning(f"No face in known image: {p}")
                continue
            self.add(base, persons[0].embedding)
        logger.info(f"Indexed {len(self.labels)} known faces.")
    
    def search(self, q_emb: np.ndarray) -> Tuple[str, float]:
        if not self.embeddings:
            return ("Unknown", 0.0)
        E = np.stack(self.embeddings, axis=0)
        q = q_emb.astype(np.float32)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        qn = q / (np.linalg.norm(q) + 1e-8)
        sims = En @ qn
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        label = self.labels[idx] if best >= FACE_SIM_THRESHOLD else "Unknown"
        return (label, best)

face_index = FaceIndex()
face_index.build_from_dir(KNOWN_DIR)

# ---------------- Access Control System ----------------
class AccessControlSystem:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=1)
        self.last_triggered = 0
        self.last_seen_plates = {}
        self.last_seen_faces = {}
        self.last_unknown_embeddings = []
        self.activity_log = []
        self.running = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self.rtsp_reader, daemon=True)
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.capture_thread.start()
        self.processing_thread.start()
    
    def rtsp_reader(self):
        while self.running:
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.warning("Cannot open stream, retrying...")
                time.sleep(2)
                continue
            
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to grab frame, reconnecting...")
                    break
                
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            
            cap.release()
            time.sleep(1)
    
    def trigger_barrier(self, reason: str):
        now = time.time()
        if now - self.last_triggered < BARRIER_COOLDOWN:
            return
        
        logger.info(f"BARRIER TRIGGERED: {reason}")
        self.last_triggered = now
        
        # In a real implementation, this would trigger hardware
        # For simulation, we'll just log it
        self.log_access("barrier", "system", True, None, True)
    
    def log_access(self, access_type: str, identifier: str, authorized: bool, 
                  snapshot_path: str = None, barrier_triggered: bool = False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log to CSV
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, access_type, identifier, authorized, snapshot_path, barrier_triggered])
        
        # Log to database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO access_log (timestamp, access_type, identifier, authorized, snapshot_path, barrier_triggered) VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, access_type, identifier, authorized, snapshot_path, barrier_triggered)
        )
        conn.commit()
        conn.close()
        
        # Add to activity log
        self.activity_log.append({
            "timestamp": timestamp,
            "access_type": access_type,
            "identifier": identifier,
            "authorized": authorized,
            "snapshot": snapshot_path,
            "barrier_triggered": barrier_triggered
        })
        
        # Keep only recent activities
        if len(self.activity_log) > 100:
            self.activity_log = self.activity_log[-100:]
    
    def save_snapshot(self, frame, identifier: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{identifier}_{timestamp}.jpg"
        path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(path, frame)
        return path
    
    def save_unknown_face(self, frame, box):
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        # Check face quality
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < 100:  # Minimum sharpness threshold
            return None
        
        # Save the image
        ts = int(time.time() * 1000)
        fname = f"unknown_{ts}.jpg"
        fpath = os.path.join(UNKNOWN_DIR, fname)
        cv2.imwrite(fpath, crop)
        return f"/unknown_faces/{fname}"
    
    def process_license_plate(self, frame, box):
        x1, y1, x2, y2 = box
        plate_roi = frame[y1:y2, x1:x2]
        if plate_roi.size == 0:
            return None, False
        
        # OCR
        ocr_results = ocr_reader.readtext(plate_roi)
        plate_text = "".join([text.upper().replace(" ", "").replace(":", "")
                              for (_, text, _) in ocr_results])
        
        if not plate_text:
            return "UNKNOWN", False
        
        # Check authorization
        authorized = plate_text in AUTHORIZED_PLATES
        owner = AUTHORIZED_PLATES.get(plate_text, "Unknown")
        
        # Check if recently seen to avoid duplicates
        now = time.time()
        if plate_text in self.last_seen_plates and (now - self.last_seen_plates[plate_text] < 10):
            return plate_text, authorized
        
        self.last_seen_plates[plate_text] = now
        
        # Save snapshot and log
        snapshot_path = self.save_snapshot(frame, plate_text)
        self.log_access("license_plate", plate_text, authorized, snapshot_path)
        
        # Trigger barrier if authorized
        if authorized:
            self.trigger_barrier(f"Authorized plate: {plate_text} ({owner})")
        
        return plate_text, authorized
    
    def process_face(self, frame, box):
        if face_app is None:
            return "Unknown", False
        
        x1, y1, x2, y2 = box
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return "Unknown", False
        
        # Get face embedding
        try:
            persons = face_app.get(face_roi)
            if not persons:
                return "Unknown", False
            
            emb = persons[0].embedding
            label, sim = face_index.search(emb)
            
            # Check if recently seen to avoid duplicates
            now = time.time()
            if label in self.last_seen_faces and (now - self.last_seen_faces[label] < 10):
                return label, label != "Unknown"
            
            self.last_seen_faces[label] = now
            
            # Check authorization
            authorized = label in AUTHORIZED_FACES
            
            # Save snapshot and log
            snapshot_path = self.save_snapshot(frame, label)
            self.log_access("face", label, authorized, snapshot_path)
            
            # Trigger barrier if authorized
            if authorized:
                self.trigger_barrier(f"Authorized face: {label}")
            
            # Save unknown face if needed
            if label == "Unknown":
                unknown_path = self.save_unknown_face(frame, box)
                if unknown_path:
                    self.log_access("unknown_face", "unknown", False, unknown_path)
            
            return label, authorized
        except Exception as e:
            logger.error(f"Face processing error: {e}")
            return "Error", False
    
    def process_frames(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=2)
            except queue.Empty:
                continue
            
            # Process license plates
            plate_results = plate_detector(frame, verbose=False)
            for r in plate_results:
                if r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                for box in boxes:
                    plate_text, authorized = self.process_license_plate(frame, box)
                    
                    # Draw bounding box and label
                    color = (0, 255, 0) if authorized else (0, 0, 255)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{plate_text} ({'Authorized' if authorized else 'Unauthorized'})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Process faces
            if face_app is not None:
                faces = face_app.get(frame)
                for face in faces:
                    box = face.bbox.astype(int)
                    label, authorized = self.process_face(frame, box)
                    
                    # Draw bounding box and label
                    color = (0, 255, 0) if authorized else (0, 0, 255)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{label} ({'Authorized' if authorized else 'Unauthorized'})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display frame
            cv2.imshow("Access Control System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
    
    def stop(self):
        self.running = False
        self.capture_thread.join()
        self.processing_thread.join()
        cv2.destroyAllWindows()

# ---------------- Flask Web Interface ----------------
app = Flask(__name__)

# Initialize the access control system
access_system = AccessControlSystem()

@app.route('/')
def index():
    return '''
    <h1>Access Control System</h1>
    <p><a href="/video">Live Video Feed</a></p>
    <p><a href="/logs">Access Logs</a></p>
    <p><a href="/snapshots">Snapshots</a></p>
    <p><a href="/unknown_faces">Unknown Faces</a></p>
    '''

def generate_frames():
    while access_system.running:
        try:
            frame = access_system.frame_queue.get(timeout=1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except queue.Empty:
            continue

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def get_logs():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM access_log ORDER BY id DESC LIMIT 100")
    logs = cursor.fetchall()
    conn.close()
    
    # Convert to list of dictionaries
    logs_list = []
    for log in logs:
        logs_list.append({
            "id": log[0],
            "timestamp": log[1],
            "access_type": log[2],
            "identifier": log[3],
            "authorized": bool(log[4]),
            "snapshot_path": log[5],
            "barrier_triggered": bool(log[6])
        })
    
    return jsonify(logs_list)

@app.route('/snapshots')
def list_snapshots():
    files = os.listdir(SNAPSHOT_DIR)
    return jsonify(files)

@app.route('/snapshots/<filename>')
def get_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)

@app.route('/unknown_faces')
def list_unknown_faces():
    files = os.listdir(UNKNOWN_DIR)
    return jsonify(files)

@app.route('/unknown_faces/<filename>')
def get_unknown_face(filename):
    return send_from_directory(UNKNOWN_DIR, filename)

@app.route('/trigger_barrier', methods=['POST'])
def manual_trigger():
    access_system.trigger_barrier("Manual trigger")
    return jsonify({"status": "success", "message": "Barrier triggered"})

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    # Initialize database
    init_db()
    
    try:
        # Start Flask app
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        access_system.stop()
        logger.info("System stopped by user")