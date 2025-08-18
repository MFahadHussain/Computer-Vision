import os
import cv2
import time
import csv
import sqlite3
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ---------------- Directories ----------------
BASE_DIR = os.getcwd()
SNAP_DIR = os.path.join(BASE_DIR, "snapshots")
ATTENDANCE_SNAP_DIR = os.path.join(BASE_DIR, "attendance_snapshots")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_SNAP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------- Config ----------------
RTSP_URL = "rtsp://admin:afaqkhan-1@192.168.18.116:554/Streaming/channels/101"
AUTHORIZED_PLATES = {"EW080": "Fahad Hussain", "XYZ789": "Jane Smith"}  # plate -> owner mapping
PAD = 5  # px padding around plate
ATTENDANCE_DB = os.path.join(BASE_DIR, "attendance.db")
CSV_LOG = os.path.join(BASE_DIR, "plate_log.csv")
ATTENDANCE_DEDUP_WINDOW_SEC = 60
PLATE_DEDUP_WINDOW_SEC = 30

# ---------------- SQLite Setup ----------------
def init_db():
    con = sqlite3.connect(ATTENDANCE_DB)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        plate TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        snapshot_path TEXT
    )
    """)
    con.commit()
    con.close()
init_db()

# ---------------- Camera Thread ----------------
class CameraThread(threading.Thread):
    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.cap = None
        self.lock = threading.Lock()
        self.last_frame = None
        self.stop_event = threading.Event()

    def run(self):
        self.cap = cv2.VideoCapture(self.url)
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.last_frame = frame
            else:
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            return None if self.last_frame is None else self.last_frame.copy()

    def stop(self):
        self.stop_event.set()
        if self.cap:
            self.cap.release()

cam = CameraThread(RTSP_URL)
cam.start()

# ---------------- Face Recognition ----------------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))  # use GPU if available

# Load known faces
face_embeddings = {}  # name -> embedding
KNOWN_DIR = os.path.join(BASE_DIR, "known")
for f in os.listdir(KNOWN_DIR):
    if f.lower().endswith((".jpg", ".png")):
        img = cv2.imread(os.path.join(KNOWN_DIR, f))
        faces = face_app.get(img)
        if faces:
            name = os.path.splitext(f)[0]
            face_embeddings[name] = faces[0].embedding

def recognize_face(embedding):
    if not embedding.any():
        return "Unknown", 0.0
    best_label = "Unknown"
    best_sim = 0.0
    qn = embedding / (np.linalg.norm(embedding)+1e-8)
    for name, emb in face_embeddings.items():
        en = emb / (np.linalg.norm(emb)+1e-8)
        sim = en @ qn
        if sim > best_sim:
            best_sim = sim
            best_label = name
    if best_sim < 0.45:
        best_label = "Unknown"
    return best_label, best_sim

# ---------------- YOLO Plate Detection ----------------
yolo_plate = YOLO("yolov8n.pt")  # make sure you have YOLO trained for plates

last_seen_plate = {}
last_seen_face = {}

def save_snapshot(frame, prefix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}.jpg"
    fpath = os.path.join(SNAP_DIR if prefix=="plate" else ATTENDANCE_SNAP_DIR, fname)
    cv2.imwrite(fpath, frame)
    return fpath

def log_csv(plate, name, snapshot_path):
    file_exists = os.path.isfile(CSV_LOG)
    with open(CSV_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp","plate","name","snapshot"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), plate, name, snapshot_path])

def insert_attendance(name, plate, snapshot_path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    con = sqlite3.connect(ATTENDANCE_DB)
    cur = con.cursor()
    cur.execute("INSERT INTO attendance (name, plate, timestamp, snapshot_path) VALUES (?,?,?,?)",
                (name, plate, ts, snapshot_path))
    con.commit()
    con.close()

# ---------------- Main Loop ----------------
try:
    while True:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # Plate detection
        res = yolo_plate.predict(frame, conf=0.35)[0]
        if hasattr(res, "boxes") and res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1 = max(0, int(x1-PAD)), max(0, int(y1-PAD))
                x2, y2 = int(x2+PAD), int(y2+PAD)
                plate_crop = frame[y1:y2, x1:x2]
                # OCR can be added here to read plate_text
                plate_text = "ABC123"  # placeholder for OCR
                ts_now = time.time()
                if plate_text in last_seen_plate and ts_now - last_seen_plate[plate_text] < PLATE_DEDUP_WINDOW_SEC:
                    continue  # skip duplicate snapshot
                last_seen_plate[plate_text] = ts_now

                plate_authorized = plate_text in AUTHORIZED_PLATES
                owner_name = AUTHORIZED_PLATES.get(plate_text, None)

                # Face recognition in driver area (for simplicity, full frame here)
                faces = face_app.get(frame)
                face_authorized = False
                face_name = "Unknown"
                for f in faces:
                    emb = f.embedding
                    label, sim = recognize_face(emb)
                    if label != "Unknown":
                        face_name = label
                        # Deduplication
                        if label in last_seen_face and ts_now - last_seen_face[label] < ATTENDANCE_DEDUP_WINDOW_SEC:
                            continue
                        last_seen_face[label] = ts_now
                        if owner_name and owner_name == label:
                            face_authorized = True

                        # Save attendance snapshot
                        snap_path = save_snapshot(frame, "face")
                        insert_attendance(label, plate_text, snap_path)
                        log_csv(plate_text, label, snap_path)

                # Barrier control
                if plate_authorized and face_authorized:
                    print(f"[ACCESS GRANTED] {plate_text} | {face_name}")
                    # open_barrier()
                else:
                    print(f"[ALERT] Unauthorized access! Plate: {plate_text} | Face: {face_name}")

except KeyboardInterrupt:
    cam.stop()
    print("Stopped.")
