import os
import cv2
import time
import json
import math
import queue
import shutil
import signal
import base64
import logging
import sqlite3
import threading
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional
from flask import Flask, Response, jsonify, send_from_directory, request
from collections import deque
import gc
import re
import concurrent.futures
import imutils
from functools import lru_cache

# ---------------- Configuration ----------------
# Performance settings for lightweight operation
USE_GPU = bool(int(os.environ.get("USE_GPU", "0")))  # Default to CPU for lighter weight
ADAPTIVE_FRAME_SKIPPING = True
MIN_PROCESSING_INTERVAL = 0.05
MAX_WORKER_THREADS = 4  # Reduced for lower memory usage
FRAME_BUFFER_SIZE = 2  # Reduced buffer size

# Model settings
FACE_MODEL_SIZE = os.environ.get("FACE_MODEL_SIZE", "small")  # Use tiny model by default
FACE_RECOGNITION_THRESHOLD = float(os.environ.get("FACE_THRESHOLD", "0.6"))
PLATE_CONFIDENCE_THRESHOLD = float(os.environ.get("PLATE_CONFIDENCE", "0.7"))
MIN_PLATE_LENGTH = int(os.environ.get("MIN_PLATE_LENGTH", "5"))
MAX_PLATE_LENGTH = int(os.environ.get("MAX_PLATE_LENGTH", "10"))

# Camera settings
RTSP_FACE_URL = os.environ.get("RTSP_FACE_URL", "rtsp://admin:afaqkhan-1@192.168.0.104:554/Streaming/channels/101")
RTSP_PLATE_URL = os.environ.get("RTSP_PLATE_URL", "rtsp://admin:afaqkhan-1@192.168.0.102:554/Streaming/channels/102")
RTSP_BACKEND = os.environ.get("RTSP_BACKEND", "ffmpeg")
CAM_READ_TIMEOUT = 5.0
RECONNECT_DELAY = 2.0
FRAME_MAX_AGE = 2.0

# Recognition settings
SAVE_UNKNOWN_FACES = False  # Disabled to save disk space and memory
SAVE_UNKNOWN_PLATES = False  # Disabled to save disk space and memory
ATTENDANCE_DEDUP_WINDOW_SEC = 60
PLATE_DEDUP_WINDOW_SEC = 30
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.45"))
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "50"))
YOLO_PLATE_WEIGHTS = os.environ.get("YOLO_PLATE_WEIGHTS", "license_plate_detector.pt")
OCR_LANGS = os.environ.get("OCR_LANGS", "en").split(",")

# Lightweight settings
MAX_ACTIVITY_LOGS = 50  # Reduced from 200
MAX_DETECTIONS = 5  # Reduced from 10
MAX_KNOWN_FACES = 100  # Limit known faces to prevent memory bloat
DB_POOL_SIZE = 3  # Reduced connection pool
JPEG_QUALITY = 70  # Reduced image quality for smaller size

# Pakistani Plate Settings
AUTHORIZED_PLATES = {
    "AHA039": "Fahad Hussain",
    "KHT9090": "Ali Akbar",
    "PES5566": "Talal Syed"
}
# Matches: ABC1234, LE163432, KHT9090 etc.
plate_pattern = re.compile(r"[A-Z]{1,3}\d{1,4}|[A-Z]{1,3}\d{1,2}\d{1,4}")

# Directories
BASE_DIR = os.getcwd()
KNOWN_DIR = os.path.join(BASE_DIR, "known")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown")
PLATE_SNAP_DIR = os.path.join(BASE_DIR, "plates")
SNAP_DIR = os.path.join(BASE_DIR, "snapshots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
ATTENDANCE_DB = os.path.join(BASE_DIR, "attendance.db")
AUTHORIZED_FACES_DIR = os.path.join(BASE_DIR, "authorized_faces")

for d in (KNOWN_DIR, UNKNOWN_DIR, SNAP_DIR, PLATE_SNAP_DIR, LOGS_DIR, AUTHORIZED_FACES_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("face-plate-attendance")

# ---------------- Database Connection Pool ----------------
DB_POOL = []
DB_LOCK = threading.Lock()

def get_db_connection():
    with DB_LOCK:
        if DB_POOL:
            return DB_POOL.pop()
        else:
            conn = sqlite3.connect(ATTENDANCE_DB, check_same_thread=False, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            return conn

def return_db_connection(conn):
    with DB_LOCK:
        if len(DB_POOL) < DB_POOL_SIZE:
            DB_POOL.append(conn)
        else:
            conn.close()

# ---------------- Barrier Control ----------------
try:
    from barrier_control import open_barrier
    BARRIER_AVAILABLE = True
    logger.info("Barrier control module loaded")
except Exception as e:
    BARRIER_AVAILABLE = False
    def open_barrier():
        logger.info("[SIM] Barrier relay triggered")
    logger.warning(f"Barrier control module not available: {e}")

barrier_status = {
    "is_open": False,
    "opened_at": None,
    "opened_by": None,
    "face_authorized": False,
    "plate_authorized": False,
    "face_name": None,
    "plate_number": None,
    "reason": None,
    "face_image": None
}
barrier_lock = threading.Lock()
last_barrier_open_time = 0.0
BARRIER_COOLDOWN = int(os.environ.get("BARRIER_COOLDOWN", "5"))
BARRIER_OPEN_DURATION = 5

def safe_open_barrier(reason: str = "", face_name: str = None, plate_number: str = None, 
                     face_authorized: bool = False, plate_authorized: bool = False, face_image: str = None):
    global last_barrier_open_time, barrier_status
    now = time.time()
    
    with barrier_lock:
        if now - last_barrier_open_time < BARRIER_COOLDOWN:
            return
            
        last_barrier_open_time = now
        
        barrier_status = {
            "is_open": True,
            "opened_at": now,
            "opened_by": reason,
            "face_authorized": face_authorized,
            "plate_authorized": plate_authorized,
            "face_name": face_name,
            "plate_number": plate_number,
            "reason": reason,
            "face_image": face_image
        }
    
    threading.Thread(
        target=_open_barrier_thread,
        args=(reason, face_name, plate_number, face_authorized, plate_authorized, face_image),
        daemon=True
    ).start()

def _open_barrier_thread(reason: str, face_name: str, plate_number: str, 
                        face_authorized: bool, plate_authorized: bool, face_image: str):
    try:
        logger.info(f"Opening barrier: {reason}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(open_barrier)
            try:
                future.result(timeout=2.0)
                logger.info(f"Barrier opened successfully: {reason}")
                
                log_barrier_event(reason, face_name, plate_number, face_authorized, plate_authorized)
                threading.Timer(BARRIER_OPEN_DURATION, close_barrier).start()
            except concurrent.futures.TimeoutError:
                logger.error("Barrier operation timed out")
            except Exception as e:
                logger.error(f"Barrier operation failed: {e}")
    except Exception as e:
        logger.error(f"Error in barrier thread: {e}")

def close_barrier():
    global barrier_status
    try:
        with barrier_lock:
            barrier_status["is_open"] = False
        logger.info("Barrier closed")
    except Exception as e:
        logger.error(f"Barrier close failed: {e}")

# ---------------- Models ----------------
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    logger.warning(f"Ultralytics not available: {e}")

insightface = None
face_app = None

def init_insightface():
    global insightface, face_app
    import importlib
    insightface = importlib.import_module("insightface")
    from insightface.app import FaceAnalysis
    
    providers = ["CPUExecutionProvider"]  # Force CPU for lighter weight
    
    logger.info(f"Preparing InsightFace (buffalo_{FACE_MODEL_SIZE[0]}) ...")
    model_name = f"buffalo_{FACE_MODEL_SIZE[0]}"
    face_app = FaceAnalysis(name=model_name, providers=providers)
    face_app.prepare(ctx_id=-1, det_size=(640, 640))  # Use CPU
    try:
        face_app.models["detection"].threshold = 0.5
    except Exception:
        pass
    logger.info("InsightFace ready.")

plate_model = None
try:
    if YOLO is not None and os.path.isfile(YOLO_PLATE_WEIGHTS):
        plate_model = YOLO(YOLO_PLATE_WEIGHTS)
        logger.info(f"YOLO plate model loaded: {YOLO_PLATE_WEIGHTS}")
    else:
        if YOLO is None:
            logger.warning("YOLO not installed; plate detection will use Haar cascade only")
        else:
            logger.warning(f"Plate weights not found: {YOLO_PLATE_WEIGHTS}; plate detection will use Haar cascade only")
except Exception as e:
    logger.warning(f"Plate model load failed: {e}")

plate_cascade = None
try:
    if os.path.isfile(PLATE_CASCADE_PATH):
        plate_cascade = cv2.CascadeClassifier(PLATE_CASCADE_PATH)
        logger.info(f"Haar cascade loaded: {PLATE_CASCADE_PATH}")
    else:
        logger.warning(f"Haar cascade not found: {PLATE_CASCADE_PATH}")
except Exception as e:
    logger.warning(f"Failed to load Haar cascade: {e}")

_easyocr_reader = None
try:
    import easyocr
    _easyocr_reader = easyocr.Reader(OCR_LANGS, gpu=False)  # Force CPU
    logger.info(f"EasyOCR ready with langs: {OCR_LANGS}")
except Exception as e:
    logger.warning(f"EasyOCR unavailable ({e}); plate OCR disabled")

# ---------------- Known face index ----------------
class FaceIndex:
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.labels: List[str] = []
        self.lock = threading.Lock()
    
    def clear(self):
        with self.lock:
            self.embeddings = []
            self.labels = []
    
    def add(self, label: str, emb: np.ndarray):
        if emb is None:
            return
        with self.lock:
            # Limit the number of known faces to prevent memory bloat
            if len(self.labels) >= MAX_KNOWN_FACES:
                self.embeddings.pop(0)
                self.labels.pop(0)
            self.embeddings.append(emb.astype(np.float32))
            self.labels.append(label)
    
    def build_from_dir(self, folder: str):
        self.clear()
        logger.info(f"Building face index from: {folder}")
        count = 0
        for name in os.listdir(folder):
            if count >= MAX_KNOWN_FACES:
                logger.warning(f"Reached maximum known faces limit ({MAX_KNOWN_FACES})")
                break
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
            count += 1
        logger.info(f"Indexed {len(self.labels)} known faces.")
    
    def search(self, q_emb: np.ndarray) -> Tuple[str, float]:
        if not self.embeddings:
            return ("Unknown", 0.0)
        
        with self.lock:
            E = np.stack(self.embeddings, axis=0)
        
        q = q_emb.astype(np.float32)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        qn = q / (np.linalg.norm(q) + 1e-8)
        sims = En @ qn
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        
        dynamic_threshold = FACE_RECOGNITION_THRESHOLD
        if q_emb.std() < 0.1:
            dynamic_threshold += 0.1
            
        label = self.labels[idx] if best >= dynamic_threshold else "Unknown"
        return (label, best)

face_index = FaceIndex()

# ---------------- SQLite Functions ----------------
def init_db():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT UNIQUE NOT NULL,
                label TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS barrier_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                reason TEXT,
                face_name TEXT,
                plate_number TEXT,
                face_authorized INTEGER,
                plate_authorized INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS camera_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT
            )
        """)
        conn.commit()
    finally:
        return_db_connection(conn)

def insert_attendance(name: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO attendance (name, timestamp) VALUES (?,?)", (name, ts))
        conn.commit()
        return ts
    except Exception as e:
        logger.error(f"Failed to insert attendance: {e}")
        return ""
    finally:
        if conn:
            return_db_connection(conn)

def get_all_attendance(limit: int = 500):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name, timestamp FROM attendance ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [{"id": r[0], "name": r[1], "timestamp": r[2]} for r in rows]
    except Exception as e:
        logger.error(f"Failed to get attendance: {e}")
        return []
    finally:
        if conn:
            return_db_connection(conn)

def get_today_attendance_count():
    today_str = date.today().strftime("%Y-%m-%d")
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM attendance WHERE date(timestamp)=?", (today_str,))
        c = cur.fetchone()[0]
        return int(c)
    except Exception as e:
        logger.error(f"Failed to get today's attendance count: {e}")
        return 0
    finally:
        if conn:
            return_db_connection(conn)

def get_total_attendance_count():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM attendance")
        c = cur.fetchone()[0]
        return int(c)
    except Exception as e:
        logger.error(f"Failed to get total attendance count: {e}")
        return 0
    finally:
        if conn:
            return_db_connection(conn)

def list_plates() -> List[Dict[str, str]]:
    # Include both database plates and hardcoded authorized plates
    db_plates = []
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, plate, label FROM plates ORDER BY plate ASC")
        rows = cur.fetchall()
        db_plates = [{"id": r[0], "plate": r[1], "label": r[2] or ""} for r in rows]
    except Exception as e:
        logger.error(f"Failed to list plates from database: {e}")
    finally:
        if conn:
            return_db_connection(conn)
    
    # Add hardcoded authorized plates if not already in database
    hardcoded_plates = []
    for plate, name in AUTHORIZED_PLATES.items():
        if not any(p["plate"] == plate for p in db_plates):
            hardcoded_plates.append({"id": -1, "plate": plate, "label": name})
    
    return db_plates + hardcoded_plates

def add_plate(plate: str, label: str = "") -> Tuple[bool, str]:
    plate = plate.strip().upper()
    if len(plate) < MIN_PLATE_LENGTH:
        return False, "Plate too short"
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO plates (plate, label) VALUES (?,?)", (plate, label))
        conn.commit()
        return True, ""
    except sqlite3.IntegrityError:
        return False, "Plate already exists"
    except Exception as e:
        return False, str(e)
    finally:
        if conn:
            return_db_connection(conn)

def remove_plate(plate: str) -> Tuple[bool, str]:
    plate = plate.strip().upper()
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM plates WHERE plate=?", (plate,))
        conn.commit()
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        if conn:
            return_db_connection(conn)

@lru_cache(maxsize=100)
def is_authorized_plate(plate: str) -> bool:
    plate = plate.strip().upper()
    # Check hardcoded authorized plates first
    if plate in AUTHORIZED_PLATES:
        return True
    
    # Then check database
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM plates WHERE plate=?", (plate,))
        r = cur.fetchone()
        return r is not None
    except Exception as e:
        logger.error(f"Failed to check plate authorization: {e}")
        return False
    finally:
        if conn:
            return_db_connection(conn)

def get_plate_owner(plate: str) -> str:
    plate = plate.strip().upper()
    # Check hardcoded authorized plates first
    if plate in AUTHORIZED_PLATES:
        return AUTHORIZED_PLATES[plate]
    
    # Then check database
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT label FROM plates WHERE plate=?", (plate,))
        r = cur.fetchone()
        return r[0] if r else ""
    except Exception as e:
        logger.error(f"Failed to get plate owner: {e}")
        return ""
    finally:
        if conn:
            return_db_connection(conn)

def log_barrier_event(reason: str, face_name: str = None, plate_number: str = None, 
                      face_authorized: bool = False, plate_authorized: bool = False):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO barrier_events 
            (timestamp, event_type, reason, face_name, plate_number, face_authorized, plate_authorized)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ts, "barrier_open", reason, face_name, plate_number, int(face_authorized), int(plate_authorized)))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log barrier event: {e}")
    finally:
        if conn:
            return_db_connection(conn)

def get_barrier_events(limit: int = 100):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, timestamp, event_type, reason, face_name, plate_number, face_authorized, plate_authorized 
            FROM barrier_events 
            ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        return [{
            "id": r[0],
            "timestamp": r[1],
            "event_type": r[2],
            "reason": r[3],
            "face_name": r[4],
            "plate_number": r[5],
            "face_authorized": bool(r[6]),
            "plate_authorized": bool(r[7])
        } for r in rows]
    except Exception as e:
        logger.error(f"Failed to get barrier events: {e}")
        return []
    finally:
        if conn:
            return_db_connection(conn)

def log_camera_event(camera_name: str, event_type: str, details: str = ""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO camera_logs (camera_name, timestamp, event_type, details)
            VALUES (?, ?, ?, ?)
        """, (camera_name, ts, event_type, details))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log camera event: {e}")
    finally:
        if conn:
            return_db_connection(conn)

def get_camera_logs(camera_name: str = None, limit: int = 100):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        if camera_name:
            cur.execute("""
                SELECT id, camera_name, timestamp, event_type, details 
                FROM camera_logs 
                WHERE camera_name = ?
                ORDER BY id DESC LIMIT ?
            """, (camera_name, limit))
        else:
            cur.execute("""
                SELECT id, camera_name, timestamp, event_type, details 
                FROM camera_logs 
                ORDER BY id DESC LIMIT ?
            """, (limit,))
        rows = cur.fetchall()
        return [{
            "id": r[0],
            "camera_name": r[1],
            "timestamp": r[2],
            "event_type": r[3],
            "details": r[4]
        } for r in rows]
    except Exception as e:
        logger.error(f"Failed to get camera logs: {e}")
        return []
    finally:
        if conn:
            return_db_connection(conn)

# ---------------- Performance Monitoring ----------------
class PerformanceMonitor:
    def __init__(self, name: str):
        self.name = name
        self.process_times = deque(maxlen=50)  # Reduced from 100
        self.last_frame_time = time.time()
        self.last_cleanup = time.time()
        self.cleanup_interval = 300
        self.fps = 0.0
        
    def track_performance(self, start_time):
        process_time = time.time() - start_time
        self.process_times.append(process_time)
        
        now = time.time()
        elapsed = now - self.last_frame_time
        if elapsed > 1.0:
            self.fps = len(self.process_times) / elapsed
            self.last_frame_time = now
            
    def periodic_cleanup(self):
        now = time.time()
        if now - self.last_cleanup > self.cleanup_interval:
            gc.collect()
            self.last_cleanup = now
            logger.info(f"{self.name} performed memory cleanup")

# ---------------- Camera Thread ----------------
class CameraThread(threading.Thread):
    def __init__(self, url: str, name: str):
        super().__init__(daemon=True)
        self.url = url
        self.name = name
        self.cap = None
        self.lock = threading.Lock()
        self.last_frame = None
        self.last_ts = 0.0
        self.stop_event = threading.Event()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        
    def _open(self):
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"{self.name}: Max reconnection attempts reached")
            time.sleep(10)
            self.reconnect_attempts = 0
            
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        backend = cv2.CAP_FFMPEG if RTSP_BACKEND.lower().startswith("ffmpeg") else cv2.CAP_GSTREAMER
        self.cap = cv2.VideoCapture(self.url, backend)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 25)
        except Exception:
            pass
        self.reconnect_attempts += 1
        
    def run(self):
        logger.info(f"Starting capture thread: {self.name}")
        log_camera_event(self.name, "camera_start", f"Camera thread started")
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                self._open()
                time.sleep(0.2)
                if self.cap is None or not self.cap.isOpened():
                    logger.warning(f"{self.name}: capture not opened, retrying...")
                    time.sleep(RECONNECT_DELAY)
                    continue
                    
            ok, frame = self.cap.read()
            if not ok or frame is None:
                logger.warning(f"{self.name}: empty frame, reconnecting...")
                log_camera_event(self.name, "frame_error", "Empty frame received, reconnecting")
                time.sleep(RECONNECT_DELAY)
                self._open()
                continue
                
            with self.lock:
                self.last_frame = frame
                self.last_ts = time.time()
                self.frame_buffer.append((frame.copy(), time.time()))
                
        logger.info(f"Capture thread stopped: {self.name}")
        log_camera_event(self.name, "camera_stop", f"Camera thread stopped")
        
    def get_latest(self):
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer[-1]
            return (None, 0.0)
            
    def stop(self):
        self.stop_event.set()
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

# ---------------- Workers ----------------
class WorkerBase(threading.Thread):
    def __init__(self, cam: CameraThread, name: str):
        super().__init__(daemon=True)
        self.cam = cam
        self.name = name
        self.stop_event = threading.Event()
        self.curr_vis = None
        self.lock = threading.Lock()
        self.fps = 0.0
        self.monitor = PerformanceMonitor(name)
        self.activity_log: List[Dict] = []
        self.latest_public_detections: List[Dict] = []
        self.last_process_time = 0
        self.last_gc_time = time.time()
        
    def _push_activity(self, event: str, details: str = ""):
        self.activity_log.append({"timestamp": int(time.time() * 1000), "event": event, "details": details})
        if len(self.activity_log) > MAX_ACTIVITY_LOGS:
            self.activity_log = self.activity_log[-MAX_ACTIVITY_LOGS:]
            
    def get_vis(self):
        with self.lock:
            return None if self.curr_vis is None else self.curr_vis.copy()
            
    def get_public_detections(self):
        with self.lock:
            return list(self.latest_public_detections)
            
    def get_activity(self):
        with self.lock:
            return list(self.activity_log)
            
    def get_fps(self):
        return float(getattr(self.monitor, 'fps', 0.0))

class FaceWorker(WorkerBase):
    def __init__(self, cam: CameraThread):
        super().__init__(cam, "FaceWorker")
        self.last_seen_ts: Dict[str, float] = {}
        self._frame_skip = 0
        
    def _draw_bbox(self, img, box, label, sim, is_authorized):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if is_authorized else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        icon = "✓" if is_authorized else "✗"
        caption = f"{icon} {label} ({sim:.2f})" if label != "Unknown" else f"{icon} Unknown"
        cv2.putText(img, caption, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    def _save_authorized_crop(self, frame, box, name):
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ts = int(time.time() * 1000)
        fname = f"authorized_{name}_{ts}.jpg"
        fpath = os.path.join(AUTHORIZED_FACES_DIR, fname)
        ok, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            with open(fpath, 'wb') as f:
                f.write(enc.tobytes())
            return f"/authorized_faces/{fname}"
        return None
        
    def _maybe_log_attendance(self, name: str, frame, box):
        now = time.time()
        last = self.last_seen_ts.get(name, 0.0)
        if now - last >= ATTENDANCE_DEDUP_WINDOW_SEC:
            ts_str = insert_attendance(name)
            self.last_seen_ts[name] = now
            self._push_activity("attendance", f"{name} marked present at {ts_str}")
            log_camera_event("face_camera", "attendance", f"{name} marked present")
            return True
        return False
            
    def _detect_and_recognize(self, frame):
        public_dets = []
        vis = frame.copy()
        persons = []
        try:
            persons = face_app.get(frame)
        except Exception as e:
            logger.warning(f"InsightFace get() failed: {e}")
            log_camera_event("face_camera", "recognition_error", f"Face recognition failed: {str(e)}")
            
        for p in persons:
            x1, y1, x2, y2 = p.bbox.astype(int).tolist()
            if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                continue
                
            label, sim = "Unknown", 0.0
            try:
                emb = p.embedding
                if emb is not None:
                    label, sim = face_index.search(emb)
            except Exception:
                pass
                
            is_authorized = label != "Unknown"
            self._draw_bbox(vis, (x1, y1, x2, y2), label, sim, is_authorized)
            
            feed_item = {
                "type": "face",
                "name": label if is_authorized else None, 
                "timestamp": int(time.time()),
                "authorized": is_authorized,
                "confidence": sim
            }
            
            if is_authorized:
                logged = self._maybe_log_attendance(label, frame, (x1, y1, x2, y2))
                face_image_path = None
                if logged:
                    face_image_path = self._save_authorized_crop(frame, (x1, y1, x2, y2), label)
                    safe_open_barrier(
                        reason=f"face: {label}",
                        face_name=label,
                        face_authorized=True,
                        face_image=face_image_path
                    )
                
                feed_item["image_path"] = face_image_path
                
            public_dets.append(feed_item)
            
        return vis, public_dets
        
    def run(self):
        logger.info("FaceWorker started")
        log_camera_event("face_camera", "worker_start", "Face worker started")
        while not self.stop_event.is_set():
            start_time = time.time()
            frame, ts = self.cam.get_latest()
            if frame is None or (time.time() - ts > FRAME_MAX_AGE):
                time.sleep(0.01)
                continue
                
            if ADAPTIVE_FRAME_SKIPPING:
                if start_time - self.last_process_time < MIN_PROCESSING_INTERVAL:
                    time.sleep(0.001)
                    continue
                self.last_process_time = start_time
                
            try:
                vis, public_dets = self._detect_and_recognize(frame)
                with self.lock:
                    self.curr_vis = vis
                    self.latest_public_detections = public_dets[-MAX_DETECTIONS:]
            except Exception as e:
                logger.error(f"Error in face processing: {e}")
                
            self.monitor.track_performance(start_time)
            self.monitor.periodic_cleanup()
            
            if time.time() - self.last_gc_time > 60:
                gc.collect()
                self.last_gc_time = time.time()
                
        logger.info("FaceWorker stopped")
        log_camera_event("face_camera", "worker_stop", "Face worker stopped")

class PlateWorker(WorkerBase):
    def __init__(self, cam: CameraThread):
        super().__init__(cam, "PlateWorker")
        self.last_plate_ts: Dict[str, float] = {}
        self.plate_history = deque(maxlen=5)
        self._frame_skip = 0
        
    def _draw_plate(self, img, box, text: str, ok: bool):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 200, 0) if ok else (0, 0, 200)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        icon = "✓" if ok else "✗"
        label = text if text else "PLATE"
        cv2.putText(img, f"{icon} {label}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    def _save_plate_snapshot(self, frame, plate_text: str, is_authorized: bool):
        ts = int(time.time() * 1000)
        if is_authorized:
            fname = f"{plate_text}_{ts}.jpg"
        else:
            fname = f"ALERT_{plate_text}_{ts}.jpg"
        fpath = os.path.join(PLATE_SNAP_DIR, fname)
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            with open(fpath, 'wb') as f:
                f.write(enc.tobytes())
            return f"/plates/{fname}"
        return None
        
    def _ocr_plate(self, crop) -> str:
        if _easyocr_reader is None:
            return ""
            
        try:
            ocr_results = _easyocr_reader.readtext(crop)
            ocr_text = "".join([text.upper().replace(" ", "").replace(":", "")
                                for (_, text, _) in ocr_results])
            
            # Extract valid plate only
            match = plate_pattern.search(ocr_text)
            plate_text = match.group(0) if match else "UNKNOWN"
            
            return plate_text
        except Exception as e:
            logger.debug(f"OCR attempt failed: {e}")
            return ""
        
    def _temporal_filter(self, plate_text: str) -> str:
        if not plate_text:
            return ""
            
        self.plate_history.append(plate_text)
        
        plate_counts = {}
        for plate in self.plate_history:
            plate_counts[plate] = plate_counts.get(plate, 0) + 1
            
        if plate_counts:
            most_frequent = max(plate_counts.items(), key=lambda x: x[1])
            if most_frequent[1] >= 2:
                return most_frequent[0]
                
        return plate_text
        
    def _detect_and_read(self, frame):
        vis = frame.copy()
        public = []
        
        if plate_model is None:
            self._push_activity("plate_fallback", "No plate model available")
            return vis, public
            
        try:
            results = plate_model(frame, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                for (x1, y1, x2, y2) in boxes:
                    # Add padding
                    pad = 5
                    x1_p = max(0, x1 - pad)
                    y1_p = max(0, y1 - pad)
                    x2_p = min(frame.shape[1], x2 + pad)
                    y2_p = min(frame.shape[0], y2 + pad)
                    plate_roi = frame[y1_p:y2_p, x1_p:x2_p]
                    if plate_roi.size == 0:
                        continue
                        
                    # OCR
                    plate_text = self._ocr_plate(plate_roi)
                    filtered_text = self._temporal_filter(plate_text)
                    
                    if filtered_text == "UNKNOWN":
                        self._draw_plate(vis, (x1, y1, x2, y2), filtered_text, False)
                        continue
                        
                    # Check authorization
                    ok = is_authorized_plate(filtered_text)
                    owner = get_plate_owner(filtered_text)
                    
                    self._draw_plate(vis, (x1, y1, x2, y2), filtered_text, ok)
                    public.append({
                        "type": "plate", 
                        "plate": filtered_text, 
                        "owner": owner,
                        "ok": ok, 
                        "timestamp": int(time.time()),
                        "authorized": ok
                    })
                    
                    ts_last = self.last_plate_ts.get(filtered_text, 0.0)
                    now = time.time()
                    if ok and (now - ts_last >= PLATE_DEDUP_WINDOW_SEC):
                        self.last_plate_ts[filtered_text] = now
                        # Save snapshot
                        self._save_plate_snapshot(frame, filtered_text, True)
                        # Open barrier for authorized plate
                        safe_open_barrier(
                            reason=f"plate: {filtered_text}",
                            plate_number=filtered_text,
                            plate_authorized=True
                        )
                        self._push_activity("plate_ok", f"Authorized plate: {filtered_text} ({owner})")
                        log_camera_event("plate_camera", "authorized_plate", f"Authorized plate: {filtered_text} ({owner})")
                    elif not ok:
                        # Save snapshot for unauthorized plates
                        self._save_plate_snapshot(frame, filtered_text, False)
                        self._push_activity("plate_unknown", f"Unknown plate: {filtered_text}")
                        log_camera_event("plate_camera", "unknown_plate", f"Unknown plate: {filtered_text}")
                        
            if not boxes:
                self._push_activity("plate_fallback", "No plate detected")
                log_camera_event("plate_camera", "plate_fallback", "No plate detected")
                
        except Exception as e:
            logger.error(f"Error in plate processing: {e}")
            
        return vis, public
        
    def run(self):
        logger.info("PlateWorker started")
        log_camera_event("plate_camera", "worker_start", "Plate worker started")
        while not self.stop_event.is_set():
            start_time = time.time()
            frame, ts = self.cam.get_latest()
            if frame is None or (time.time() - ts > FRAME_MAX_AGE):
                time.sleep(0.01)
                continue
                
            if ADAPTIVE_FRAME_SKIPPING:
                if start_time - self.last_process_time < MIN_PROCESSING_INTERVAL:
                    time.sleep(0.001)
                    continue
                self.last_process_time = start_time
                
            try:
                vis, public = self._detect_and_read(frame)
                with self.lock:
                    self.curr_vis = vis
                    self.latest_public_detections = public[-MAX_DETECTIONS:]
            except Exception as e:
                logger.error(f"Error in plate processing: {e}")
                
            self.monitor.track_performance(start_time)
            self.monitor.periodic_cleanup()
            
            if time.time() - self.last_gc_time > 60:
                gc.collect()
                self.last_gc_time = time.time()
                
        logger.info("PlateWorker stopped")
        log_camera_event("plate_camera", "worker_stop", "Plate worker stopped")

# ---------------- Flask App ----------------
app = Flask(__name__)

# Global camera/workers
cam_face = CameraThread(RTSP_FACE_URL, name="face_cam")
cam_plate = CameraThread(RTSP_PLATE_URL, name="plate_cam")
face_worker = FaceWorker(cam_face)
plate_worker = PlateWorker(cam_plate)

# ---------------- HTML ----------------
INDEX_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <title>AI Security Dashboard</title>
  <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap\" rel=\"stylesheet\">
  <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css\"> 
  <style>
    :root { 
      --primary: #2563eb;
      --secondary: #3b82f6;
      --accent: #60a5fa;
      --success: #10b981;
      --danger: #ef4444;
      --warning: #f59e0b;
      --dark: #1e293b;
      --light: #f8fafc;
      --gray: #64748b;
      --card-bg: #ffffff;
      --text-primary: #0f172a;
      --text-secondary: #475569;
      --border: #e2e8f0;
      --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
      color: var(--text-primary);
      line-height: 1.6;
      min-height: 100vh;
    }
    
    .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 20px;
    }
    
    header {
      background: linear-gradient(135deg, var(--primary), var(--secondary));
      color: white;
      padding: 30px 0;
      border-radius: 16px;
      box-shadow: var(--shadow-lg);
      margin-bottom: 30px;
      text-align: center;
      position: relative;
      overflow: hidden;
    }
    
    header::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><rect width="100" height="100" fill="none"/><path d="M0,0 L100,100 M100,0 L0,100" stroke="rgba(255,255,255,0.1)" stroke-width="2"/></svg>');
      opacity: 0.1;
    }
    
    header h1 {
      font-size: 2.5rem;
      font-weight: 800;
      margin-bottom: 10px;
      position: relative;
      z-index: 1;
    }
    
    header p {
      font-size: 1.1rem;
      opacity: 0.9;
      position: relative;
      z-index: 1;
    }
    
    .dashboard-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
      gap: 20px;
      margin-bottom: 30px;
    }
    
    .card {
      background: var(--card-bg);
      border-radius: 12px;
      box-shadow: var(--shadow);
      overflow: hidden;
      transition: all 0.3s ease;
    }
    
    .card:hover {
      transform: translateY(-5px);
      box-shadow: var(--shadow-lg);
    }
    
    .card-header {
      background: linear-gradient(135deg, var(--secondary), var(--accent));
      color: white;
      padding: 15px 20px;
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .card-header h3 {
      font-size: 1.1rem;
      font-weight: 600;
    }
    
    .card-body {
      padding: 20px;
    }
    
    .status-indicator {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 8px;
      animation: pulse 2s infinite;
    }
    
    .status-live {
      background: var(--success);
      box-shadow: 0 0 8px var(--success);
    }
    
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
      70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
      100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    .video-container {
      position: relative;
      width: 100%;
      padding-bottom: 56.25%;
      height: 0;
      overflow: hidden;
      border-radius: 8px;
      background: #000;
    }
    
    .video-container img {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    
    .barrier-control {
      background: var(--card-bg);
      border-radius: 12px;
      padding: 25px;
      margin-bottom: 30px;
      box-shadow: var(--shadow-lg);
      text-align: center;
    }
    
    .barrier-status {
      padding: 20px;
      border-radius: 12px;
      margin-bottom: 20px;
      font-weight: 600;
      font-size: 1.3rem;
      transition: all 0.5s ease;
    }
    
    .barrier-open {
      background: linear-gradient(135deg, var(--success), #34d399);
      color: white;
    }
    
    .barrier-closed {
      background: var(--light);
      color: var(--text-secondary);
    }
    
    .barrier-status i {
      font-size: 2rem;
      margin-bottom: 10px;
      display: block;
    }
    
    .auth-status {
      display: flex;
      justify-content: space-around;
      margin-top: 15px;
    }
    
    .auth-item {
      text-align: center;
    }
    
    .auth-icon {
      font-size: 1.5rem;
      margin-bottom: 5px;
    }
    
    .auth-label {
      font-size: 0.9rem;
      opacity: 0.9;
    }
    
    .auth-authorized {
      color: var(--success);
    }
    
    .auth-unauthorized {
      color: var(--danger);
    }
    
    .auth-unknown {
      color: var(--gray);
    }
    
    .manual-control {
      display: flex;
      justify-content: center;
      gap: 15px;
      margin-top: 20px;
    }
    
    .btn {
      display: inline-block;
      background: var(--primary);
      color: white;
      border: none;
      padding: 12px 24px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 1rem;
      font-weight: 500;
      transition: all 0.3s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .btn:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-lg);
    }
    
    .btn-success {
      background: var(--success);
    }
    
    .btn-success:hover {
      background: #34d399;
    }
    
    .btn-danger {
      background: var(--danger);
    }
    
    .btn-danger:hover {
      background: #f87171;
    }
    
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 15px;
      margin-bottom: 30px;
    }
    
    .stat-card {
      background: var(--card-bg);
      border-radius: 12px;
      padding: 20px;
      box-shadow: var(--shadow);
      text-align: center;
      transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
      transform: translateY(-3px);
    }
    
    .stat-icon {
      width: 60px;
      height: 60px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 15px;
      font-size: 1.5rem;
    }
    
    .icon-primary {
      background: rgba(30, 58, 138, 0.1);
      color: var(--primary);
    }
    
    .icon-danger {
      background: rgba(239, 68, 68, 0.1);
      color: var(--danger);
    }
    
    .icon-success {
      background: rgba(16, 185, 129, 0.1);
      color: var(--success);
    }
    
    .icon-warning {
      background: rgba(245, 158, 11, 0.1);
      color: var(--warning);
    }
    
    .stat-value {
      font-size: 2rem;
      font-weight: 700;
      margin-bottom: 5px;
    }
    
    .stat-label {
      color: var(--text-secondary);
      font-size: 0.9rem;
    }
    
    .detection-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px;
      border-radius: 8px;
      margin-bottom: 10px;
      transition: all 0.3s ease;
    }
    
    .detection-item:hover {
      transform: translateX(5px);
    }
    
    .detection-item.authorized {
      background: rgba(16, 185, 129, 0.1);
      border-left: 4px solid var(--success);
    }
    
    .detection-item.unauthorized {
      background: rgba(239, 68, 68, 0.1);
      border-left: 4px solid var(--danger);
    }
    
    .detection-icon {
      font-size: 1.3rem;
      width: 40px;
      height: 40px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .detection-icon.authorized {
      background: var(--success);
      color: white;
    }
    
    .detection-icon.unauthorized {
      background: var(--danger);
      color: white;
    }
    
    .detection-details {
      flex: 1;
    }
    
    .detection-name {
      font-weight: 600;
      margin-bottom: 2px;
    }
    
    .detection-time {
      font-size: 0.8rem;
      color: var(--text-secondary);
    }
    
    .table-container {
      overflow-x: auto;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    
    table {
      width: 100%;
      border-collapse: collapse;
    }
    
    th, td {
      padding: 12px 15px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    
    th {
      background: var(--light);
      font-weight: 600;
      color: var(--text-primary);
    }
    
    tr:hover {
      background: rgba(0, 0, 0, 0.02);
    }
    
    .face-gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
      gap: 15px;
      margin-top: 15px;
    }
    
    .face-item {
      position: relative;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: var(--shadow);
      transition: transform 0.3s ease;
    }
    
    .face-item:hover {
      transform: scale(1.05);
    }
    
    .face-item img {
      width: 100%;
      height: 120px;
      object-fit: cover;
    }
    
    .face-item .label {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 5px;
      font-size: 0.8em;
      text-align: center;
    }
    
    .btn-sm {
      padding: 6px 12px;
      font-size: 0.875rem;
    }
    
    .input {
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.9rem;
      transition: border-color 0.3s ease;
      flex: 1;
    }
    
    .input:focus {
      outline: none;
      border-color: var(--secondary);
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .log-entry {
      display: flex;
      gap: 10px;
      margin-bottom: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }
    
    .log-time {
      color: var(--text-secondary);
      font-size: 0.85rem;
      min-width: 80px;
    }
    
    .log-content {
      flex: 1;
    }
    
    .log-event {
      font-weight: 600;
      margin-bottom: 2px;
    }
    
    .log-details {
      font-size: 0.9rem;
      color: var(--text-secondary);
    }
    
    .tabs {
      display: flex;
      margin-bottom: 20px;
      border-bottom: 2px solid var(--border);
    }
    
    .tab {
      padding: 12px 24px;
      cursor: pointer;
      border-bottom: 3px solid transparent;
      transition: all 0.3s ease;
      font-weight: 500;
      color: var(--text-secondary);
    }
    
    .tab:hover {
      background: rgba(0, 0, 0, 0.02);
    }
    
    .tab.active {
      border-bottom-color: var(--primary);
      color: var(--primary);
      font-weight: 600;
    }
    
    .tab-content {
      display: none;
    }
    
    .tab-content.active {
      display: block;
    }
    
    .flex {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    
    .settings-section {
      margin-bottom: 30px;
    }
    
    .settings-section h3 {
      margin-bottom: 15px;
      color: var(--primary);
      font-size: 1.2rem;
    }
    
    .form-group {
      margin-bottom: 15px;
    }
    
    .form-group label {
      display: block;
      margin-bottom: 5px;
      font-weight: 500;
    }
    
    .authorized-person {
      text-align: center;
      padding: 20px;
      background: rgba(16, 185, 129, 0.1);
      border-radius: 12px;
      margin-bottom: 20px;
    }
    
    .authorized-person img {
      max-width: 200px;
      border-radius: 50%;
      box-shadow: var(--shadow-lg);
      margin-bottom: 15px;
    }
    
    .authorized-person h3 {
      color: var(--success);
      font-size: 1.5rem;
      margin-bottom: 5px;
    }
    
    .unauthorized-banner {
      background: var(--danger);
      color: white;
      padding: 15px;
      text-align: center;
      border-radius: 8px;
      margin-bottom: 20px;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }
    
    .notification {
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 15px 20px;
      border-radius: 8px;
      color: white;
      font-weight: 500;
      z-index: 2000;
      animation: slideIn 0.3s ease;
    }
    
    .notification.success {
      background: var(--success);
    }
    
    .notification.error {
      background: var(--danger);
    }
    
    @keyframes slideIn {
      from {
        transform: translateX(100%);
        opacity: 0;
      }
      to {
        transform: translateX(0);
        opacity: 1;
      }
    }
    
    @media (max-width: 768px) {
      .dashboard-grid {
        grid-template-columns: 1fr;
      }
      
      .stats-grid {
        grid-template-columns: repeat(2, 1fr);
      }
      
      header h1 {
        font-size: 2rem;
      }
      
      .manual-control {
        flex-direction: column;
      }
      
      .btn {
        width: 100%;
        justify-content: center;
      }
      
      .tabs {
        flex-wrap: wrap;
      }
      
      .tab {
        padding: 8px 16px;
        font-size: 0.9rem;
      }
    }
  </style>
</head>
<body>
  <div class=\"container\">
    <header>
      <h1><i class=\"fas fa-shield-alt\"></i> AI Security Dashboard</h1>
      <p>Lightweight Face & Pakistani Plate Recognition System</p>
    </header>
    
    <div class=\"barrier-control\">
      <div id=\"barrier-status\" class=\"barrier-status barrier-closed\">
        <i class=\"fas fa-lock\"></i>
        <div>Barrier Status: <span id=\"barrier-text\">CLOSED</span></div>
        <div class=\"auth-status\">
          <div class=\"auth-item\">
            <div id=\"face-auth-icon\" class=\"auth-icon auth-unknown\"><i class=\"fas fa-user\"></i></div>
            <div class=\"auth-label\">Face: <span id=\"face-auth-text\">Unknown</span></div>
          </div>
          <div class=\"auth-item\">
            <div id=\"plate-auth-icon\" class=\"auth-icon auth-unknown\"><i class=\"fas fa-car\"></i></div>
            <div class=\"auth-label\">Plate: <span id=\"plate-auth-text\">Unknown</span></div>
          </div>
        </div>
      </div>
      
      <div class=\"manual-control\">
        <button class=\"btn btn-success\" onclick=\"manualOpenBarrier()\">
          <i class=\"fas fa-unlock\"></i> Open Barrier
        </button>
        <button class=\"btn btn-danger\" onclick=\"manualCloseBarrier()\">
          <i class=\"fas fa-lock\"></i> Close Barrier
        </button>
      </div>
    </div>
    
    <div id=\"unauthorized-banner\" class=\"unauthorized-banner\" style=\"display:none;\">
      <i class=\"fas fa-exclamation-triangle\"></i>
      <span>Unauthorized Access Attempt Detected</span>
    </div>
    
    <div id=\"authorized-person\" class=\"authorized-person\" style=\"display:none;\">
      <img id=\"authorized-person-img\" src=\"\" alt=\"Authorized Person\">
      <h3 id=\"authorized-person-name\"></h3>
      <p>Access Granted</p>
    </div>
    
    <div class=\"stats-grid\">
      <div class=\"stat-card\">
        <div class=\"stat-icon icon-primary\"><i class=\"fas fa-users\"></i></div>
        <div class=\"stat-value\" id=\"known-count\">0</div>
        <div class=\"stat-label\">Known Faces</div>
      </div>
      <div class=\"stat-card\">
        <div class=\"stat-icon icon-danger\"><i class=\"fas fa-user-slash\"></i></div>
        <div class=\"stat-value\" id=\"unknown-count\">0</div>
        <div class=\"stat-label\">Unknown Faces</div>
      </div>
      <div class=\"stat-card\">
        <div class=\"stat-icon icon-success\"><i class=\"fas fa-car\"></i></div>
        <div class=\"stat-value\" id=\"plate-count\">0</div>
        <div class=\"stat-label\">Authorized Plates</div>
      </div>
      <div class=\"stat-card\">
        <div class=\"stat-icon icon-warning\"><i class=\"fas fa-calendar-check\"></i></div>
        <div class=\"stat-value\" id=\"today-count\">0</div>
        <div class=\"stat-label\">Today's Attendance</div>
      </div>
    </div>
    
    <div class=\"tabs\">
      <div class=\"tab active\" onclick=\"switchTab('dashboard-tab')\"><i class=\"fas fa-tachometer-alt\"></i> Dashboard</div>
      <div class=\"tab\" onclick=\"switchTab('settings-tab')\"><i class=\"fas fa-cog\"></i> Settings</div>
    </div>
    
    <div id=\"dashboard-tab\" class=\"tab-content active\">
      <div class=\"dashboard-grid\">
        <div class=\"card\">
          <div class=\"card-header\">
            <h3><span class=\"status-indicator status-live\"></span>Face Camera</h3>
          </div>
          <div class=\"card-body\">
            <div class=\"video-container\">
              <img id=\"live-face\" src=\"/video_feed_face\" alt=\"Face Feed\">
            </div>
            <div style=\"margin-top: 15px;\">
              <h4>Recent Detections</h4>
              <div id=\"face-dets\" style=\"max-height: 200px; overflow-y: auto;\"></div>
            </div>
          </div>
        </div>
        
        <div class=\"card\">
          <div class=\"card-header\">
            <h3><span class=\"status-indicator status-live\"></span>Plate Camera</h3>
          </div>
          <div class=\"card-body\">
            <div class=\"video-container\">
              <img id=\"live-plate\" src=\"/video_feed_plate\" alt=\"Plate Feed\">
            </div>
            <div style=\"margin-top: 15px;\">
              <h4>Recent Detections</h4>
              <div id=\"plate-dets\" style=\"max-height: 200px; overflow-y: auto;\"></div>
            </div>
          </div>
        </div>
        
        <div class=\"card\">
          <div class=\"card-header\">
            <h3><i class=\"fas fa-list\"></i> All Detections</h3>
          </div>
          <div class=\"card-body\">
            <div id=\"combined-detections\" style=\"max-height: 300px; overflow-y: auto;\"></div>
          </div>
        </div>
        
        <div class=\"card\">
          <div class=\"card-header\">
            <h3><i class=\"fas fa-clipboard-list\"></i> Attendance</h3>
            <button class=\"btn btn-sm\" onclick=\"updateAttendance()\"><i class=\"fas fa-sync-alt\"></i></button>
          </div>
          <div class=\"card-body\">
            <div class=\"table-container\">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody id=\"attendance-table\"></tbody>
              </table>
            </div>
          </div>
        </div>
        
        <div class=\"card\">
          <div class=\"card-header\">
            <h3><i class=\"fas fa-history\"></i> Activity Log</h3>
            <button class=\"btn btn-sm\" onclick=\"updateActivity()\"><i class=\"fas fa-sync-alt\"></i></button>
          </div>
          <div class=\"card-body\">
            <div id=\"activity-feed\" style=\"max-height: 300px; overflow-y: auto;\"></div>
          </div>
        </div>
        
        <div class=\"card\">
          <div class=\"card-header\">
            <h3><i class=\"fas fa-door-open\"></i> Barrier Logs</h3>
            <button class=\"btn btn-sm\" onclick=\"updateBarrierLogs()\"><i class=\"fas fa-sync-alt\"></i></button>
          </div>
          <div class=\"card-body\">
            <div id=\"barrier-logs\" style=\"max-height: 300px; overflow-y: auto;\"></div>
          </div>
        </div>
      </div>
    </div>
    
    <div id=\"settings-tab\" class=\"tab-content\">
      <div class=\"card\">
        <div class=\"card-header\">
          <h3><i class=\"fas fa-user-cog\"></i> System Settings</h3>
        </div>
        <div class=\"card-body\">
          <div class=\"settings-section\">
            <h3><i class=\"fas fa-user-plus\"></i> Add New Face</h3>
            <form id=\"add-face-form\" enctype=\"multipart/form-data\">
              <div class=\"form-group\">
                <label>Name</label>
                <input type=\"text\" id=\"face-name\" class=\"input\" required>
              </div>
              <div class=\"form-group\">
                <label>Face Image</label>
                <input type=\"file\" id=\"face-image\" class=\"input\" accept=\"image/*\" required>
              </div>
              <button type=\"submit\" class=\"btn\"><i class=\"fas fa-plus\"></i> Add Face</button>
            </form>
          </div>
          
          <div class=\"settings-section\">
            <h3><i class=\"fas fa-car\"></i> Add Authorized Plate</h3>
            <div class=\"flex\" style=\"margin-bottom: 15px;\">
              <input id=\"plate-input\" class=\"input\" placeholder=\"Enter plate number\"> 
              <input id=\"plate-label\" class=\"input\" placeholder=\"Label (optional)\"> 
              <button class=\"btn\" onclick=\"addPlateUI()\"><i class=\"fas fa-plus\"></i> Add</button>
            </div>
            
            <div class=\"table-container\">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Plate</th>
                    <th>Label</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody id=\"plates-table\"></tbody>
              </table>
            </div>
          </div>
          
          <div class=\"settings-section\">
            <h3><i class=\"fas fa-users\"></i> Known Faces</h3>
            <div class=\"flex\" style=\"margin-bottom: 15px;\">
              <button class=\"btn\" onclick=\"reloadIndex()\"><i class=\"fas fa-sync-alt\"></i> Rebuild Index</button>
              <button class=\"btn\" onclick=\"loadKnownFaces()\"><i class=\"fas fa-refresh\"></i> Refresh</button>
            </div>
            <div class=\"face-gallery\" id=\"known-faces-gallery\"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
  
  <script>
    function showNotification(message, type = 'success') {
      const notification = document.createElement('div');
      notification.className = `notification ${type}`;
      notification.textContent = message;
      document.body.appendChild(notification);
      
      setTimeout(() => {
        notification.remove();
      }, 3000);
    }
    
    function switchTab(tabId) {
      document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
      });
      
      document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
      });
      
      document.getElementById(tabId).classList.add('active');
      event.target.classList.add('active');
    }
    
    function updateBarrierStatus() {
      fetch('/barrier_status')
        .then(r => r.json())
        .then(data => {
          const statusEl = document.getElementById('barrier-status');
          const barrierTextEl = document.getElementById('barrier-text');
          const faceIconEl = document.getElementById('face-auth-icon');
          const faceTextEl = document.getElementById('face-auth-text');
          const plateIconEl = document.getElementById('plate-auth-icon');
          const plateTextEl = document.getElementById('plate-auth-text');
          const authPersonEl = document.getElementById('authorized-person');
          const authPersonImgEl = document.getElementById('authorized-person-img');
          const authPersonNameEl = document.getElementById('authorized-person-name');
          const unauthorizedBannerEl = document.getElementById('unauthorized-banner');
          
          if (data.is_open) {
            statusEl.className = 'barrier-status barrier-open';
            barrierTextEl.textContent = 'OPEN';
          } else {
            statusEl.className = 'barrier-status barrier-closed';
            barrierTextEl.textContent = 'CLOSED';
          }
          
          if (data.face_authorized) {
            faceIconEl.className = 'auth-icon auth-authorized';
            faceTextEl.textContent = data.face_name || 'Authorized';
          } else {
            faceIconEl.className = 'auth-icon auth-unauthorized';
            faceTextEl.textContent = data.face_name || 'Unauthorized';
          }
          
          if (data.plate_authorized) {
            plateIconEl.className = 'auth-icon auth-authorized';
            plateTextEl.textContent = data.plate_number || 'Authorized';
          } else {
            plateIconEl.className = 'auth-icon auth-unauthorized';
            plateTextEl.textContent = data.plate_number || 'Unauthorized';
          }
          
          if (data.face_authorized && data.face_image) {
            authPersonEl.style.display = 'block';
            authPersonImgEl.src = data.face_image;
            authPersonNameEl.textContent = data.face_name;
          } else {
            authPersonEl.style.display = 'none';
          }
          
          if (!data.face_authorized && !data.plate_authorized) {
            unauthorizedBannerEl.style.display = 'flex';
          } else {
            unauthorizedBannerEl.style.display = 'none';
          }
        })
        .catch(() => {});
    }
    
    function updateBarrierLogs() {
      fetch('/barrier_logs')
        .then(r => r.json())
        .then(data => {
          const container = document.getElementById('barrier-logs');
          container.innerHTML = '';
          
          if (!data.length) {
            container.innerHTML = '<p style=\"text-align: center; color: var(--text-secondary);\">No barrier events</p>';
            return;
          }
          
          data.forEach(event => {
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            
            const time = document.createElement('div');
            time.className = 'log-time';
            time.textContent = event.timestamp;
            
            const content = document.createElement('div');
            content.className = 'log-content';
            
            let authInfo = '';
            if (event.face_authorized && event.plate_authorized) {
              authInfo = 'Both Face and Plate Authorized';
            } else if (event.face_authorized) {
              authInfo = 'Face Authorized';
            } else if (event.plate_authorized) {
              authInfo = 'Plate Authorized';
            }
            
            content.innerHTML = `
              <div class=\"log-event\">Barrier Opened</div>
              <div class=\"log-details\">${authInfo}</div>
              <div class=\"log-details\">Reason: ${event.reason || 'N/A'}</div>
            `;
            
            entry.appendChild(time);
            entry.appendChild(content);
            container.appendChild(entry);
          });
        })
        .catch(() => {});
    }
    
    function updateFaceDetections() {
      fetch('/detections_face')
        .then(r => r.json())
        .then(data => {
          const c = document.getElementById('face-dets');
          c.innerHTML = '';
          
          if (!data.length) {
            c.innerHTML = '<p style=\"text-align: center; color: var(--text-secondary);\">No recent detections</p>';
            return;
          }
          
          data.slice(-5).reverse().forEach(item => {
            const div = document.createElement('div');
            div.className = 'detection-item ' + (item.authorized ? 'authorized' : 'unauthorized');
            
            const icon = document.createElement('div');
            icon.className = 'detection-icon ' + (item.authorized ? 'authorized' : 'unauthorized');
            icon.innerHTML = item.authorized ? '<i class=\"fas fa-check\"></i>' : '<i class=\"fas fa-times\"></i>';
            
            const details = document.createElement('div');
            details.className = 'detection-details';
            details.innerHTML = `
              <div class=\"detection-name\">${item.name || 'Unknown'}</div>
              <div class=\"detection-time\">${new Date(item.timestamp * 1000).toLocaleTimeString()}</div>
            `;
            
            div.appendChild(icon);
            div.appendChild(details);
            c.appendChild(div);
          });
        })
        .catch(() => {});
    }
    
    function updatePlateDetections() {
      fetch('/detections_plate')
        .then(r => r.json())
        .then(data => {
          const c = document.getElementById('plate-dets');
          c.innerHTML = '';
          
          if (!data.length) {
            c.innerHTML = '<p style=\"text-align: center; color: var(--text-secondary);\">No recent detections</p>';
            return;
          }
          
          data.slice(-5).reverse().forEach(item => {
            const div = document.createElement('div');
            div.className = 'detection-item ' + (item.ok ? 'authorized' : 'unauthorized');
            
            const icon = document.createElement('div');
            icon.className = 'detection-icon ' + (item.ok ? 'authorized' : 'unauthorized');
            icon.innerHTML = item.ok ? '<i class=\"fas fa-check\"></i>' : '<i class=\"fas fa-times\"></i>';
            
            const details = document.createElement('div');
            details.className = 'detection-details';
            details.innerHTML = `
              <div class=\"detection-name\">${item.plate || 'Unknown Plate'}</div>
              <div class=\"detection-time\">${item.owner ? '(' + item.owner + ')' : ''}</div>
              <div class=\"detection-time\">${new Date(item.timestamp * 1000).toLocaleTimeString()}</div>
            `;
            
            div.appendChild(icon);
            div.appendChild(details);
            c.appendChild(div);
          });
        })
        .catch(() => {});
    }
    
    function updateCombinedDetections() {
      Promise.all([
        fetch('/detections_face').then(r => r.json()),
        fetch('/detections_plate').then(r => r.json())
      ]).then(([faceData, plateData]) => {
        const container = document.getElementById('combined-detections');
        container.innerHTML = '';
        
        const allDetections = [
          ...faceData.map(d => ({...d, source: 'face'})),
          ...plateData.map(d => ({...d, source: 'plate'}))
        ].sort((a, b) => b.timestamp - a.timestamp);
        
        if (!allDetections.length) {
          container.innerHTML = '<p style=\"text-align: center; color: var(--text-secondary);\">No recent detections</p>';
          return;
        }
        
        allDetections.slice(-10).forEach(item => {
          const div = document.createElement('div');
          div.className = 'detection-item ' + (item.authorized ? 'authorized' : 'unauthorized');
          
          const icon = document.createElement('div');
          icon.className = 'detection-icon ' + (item.authorized ? 'authorized' : 'unauthorized');
          icon.innerHTML = item.authorized ? '<i class=\"fas fa-check\"></i>' : '<i class=\"fas fa-times\"></i>';
          
          const details = document.createElement('div');
          details.className = 'detection-details';
          
          let name = '';
          if (item.source === 'face') {
            name = item.name || 'Unknown Face';
          } else {
            name = item.plate || 'Unknown Plate';
            if (item.owner) {
              name += ' (' + item.owner + ')';
            }
          }
          
          details.innerHTML = `
            <div class=\"detection-name\">
              <i class=\"fas fa-${item.source === 'face' ? 'user' : 'car'}\"></i> ${name}
            </div>
            <div class=\"detection-time\">${new Date(item.timestamp * 1000).toLocaleTimeString()}</div>
          `;
          
          div.appendChild(icon);
          div.appendChild(details);
          container.appendChild(div);
        });
      })
      .catch(() => {});
    }
    
    function updateAttendance() {
      fetch('/attendance')
        .then(r => r.json())
        .then(rows => {
          const tbody = document.getElementById('attendance-table');
          tbody.innerHTML = '';
          
          if (!rows.length) {
            tbody.innerHTML = '<tr><td colspan=\"3\" style=\"text-align: center; color: var(--text-secondary);\">No attendance records</td></tr>';
            return;
          }
          
          rows.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td>${row.id}</td>
              <td>${row.name}</td>
              <td>${row.timestamp}</td>
            `;
            tbody.appendChild(tr);
          });
        })
        .catch(() => {});
    }
    
    function updateActivity() {
      fetch('/activity_all')
        .then(r => r.json())
        .then(data => {
          const c = document.getElementById('activity-feed');
          c.innerHTML = '';
          
          if (!data.length) {
            c.innerHTML = '<p style=\"text-align: center; color: var(--text-secondary);\">No activity records</p>';
            return;
          }
          
          data.slice().reverse().forEach(item => {
            const div = document.createElement('div');
            div.className = 'log-entry';
            
            const time = document.createElement('div');
            time.className = 'log-time';
            time.textContent = new Date(item.timestamp).toLocaleTimeString();
            
            const content = document.createElement('div');
            content.className = 'log-content';
            content.innerHTML = `
              <div class=\"log-event\">${item.event}</div>
              <div class=\"log-details\">${item.details || ''}</div>
            `;
            
            div.appendChild(time);
            div.appendChild(content);
            c.appendChild(div);
          });
        })
        .catch(() => {});
    }
    
    function updateStats() {
      fetch('/stats')
        .then(r => r.json())
        .then(d => {
          document.getElementById('known-count').textContent = d.known_faces;
          document.getElementById('unknown-count').textContent = d.unknown_faces;
          document.getElementById('plate-count').textContent = d.authorized_plates;
          document.getElementById('today-count').textContent = d.today_attendance;
        })
        .catch(() => {});
    }
    
    function reloadIndex() {
      fetch('/reload_index', {method: 'POST'})
        .then(r => r.json())
        .then(d => {
          if (d.success) {
            showNotification('Index rebuilt successfully. Known faces: ' + d.count);
            loadKnownFaces();
            updateStats();
          } else {
            showNotification('Failed: ' + (d.error || 'unknown'), 'error');
          }
        })
        .catch(() => showNotification('Request failed', 'error'));
    }
    
    function loadKnownFaces() {
      fetch('/known_faces')
        .then(r => r.json())
        .then(data => {
          const g = document.getElementById('known-faces-gallery');
          g.innerHTML = '';
          
          if (!data.length) {
            g.innerHTML = '<p style=\"text-align: center; color: var(--text-secondary);\">No known faces</p>';
            return;
          }
          
          data.forEach(name => {
            const div = document.createElement('div');
            div.className = 'face-item';
            div.innerHTML = `
              <img src=\"/known/${name}.jpg\" alt=\"${name}\">
              <div class=\"label\">${name}</div>
            `;
            g.appendChild(div);
          });
        })
        .catch(() => {});
    }
    
    function loadPlates() {
      fetch('/plates')
        .then(r => r.json())
        .then(rows => {
          const tbody = document.getElementById('plates-table');
          tbody.innerHTML = '';
          
          if (!rows.length) {
            tbody.innerHTML = '<tr><td colspan=\"4\" style=\"text-align: center; color: var(--text-secondary);\">No authorized plates</td></tr>';
            return;
          }
          
          rows.forEach(r => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td>${r.id}</td>
              <td>${r.plate}</td>
              <td>${r.label || ''}</td>
              <td><button class=\"btn btn-sm btn-danger\" onclick=\"delPlate('${r.plate}')\">Delete</button></td>
            `;
            tbody.appendChild(tr);
          });
        })
        .catch(() => {});
    }
    
    function addPlateUI() {
      const p = document.getElementById('plate-input').value.trim();
      const l = document.getElementById('plate-label').value.trim();
      
      if (!p) {
        showNotification('Please enter a plate number', 'error');
        return;
      }
      
      fetch('/plates', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({plate: p, label: l})
      })
      .then(r => r.json())
      .then(d => {
        if (d.success) {
          document.getElementById('plate-input').value = '';
          document.getElementById('plate-label').value = '';
          loadPlates();
          updateStats();
          showNotification('Plate added successfully');
        } else {
          showNotification('Failed: ' + (d.error || 'unknown'), 'error');
        }
      })
      .catch(() => showNotification('Request failed', 'error'));
    }
    
    function delPlate(p) {
      if (!confirm('Delete plate ' + p + '?')) return;
      
      fetch('/plates/' + encodeURIComponent(p), {method: 'DELETE'})
        .then(r => r.json())
        .then(d => {
          if (d.success) {
            loadPlates();
            updateStats();
            showNotification('Plate deleted successfully');
          } else {
            showNotification('Failed: ' + (d.error || 'unknown'), 'error');
          }
        });
    }
    
    function manualOpenBarrier() {
      fetch('/manual_open_barrier', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({reason: 'Manual open'})
      })
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          showNotification('Barrier opened successfully');
        } else {
          showNotification('Failed to open barrier', 'error');
        }
      })
      .catch(error => {
        console.error('Error:', error);
        showNotification('Failed to open barrier', 'error');
      });
    }
    
    function manualCloseBarrier() {
      fetch('/manual_close_barrier', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
      })
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          showNotification('Barrier closed successfully');
        } else {
          showNotification('Failed to close barrier', 'error');
        }
      })
      .catch(error => {
        console.error('Error:', error);
        showNotification('Failed to close barrier', 'error');
      });
    }
    
    document.getElementById('add-face-form').addEventListener('submit', function(e) {
      e.preventDefault();
      
      const fd = new FormData();
      fd.append('name', document.getElementById('face-name').value);
      fd.append('image', document.getElementById('face-image').files[0]);
      
      fetch('/add_face', {
        method: 'POST',
        body: fd
      })
      .then(r => r.json())
      .then(data => {
        if (data.success) {
          showNotification('Face added successfully');
          document.getElementById('face-name').value = '';
          document.getElementById('face-image').value = '';
          loadKnownFaces();
          updateStats();
        } else {
          showNotification('Error: ' + data.error, 'error');
        }
      })
      .catch(() => showNotification('Error adding face', 'error'));
    });
    
    function refreshFeeds() {
      const faceImg = document.getElementById('live-face');
      const plateImg = document.getElementById('live-plate');
      
      if (faceImg) {
        faceImg.src = '/video_feed_face?' + Date.now();
      }
      
      if (plateImg) {
        plateImg.src = '/video_feed_plate?' + Date.now();
      }
    }
    
    setInterval(refreshFeeds, 1000);
    setInterval(updateBarrierStatus, 1000);
    setInterval(updateFaceDetections, 3000);
    setInterval(updatePlateDetections, 3000);
    setInterval(updateCombinedDetections, 3000);
    setInterval(updateAttendance, 5000);
    setInterval(updateActivity, 5000);
    setInterval(updateBarrierLogs, 5000);
    setInterval(updateStats, 10000);
    
    document.addEventListener('DOMContentLoaded', function() {
      refreshFeeds();
      updateBarrierStatus();
      updateFaceDetections();
      updatePlateDetections();
      updateCombinedDetections();
      updateAttendance();
      updateActivity();
      loadKnownFaces();
      loadPlates();
      updateStats();
      updateBarrierLogs();
    });
  </script>
</body>
</html>
"""

# ---------------- Routes ----------------
@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")

def _mjpeg_generator(get_frame_fn):
    boundary = b"--frame"
    while True:
        vis = get_frame_fn()
        if vis is None:
            time.sleep(0.01)
            continue
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        frame = buf.tobytes()
        yield (boundary + b"\r\n" + b"Content-Type: image/jpeg\r\n" + f"Content-Length: {len(frame)}\r\n\r\n".encode() + frame + b"\r\n")

@app.route("/video_feed_face")
def video_feed_face():
    return Response(_mjpeg_generator(face_worker.get_vis), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_plate")
def video_feed_plate():
    return Response(_mjpeg_generator(plate_worker.get_vis), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detections_face")
def detections_face():
    return jsonify(face_worker.get_public_detections())

@app.route("/detections_plate")
def detections_plate():
    return jsonify(plate_worker.get_public_detections())

@app.route("/activity_all")
def activity_all():
    a = face_worker.get_activity() + plate_worker.get_activity()
    a.sort(key=lambda x: x.get("timestamp", 0))
    return jsonify(a[-MAX_ACTIVITY_LOGS:])

@app.route("/attendance")
def attendance_route():
    return jsonify(get_all_attendance())

@app.route("/known_faces")
def known_faces():
    faces = []
    for name in os.listdir(KNOWN_DIR):
        p = os.path.join(KNOWN_DIR, name)
        if os.path.isfile(p):
            base, ext = os.path.splitext(name)
            if ext.lower() in [".jpg", ".jpeg", ".png"]:
                faces.append(base)
    faces.sort()
    return jsonify(faces)

@app.route('/known/<path:filename>')
def serve_known(filename):
    fpath = os.path.join(KNOWN_DIR, filename)
    if not os.path.isfile(fpath):
        return ("Not found", 404)
    return send_from_directory(KNOWN_DIR, filename)

@app.route('/authorized_faces/<path:filename>')
def serve_authorized_face(filename):
    fpath = os.path.join(AUTHORIZED_FACES_DIR, filename)
    if not os.path.isfile(fpath):
        return ("Not found", 404)
    return send_from_directory(AUTHORIZED_FACES_DIR, filename)

@app.route('/plates/<path:filename>')
def serve_plate_snap(filename):
    fpath = os.path.join(PLATE_SNAP_DIR, filename)
    if not os.path.isfile(fpath):
        return ("Not found", 404)
    return send_from_directory(PLATE_SNAP_DIR, filename)

@app.route("/stats")
def stats():
    known_count = len([f for f in os.listdir(KNOWN_DIR) if os.path.isfile(os.path.join(KNOWN_DIR, f))])
    unknown_count = len([f for f in os.listdir(UNKNOWN_DIR) if os.path.isfile(os.path.join(UNKNOWN_DIR, f))])
    total_att = get_total_attendance_count()
    today_att = get_today_attendance_count()
    auth_plates = len(list_plates())
    return jsonify({
        "known_faces": known_count,
        "unknown_faces": unknown_count,
        "total_attendance": total_att,
        "today_attendance": today_att,
        "authorized_plates": auth_plates,
    })

@app.route("/add_face", methods=["POST"])
def add_face_route():
    try:
        name = request.form.get("name", "").strip()
        imgfile = request.files.get("image")
        if not name or not imgfile:
            return jsonify({"success": False, "error": "Name and image required"}), 400
        safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        out_path = os.path.join(KNOWN_DIR, f"{safe_name}.jpg")
        arr = np.frombuffer(imgfile.read(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return jsonify({"success": False, "error": "Encode failed"}), 500
        with open(out_path, 'wb') as f:
            f.write(enc.tobytes())
        person = face_app.get(img)
        if not person:
            return jsonify({"success": False, "error": "No face found in image"}), 400
        face_index.add(safe_name, person[0].embedding)
        return jsonify({"success": True})
    except Exception as e:
        logger.exception("add_face failed")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/reload_index", methods=["POST"])
def reload_index():
    try:
        face_index.build_from_dir(KNOWN_DIR)
        return jsonify({"success": True, "count": len(face_index.labels)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/plates", methods=["GET"])
def api_list_plates():
    return jsonify(list_plates())

@app.route("/plates", methods=["POST"])
def api_add_plate():
    try:
        data = request.get_json(force=True)
        plate = data.get("plate", "").strip().upper()
        label = data.get("label", "").strip()
        ok, err = add_plate(plate, label)
        return jsonify({"success": ok, "error": err})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/plates/<plate>", methods=["DELETE"])
def api_del_plate(plate):
    ok, err = remove_plate(plate)
    status = 200 if ok else 400
    return jsonify({"success": ok, "error": err}), status

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/barrier_status")
def barrier_status_route():
    with barrier_lock:
        status = barrier_status.copy()
    
    if status["opened_at"]:
        status["opened_at"] = datetime.fromtimestamp(status["opened_at"]).strftime("%Y-%m-%d %H:%M:%S")
    
    return jsonify(status)

@app.route("/manual_open_barrier", methods=["POST"])
def manual_open_barrier():
    try:
        data = request.get_json()
        reason = data.get("reason", "Manual open")
        safe_open_barrier(reason=reason)
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Manual barrier open failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/manual_close_barrier", methods=["POST"])
def manual_close_barrier():
    try:
        close_barrier()
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Manual barrier close failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/barrier_logs")
def barrier_logs_route():
    limit = request.args.get("limit", 100, type=int)
    return jsonify(get_barrier_events(limit))

@app.route("/camera_logs")
@app.route("/camera_logs/<camera_name>")
def camera_logs_route(camera_name=None):
    limit = request.args.get("limit", 100, type=int)
    return jsonify(get_camera_logs(camera_name, limit))

# ---------------- Graceful Shutdown ----------------
stop_once = threading.Event()

def shutdown(*_):
    if stop_once.is_set():
        return
    stop_once.set()
    try:
        face_worker.stop_event.set()
        plate_worker.stop_event.set()
        cam_face.stop()
        cam_plate.stop()
    except Exception:
        pass

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ---------------- Boot ----------------
if __name__ == "__main__":
    init_db()
    init_insightface()
    face_index.build_from_dir(KNOWN_DIR)
    cam_face.start()
    cam_plate.start()
    face_worker.start()
    plate_worker.start()
    
    if BARRIER_AVAILABLE:
        logger.info("Testing barrier at startup...")
        safe_open_barrier(reason="System startup test")
    
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)