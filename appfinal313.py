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

# ---------------- Configuration ----------------
# Performance settings
USE_GPU = bool(int(os.environ.get("USE_GPU", "1")))
ADAPTIVE_FRAME_SKIPPING = True
MIN_PROCESSING_INTERVAL = 0.05  # Minimum time between processing frames
MAX_WORKER_THREADS = 4
FRAME_BUFFER_SIZE = 3

# Model settings
FACE_MODEL_SIZE = os.environ.get("FACE_MODEL_SIZE", "small")  # tiny, small, medium, large
PLATE_MODEL_SIZE = os.environ.get("PLATE_MODEL_SIZE", "nano")  # nano, small, medium
FACE_RECOGNITION_THRESHOLD = float(os.environ.get("FACE_THRESHOLD", "0.6"))
PLATE_CONFIDENCE_THRESHOLD = float(os.environ.get("PLATE_CONFIDENCE", "0.7"))
MIN_PLATE_ASPECT_RATIO = float(os.environ.get("MIN_PLATE_RATIO", "2.0"))  # Pakistan plates are wider
MAX_PLATE_ASPECT_RATIO = float(os.environ.get("MAX_PLATE_RATIO", "5.0"))

# Camera settings
RTSP_FACE_URL = os.environ.get("RTSP_FACE_URL", "rtsp://admin:afaqkhan-1@192.168.18.139:554/Streaming/channels/101")
RTSP_PLATE_URL = os.environ.get("RTSP_PLATE_URL", "rtsp://admin:afaqkhan-1@192.168.18.139:554/Streaming/channels/101")
RTSP_BACKEND = os.environ.get("RTSP_BACKEND", "ffmpeg")
CAM_READ_TIMEOUT = 5.0
RECONNECT_DELAY = 2.0
FRAME_MAX_AGE = 2.0

# Recognition settings
SAVE_UNKNOWN_FACES = True
SAVE_UNKNOWN_PLATES = True
ATTENDANCE_DEDUP_WINDOW_SEC = 60
PLATE_DEDUP_WINDOW_SEC = 30
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.45"))
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "50"))
YOLO_PLATE_WEIGHTS = os.environ.get("YOLO_PLATE_WEIGHTS", "license_plate_detector.pt")
OCR_LANGS = os.environ.get("OCR_LANGS", "en").split(",")
AUTHORIZED_PLATE_PATTERN_MINLEN = int(os.environ.get("PLATE_MINLEN", "6"))  # Pakistan plates are typically 6-7 chars

# Pakistan-specific settings
PAKISTAN_PROVINCE_CODES = {
    "ICT": "Islamabad Capital Territory",
    "ISB": "Islamabad",
    "RWP": "Rawalpindi",
    "LHR": "Lahore",
    "KHI": "Karachi",
    "PEW": "Peshawar",
    "SWL": "Sialkot",
    "FSD": "Faisalabad",
    "MULTAN": "Multan",
    "PESH": "Peshawar",
    "QUETTA": "Quetta",
    "HYD": "Hyderabad",
    "SUKKUR": "Sukkur",
    "DGK": "Dera Ghazi Khan",
    "BWP": "Bahawalpur",
    "SAHIWAL": "Sahiwal",
    "GUJRANWALA": "Gujranwala",
    "SARGODHA": "Sargodha",
    "MARDAN": "Mardan",
    "ABBOTABAD": "Abbottabad",
    "MINGORA": "Mingora",
    "GILGIT": "Gilgit",
    "SKARDU": "Skardu",
    "TURBAT": "Turbat"
}

# Pakistan plate patterns - expanded for better detection
PAKISTAN_PLATE_PATTERNS = [
    r'^[A-Z]{3}-\d{3}$',      # ABC-123 (most common)
    r'^[A-Z]{3}-\d{4}$',      # ABC-1234
    r'^[A-Z]{2}-\d{4}$',      # AB-1234
    r'^[A-Z]{3}\d{3}$',       # ABC123 (no dash)
    r'^[A-Z]{3}\d{4}$',       # ABC1234 (no dash)
    r'^[A-Z]{2}\d{4}$',       # AB1234 (no dash)
    r'^[A-Z]{2}-\d{3}[A-Z]$', # AB-123C (rare)
    r'^[A-Z]{3}-\d{3}-\d{1}$', # ABC-123-1 (Islamabad format)
    r'^\d{3}-[A-Z]{3}$',      # 123-ABC (military)
    r'^[A-Z]{2}\d{5}$',       # AB12345 (commercial)
    r'^[A-Z]{3}-\d{2}$',      # ABC-12 (some old plates)
    r'^[A-Z]{1}\d{4}$',       # A1234 (some special plates)
]

# Directories
BASE_DIR = os.getcwd()
KNOWN_DIR = os.path.join(BASE_DIR, "known")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown")
PLATE_SNAP_DIR = os.path.join(BASE_DIR, "plates")
SNAP_DIR = os.path.join(BASE_DIR, "snapshots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
ATTENDANCE_DB = os.path.join(BASE_DIR, "attendance.db")
for d in (KNOWN_DIR, UNKNOWN_DIR, SNAP_DIR, PLATE_SNAP_DIR, LOGS_DIR):
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
DB_POOL_SIZE = 5
DB_LOCK = threading.Lock()

def get_db_connection():
    with DB_LOCK:
        if not DB_POOL:
            conn = sqlite3.connect(ATTENDANCE_DB, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
            DB_POOL.append(conn)
        return DB_POOL.pop()

def return_db_connection(conn):
    with DB_LOCK:
        if len(DB_POOL) < DB_POOL_SIZE:
            DB_POOL.append(conn)
        else:
            conn.close()

# ---------------- Barrier Control ----------------
try:
    import barrier_control
    BARRIER_AVAILABLE = True
    logger.info("Barrier control module loaded")
except Exception as e:
    BARRIER_AVAILABLE = False
    class BarrierControl:
        def open_barrier(self):
            logger.info("[SIM] Barrier relay triggered")
        def close_barrier(self):
            logger.info("[SIM] Barrier closed")
        def timed_open_barrier(self, duration=5):
            logger.info(f"[SIM] Barrier opened for {duration} seconds")
    barrier_control = BarrierControl()
    logger.warning(f"Barrier control module not available: {e}")

# Barrier status tracking
barrier_status = {
    "is_open": False,
    "opened_at": None,
    "opened_by": None,
    "face_authorized": False,
    "plate_authorized": False,
    "face_name": None,
    "plate_number": None,
    "reason": None
}
barrier_lock = threading.Lock()
last_barrier_open_time = 0.0
BARRIER_COOLDOWN = int(os.environ.get("BARRIER_COOLDOWN", "5"))
BARRIER_OPEN_DURATION = 5  # seconds

def safe_open_barrier(reason: str = "", face_name: str = None, plate_number: str = None, 
                     face_authorized: bool = False, plate_authorized: bool = False, manual: bool = False):
    global last_barrier_open_time, barrier_status
    now = time.time()
    
    with barrier_lock:
        if now - last_barrier_open_time < BARRIER_COOLDOWN and not manual:
            logger.info(f"Barrier cooldown active, skipping open request")
            return
            
        last_barrier_open_time = now
        
        # Update barrier status
        barrier_status = {
            "is_open": True,
            "opened_at": now,
            "opened_by": reason,
            "face_authorized": face_authorized,
            "plate_authorized": plate_authorized,
            "face_name": face_name,
            "plate_number": plate_number,
            "reason": reason
        }
    
    # Run barrier opening in a separate thread to avoid blocking
    threading.Thread(
        target=_open_barrier_thread,
        args=(reason, face_name, plate_number, face_authorized, plate_authorized, manual),
        daemon=True
    ).start()

def _open_barrier_thread(reason: str, face_name: str, plate_number: str, 
                        face_authorized: bool, plate_authorized: bool, manual: bool):
    try:
        logger.info(f"Opening barrier: {reason}")
        
        # Use timeout for barrier operation
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(barrier_control.timed_open_barrier, BARRIER_OPEN_DURATION)
            try:
                future.result(timeout=BARRIER_OPEN_DURATION + 2)  # Add buffer time
                logger.info(f"Barrier opened successfully: {reason}")
                
                # Log barrier event
                log_barrier_event(reason, face_name, plate_number, face_authorized, plate_authorized, manual)
                
                # Update barrier status after it closes
                with barrier_lock:
                    barrier_status["is_open"] = False
            except concurrent.futures.TimeoutError:
                logger.error("Barrier operation timed out")
                # Try to close the barrier anyway
                try:
                    barrier_control.close_barrier()
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"Barrier operation failed: {e}")
    except Exception as e:
        logger.error(f"Error in barrier thread: {e}")

def close_barrier():
    global barrier_status
    try:
        # Call the actual barrier control function
        if BARRIER_AVAILABLE:
            barrier_control.close_barrier()
        else:
            logger.info("[SIM] Barrier closed")
        
        with barrier_lock:
            barrier_status["is_open"] = False
    except Exception as e:
        logger.error(f"Barrier close failed: {e}")

# ---------------- Async Image Savers ----------------
unknown_face_queue = queue.Queue(maxsize=100)
unknown_plate_queue = queue.Queue(maxsize=100)

def _async_image_saver(q: queue.Queue, outdir: str):
    while True:
        try:
            item = q.get(timeout=1)
            if item is None:
                break
            fname, crop = item
            fpath = os.path.join(outdir, fname)
            ok, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                with open(fpath, 'wb') as f:
                    f.write(enc.tobytes())
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Save failed {outdir}/{fname}: {e}")

threading.Thread(target=_async_image_saver, args=(unknown_face_queue, UNKNOWN_DIR), daemon=True).start()
threading.Thread(target=_async_image_saver, args=(unknown_plate_queue, PLATE_SNAP_DIR), daemon=True).start()

# ---------------- ONNX providers ----------------
def get_onnx_providers():
    try:
        from onnxruntime.capi._pybind_state import get_available_providers
        avail = get_available_providers()
        logger.info(f"ONNX providers available: {avail}")
        if USE_GPU and "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    except Exception as e:
        logger.warning(f"ONNX providers query failed, CPU only. Reason: {e}")
        return ["CPUExecutionProvider"]

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
    providers = get_onnx_providers()
    logger.info(f"Preparing InsightFace (buffalo_{FACE_MODEL_SIZE[0]}) ...")
    model_name = f"buffalo_{FACE_MODEL_SIZE[0]}"  # b_tiny, b_small, etc.
    face_app = FaceAnalysis(name=model_name, providers=providers)
    ctx_id = 0 if ("CUDAExecutionProvider" in providers) else -1
    face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
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
            logger.warning("YOLO not installed; plate detection will use OCR-only fallback")
        else:
            logger.warning(f"Plate weights not found: {YOLO_PLATE_WEIGHTS}; OCR-only fallback")
except Exception as e:
    logger.warning(f"Plate model load failed: {e}")

# OCR
_easyocr_reader = None
try:
    import easyocr
    _easyocr_reader = easyocr.Reader(OCR_LANGS)
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
        
        with self.lock:
            E = np.stack(self.embeddings, axis=0)
        
        q = q_emb.astype(np.float32)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        qn = q / (np.linalg.norm(q) + 1e-8)
        sims = En @ qn
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        
        # Dynamic threshold based on embedding quality
        dynamic_threshold = FACE_RECOGNITION_THRESHOLD
        if q_emb.std() < 0.1:  # Low quality embedding
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
                timestamp TEXT NOT NULL,
                plate_number TEXT,
                entry_type TEXT DEFAULT 'face'
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
                plate_authorized INTEGER,
                manual INTEGER DEFAULT 0
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

def insert_attendance(name: str, plate_number: str = None, entry_type: str = 'face') -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO attendance (name, timestamp, plate_number, entry_type) VALUES (?,?,?,?)", 
                   (name, ts, plate_number, entry_type))
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
        cur.execute("SELECT id, name, timestamp, plate_number, entry_type FROM attendance ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [{
            "id": r[0], 
            "name": r[1], 
            "timestamp": r[2],
            "plate_number": r[3],
            "entry_type": r[4]
        } for r in rows]
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
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, plate, label FROM plates ORDER BY plate ASC")
        rows = cur.fetchall()
        return [{"id": r[0], "plate": r[1], "label": r[2] or ""} for r in rows]
    except Exception as e:
        logger.error(f"Failed to list plates: {e}")
        return []
    finally:
        if conn:
            return_db_connection(conn)

def add_plate(plate: str, label: str = "") -> Tuple[bool, str]:
    plate = plate.strip().upper()
    if len(plate) < AUTHORIZED_PLATE_PATTERN_MINLEN:
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

def is_authorized_plate(plate: str) -> bool:
    plate = plate.strip().upper()
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

def log_barrier_event(reason: str, face_name: str = None, plate_number: str = None, 
                      face_authorized: bool = False, plate_authorized: bool = False, manual: bool = False):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO barrier_events 
            (timestamp, event_type, reason, face_name, plate_number, face_authorized, plate_authorized, manual)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, "barrier_open", reason, face_name, plate_number, int(face_authorized), int(plate_authorized), int(manual)))
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
            SELECT id, timestamp, event_type, reason, face_name, plate_number, face_authorized, plate_authorized, manual 
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
            "plate_authorized": bool(r[7]),
            "manual": bool(r[8])
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
        self.process_times = deque(maxlen=100)
        self.last_frame_time = time.time()
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
    def track_performance(self, start_time):
        process_time = time.time() - start_time
        self.process_times.append(process_time)
        
        # Calculate real FPS
        now = time.time()
        elapsed = now - self.last_frame_time
        if elapsed > 1.0:
            self.fps = len(self.process_times) / elapsed
            self.last_frame_time = now
            
    def periodic_cleanup(self):
        now = time.time()
        if now - self.last_cleanup > self.cleanup_interval:
            gc.collect()  # Force garbage collection
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
            time.sleep(10)  # Long pause before retrying
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
        if len(self.activity_log) > 200:
            self.activity_log = self.activity_log[-200:]
            
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
        
    def _draw_bbox(self, img, box, label, sim):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        caption = f"{label} ({sim:.2f})" if label != "Unknown" else "Unknown"
        cv2.putText(img, caption, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    def _save_unknown_crop(self, frame, box):
        if not SAVE_UNKNOWN_FACES:
            return None
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ts = int(time.time() * 1000)
        fname = f"unknown_{ts}.jpg"
        try:
            unknown_face_queue.put_nowait((fname, crop))
        except queue.Full:
            logger.warning("Unknown face queue full, dropping frame")
        return fname
        
    def _maybe_log_attendance(self, name: str):
        now = time.time()
        last = self.last_seen_ts.get(name, 0.0)
        if now - last >= ATTENDANCE_DEDUP_WINDOW_SEC:
            ts_str = insert_attendance(name, entry_type='face')
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
                
            self._draw_bbox(vis, (x1, y1, x2, y2), label, sim)
            feed_item = {
                "type": "known" if label != "Unknown" else "unknown", 
                "name": label if label != "Unknown" else None, 
                "timestamp": int(time.time())
            }
            
            if label == "Unknown":
                fname = self._save_unknown_crop(vis, (x1, y1, x2, y2))
                if fname:
                    feed_item["path"] = f"/unknown/{fname}"
                    self._push_activity("unknown_face", f"Unknown face captured: {fname}")
                    log_camera_event("face_camera", "unknown_face", f"Unknown face captured: {fname}")
            else:
                logged = self._maybe_log_attendance(label)
                if logged:
                    # Open barrier for authorized face
                    safe_open_barrier(
                        reason=f"face: {label}",
                        face_name=label,
                        face_authorized=True
                    )
                
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
                
            # Adaptive frame skipping
            if ADAPTIVE_FRAME_SKIPPING:
                if start_time - self.last_process_time < MIN_PROCESSING_INTERVAL:
                    time.sleep(0.001)
                    continue
                self.last_process_time = start_time
                
            # Process frame
            try:
                vis, public_dets = self._detect_and_recognize(frame)
                with self.lock:
                    self.curr_vis = vis
                    self.latest_public_detections = public_dets
            except Exception as e:
                logger.error(f"Error in face processing: {e}")
                
            # Performance monitoring
            self.monitor.track_performance(start_time)
            self.monitor.periodic_cleanup()
            
            # Periodic garbage collection
            if time.time() - self.last_gc_time > 60:
                gc.collect()
                self.last_gc_time = time.time()
                
        logger.info("FaceWorker stopped")
        log_camera_event("face_camera", "worker_stop", "Face worker stopped")

class PlateWorker(WorkerBase):
    def __init__(self, cam: CameraThread):
        super().__init__(cam, "PlateWorker")
        self.last_plate_ts: Dict[str, float] = {}
        self.plate_history = deque(maxlen=10)  # Store recent plate readings for temporal filtering
        self._frame_skip = 0
        
    def _draw_plate(self, img, box, text: str, ok: bool):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 200, 0) if ok else (0, 0, 200)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = text if text else "PLATE"
        cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    def _validate_plate(self, text: str, box: List[int]) -> bool:
        # Basic format validation
        if len(text) < AUTHORIZED_PLATE_PATTERN_MINLEN or len(text) > 10:
            return False
            
        # Aspect ratio check - Pakistan plates are wider
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        aspect = width / height if height > 0 else 0
        
        if aspect < MIN_PLATE_ASPECT_RATIO or aspect > MAX_PLATE_ASPECT_RATIO:
            return False
            
        # Character composition check - Pakistan plates have both letters and digits
        if not any(c.isdigit() for c in text) or not any(c.isalpha() for c in text):
            return False
            
        # Check against Pakistan plate patterns
        if any(re.match(pattern, text) for pattern in PAKISTAN_PLATE_PATTERNS):
            return True
            
        return False
        
    def _get_province_info(self, plate_text: str) -> Tuple[str, str]:
        """Extract province information from plate text"""
        plate_text = plate_text.replace('-', '').upper()
        
        # Check for province codes
        for code, name in PAKISTAN_PROVINCE_CODES.items():
            if plate_text.startswith(code):
                return code, name
                
        # Check for common patterns
        if plate_text.startswith('L'):
            return 'LHR', 'Lahore'
        elif plate_text.startswith('K'):
            return 'KHI', 'Karachi'
        elif plate_text.startswith('I'):
            return 'ICT', 'Islamabad'
        elif plate_text.startswith('R'):
            return 'RWP', 'Rawalpindi'
        elif plate_text.startswith('G'):
            return 'Gujranwala', 'Gujranwala'
        elif plate_text.startswith('S'):
            return 'SWL', 'Sialkot'
        elif plate_text.startswith('F'):
            return 'FSD', 'Faisalabad'
        elif plate_text.startswith('M'):
            return 'MULTAN', 'Multan'
        elif plate_text.startswith('P'):
            return 'PESH', 'Peshawar'
        elif plate_text.startswith('Q'):
            return 'QUETTA', 'Quetta'
        elif plate_text.startswith('H'):
            return 'HYD', 'Hyderabad'
            
        return 'UNKNOWN', 'Unknown Province'
        
    def _enhance_plate_region(self, crop):
        """Apply multiple enhancement techniques to improve OCR accuracy for Pakistan plates"""
        enhanced = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced.append(gray)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        enhanced.append(bilateral)
        
        # Gaussian blur + threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced.append(thresh)
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        enhanced.append(adaptive)
        
        # Morphological operations - specifically for Pakistan plates
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        enhanced.append(morph)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        enhanced.append(edges)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        enhanced.append(clahe_img)
        
        # Additional preprocessing for better OCR
        # Resize to make text larger
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        enhanced.append(resized)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        enhanced.append(denoised)
        
        return enhanced
        
    def _ocr_plate(self, crop) -> str:
        if _easyocr_reader is None:
            return ""
            
        best_text = ""
        best_conf = 0
        
        # Try multiple enhanced versions
        enhanced_images = self._enhance_plate_region(crop)
        
        for img in enhanced_images:
            try:
                # Use detail=0 for faster processing
                results = _easyocr_reader.readtext(img, detail=1, 
                                                  paragraph=False, 
                                                  batch_size=1,
                                                  contrast_ths=0.5,
                                                  adjust_contrast=0.5,
                                                  filter_thres=0.003,
                                                  allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')  # Only allow plate characters
                for _, text, conf in results:
                    # Clean the text - Pakistan plates can have dashes
                    text = text.strip().upper()
                    # Remove non-alphanumeric characters except dash
                    text = re.sub(r'[^A-Z0-9-]', '', text)
                    
                    # Validate basic format
                    if len(text) < AUTHORIZED_PLATE_PATTERN_MINLEN:
                        continue
                        
                    # Use the highest confidence result
                    if conf > best_conf and conf > PLATE_CONFIDENCE_THRESHOLD:
                        best_conf = conf
                        best_text = text
                        
            except Exception as e:
                logger.debug(f"OCR attempt failed: {e}")
                continue
                
        return best_text
        
    def _temporal_filter(self, plate_text: str) -> str:
        """Apply temporal filtering to stabilize plate readings"""
        if not plate_text:
            return ""
            
        # Add current reading to history
        self.plate_history.append(plate_text)
        
        # Count occurrences of each plate in recent history
        plate_counts = {}
        for plate in self.plate_history:
            plate_counts[plate] = plate_counts.get(plate, 0) + 1
            
        # Find the most frequent plate
        if plate_counts:
            most_frequent = max(plate_counts.items(), key=lambda x: x[1])
            # Only return if it appears at least twice in recent history
            if most_frequent[1] >= 2:
                return most_frequent[0]
                
        return plate_text
        
    def _detect_and_read(self, frame):
        vis = frame.copy()
        public = []
        det_boxes = []
        
        # detect plates
        if plate_model is not None:
            try:
                res = plate_model.predict(frame, verbose=False, conf=0.30, iou=0.45)[0]
                boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
                det_boxes.extend([[int(v) for v in b] for b in boxes])
            except Exception as e:
                logger.debug(f"YOLO plate predict failed: {e}")
                log_camera_event("plate_camera", "detection_error", f"Plate detection failed: {str(e)}")
        else:
            # Fallback: try whole frame OCR (less accurate)
            h, w = frame.shape[:2]
            det_boxes.append([w//4, h//3, 3*w//4, 2*h//3])
            
        seen_any = False
        for (x1, y1, x2, y2) in det_boxes:
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
            if x2 - x1 < 30 or y2 - y1 < 15:
                continue
                
            crop = frame[y1:y2, x1:x2]
            text = self._ocr_plate(crop)
            
            # Apply temporal filtering
            filtered_text = self._temporal_filter(text)
            
            if not filtered_text or not self._validate_plate(filtered_text, [x1, y1, x2, y2]):
                self._draw_plate(vis, (x1, y1, x2, y2), filtered_text, False)
                continue
                
            seen_any = True
            ok = is_authorized_plate(filtered_text)
            self._draw_plate(vis, (x1, y1, x2, y2), filtered_text, ok)
            public.append({"plate": filtered_text, "ok": ok, "timestamp": int(time.time())})
            
            ts_last = self.last_plate_ts.get(filtered_text, 0.0)
            now = time.time()
            if ok and (now - ts_last >= PLATE_DEDUP_WINDOW_SEC):
                self.last_plate_ts[filtered_text] = now
                
                # Get province info
                province_code, province_name = self._get_province_info(filtered_text)
                
                # Log attendance for plate
                insert_attendance(f"Vehicle: {filtered_text}", filtered_text, 'plate')
                
                # Open barrier for authorized plate
                safe_open_barrier(
                    reason=f"plate: {filtered_text} ({province_name})",
                    plate_number=filtered_text,
                    plate_authorized=True
                )
                
                if SAVE_UNKNOWN_PLATES:
                    fname = f"plate_{filtered_text}_{int(now*1000)}.jpg"
                    try:
                        unknown_plate_queue.put_nowait((fname, crop))
                    except queue.Full:
                        logger.warning("Unknown plate queue full, dropping frame")
                        
                self._push_activity("plate_ok", f"Authorized plate: {filtered_text} from {province_name}")
                log_camera_event("plate_camera", "authorized_plate", f"Authorized plate: {filtered_text} from {province_name}")
                
            elif not ok:
                if SAVE_UNKNOWN_PLATES:
                    fname = f"unknown_plate_{int(time.time()*1000)}.jpg"
                    try:
                        unknown_plate_queue.put_nowait((fname, crop))
                    except queue.Full:
                        logger.warning("Unknown plate queue full, dropping frame")
                        
                # Get province info for unknown plates too
                province_code, province_name = self._get_province_info(filtered_text)
                self._push_activity("plate_unknown", f"Unknown plate: {filtered_text} from {province_name}")
                log_camera_event("plate_camera", "unknown_plate", f"Unknown plate: {filtered_text} from {province_name}")
                
        if not seen_any and plate_model is None:
            self._push_activity("plate_fallback", "OCR-only fallback used; no plate box")
            log_camera_event("plate_camera", "plate_fallback", "OCR-only fallback used; no plate box")
            
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
                
            # Adaptive frame skipping
            if ADAPTIVE_FRAME_SKIPPING:
                if start_time - self.last_process_time < MIN_PROCESSING_INTERVAL:
                    time.sleep(0.001)
                    continue
                self.last_process_time = start_time
                
            # Process frame
            try:
                vis, public = self._detect_and_read(frame)
                with self.lock:
                    self.curr_vis = vis
                    self.latest_public_detections = public[-10:]
            except Exception as e:
                logger.error(f"Error in plate processing: {e}")
                
            # Performance monitoring
            self.monitor.track_performance(start_time)
            self.monitor.periodic_cleanup()
            
            # Periodic garbage collection
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
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Security Dashboard - Pakistan</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }
            100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
        }
        .status-live {
            animation: pulse 2s infinite;
        }
        .glass-effect {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card-hover {
            transition: all 0.3s ease;
        }
        .card-hover:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        .barrier-open {
            background: linear-gradient(135deg, #10b981, #059669);
        }
        .barrier-closed {
            background: linear-gradient(135deg, #ef4444, #dc2626);
        }
    </style>
</head>
<body class="bg-gray-100">
    <!-- Header -->
    <header class="gradient-bg text-white py-6 shadow-lg">
        <div class="container mx-auto px-4">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-3xl font-bold flex items-center">
                        <i class="fas fa-shield-alt mr-3"></i>
                        AI Security Dashboard - Pakistan
                    </h1>
                    <p class="text-blue-100 mt-2">Face Recognition & Pakistan Number Plate Detection</p>
                </div>
                <div class="flex items-center space-x-4">
                    <div class="text-right">
                        <div class="text-sm text-blue-100">System Status</div>
                        <div class="flex items-center">
                            <span class="w-3 h-3 bg-green-400 rounded-full status-live mr-2"></span>
                            <span class="font-semibold">Online</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <div class="container mx-auto px-4 py-6">
        <!-- Stats Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex items-center">
                    <div class="p-3 rounded-full bg-blue-100 text-blue-600 mr-4">
                        <i class="fas fa-users text-2xl"></i>
                    </div>
                    <div>
                        <p class="text-gray-500 text-sm">Known Faces</p>
                        <p class="text-2xl font-bold" id="known-count">0</p>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex items-center">
                    <div class="p-3 rounded-full bg-red-100 text-red-600 mr-4">
                        <i class="fas fa-user-slash text-2xl"></i>
                    </div>
                    <div>
                        <p class="text-gray-500 text-sm">Unknown Faces</p>
                        <p class="text-2xl font-bold" id="unknown-count">0</p>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex items-center">
                    <div class="p-3 rounded-full bg-green-100 text-green-600 mr-4">
                        <i class="fas fa-car text-2xl"></i>
                    </div>
                    <div>
                        <p class="text-gray-500 text-sm">Authorized Plates</p>
                        <p class="text-2xl font-bold" id="plate-count">0</p>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-xl shadow-md p-6 card-hover">
                <div class="flex items-center">
                    <div class="p-3 rounded-full bg-purple-100 text-purple-600 mr-4">
                        <i class="fas fa-calendar-check text-2xl"></i>
                    </div>
                    <div>
                        <p class="text-gray-500 text-sm">Today's Attendance</p>
                        <p class="text-2xl font-bold" id="today-count">0</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Barrier Status & Control -->
        <div class="bg-white rounded-xl shadow-md p-6 mb-8">
            <div class="flex flex-col md:flex-row justify-between items-center">
                <div class="mb-4 md:mb-0 flex-1">
                    <h2 class="text-xl font-bold text-gray-800 mb-2">Barrier Control</h2>
                    <div id="barrier-status" class="flex items-center">
                        <div id="barrier-indicator" class="w-4 h-4 rounded-full mr-2"></div>
                        <span id="barrier-text" class="font-semibold text-lg"></span>
                        <div class="ml-4 text-sm text-gray-600">
                            <span id="face-auth-icon" class="mr-2"><i class="fas fa-user-circle"></i> <span id="face-auth-text">Unknown</span></span>
                            <span id="plate-auth-icon" class="ml-4"><i class="fas fa-car"></i> <span id="plate-auth-text">Unknown</span></span>
                        </div>
                    </div>
                </div>
                <div class="flex space-x-3">
                    <button onclick="manualOpenBarrier()" class="bg-green-500 hover:bg-green-600 text-white font-bold py-3 px-6 rounded-lg transition duration-300 transform hover:scale-105">
                        <i class="fas fa-lock-open mr-2"></i>Open Barrier
                    </button>
                    <button onclick="manualCloseBarrier()" class="bg-red-500 hover:bg-red-600 text-white font-bold py-3 px-6 rounded-lg transition duration-300 transform hover:scale-105">
                        <i class="fas fa-lock mr-2"></i>Close Barrier
                    </button>
                </div>
            </div>
        </div>

        <!-- Camera Feeds -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Face Camera -->
            <div class="bg-white rounded-xl shadow-md overflow-hidden card-hover">
                <div class="bg-gray-800 text-white p-4 flex justify-between items-center">
                    <h3 class="font-bold flex items-center">
                        <span class="w-3 h-3 bg-green-400 rounded-full status-live mr-2"></span>
                        Face Camera
                    </h3>
                    <span class="text-sm" id="fps-face">FPS: 0</span>
                </div>
                <div class="relative bg-black" style="padding-bottom: 56.25%;">
                    <img id="live-face" src="/video_feed_face" alt="Face Feed" class="absolute top-0 left-0 w-full h-full object-contain">
                </div>
                <div class="p-4">
                    <h4 class="font-semibold mb-2">Recent Detections</h4>
                    <div id="face-dets" class="space-y-2 max-h-32 overflow-y-auto"></div>
                </div>
            </div>

            <!-- Plate Camera -->
            <div class="bg-white rounded-xl shadow-md overflow-hidden card-hover">
                <div class="bg-gray-800 text-white p-4 flex justify-between items-center">
                    <h3 class="font-bold flex items-center">
                        <span class="w-3 h-3 bg-green-400 rounded-full status-live mr-2"></span>
                        Plate Camera
                    </h3>
                    <span class="text-sm" id="fps-plate">FPS: 0</span>
                </div>
                <div class="relative bg-black" style="padding-bottom: 56.25%;">
                    <img id="live-plate" src="/video_feed_plate" alt="Plate Feed" class="absolute top-0 left-0 w-full h-full object-contain">
                </div>
                <div class="p-4">
                    <h4 class="font-semibold mb-2">Recent Plate Reads</h4>
                    <div id="plate-dets" class="space-y-2 max-h-32 overflow-y-auto"></div>
                </div>
            </div>
        </div>

        <!-- Tabs Section -->
        <div class="bg-white rounded-xl shadow-md p-6">
            <div class="flex flex-wrap border-b border-gray-200 mb-4">
                <button onclick="switchTab('attendance')" class="tab-btn px-4 py-2 font-semibold text-blue-600 border-b-2 border-blue-600 focus:outline-none" data-tab="attendance">
                    <i class="fas fa-clipboard-list mr-2"></i>Attendance
                </button>
                <button onclick="switchTab('authorized-log')" class="tab-btn px-4 py-2 font-semibold text-gray-600 hover:text-blue-600 focus:outline-none" data-tab="authorized-log">
                    <i class="fas fa-history mr-2"></i>Authorized Log
                </button>
                <button onclick="switchTab('activity')" class="tab-btn px-4 py-2 font-semibold text-gray-600 hover:text-blue-600 focus:outline-none" data-tab="activity">
                    <i class="fas fa-stream mr-2"></i>Activity Log
                </button>
                <button onclick="switchTab('known-faces')" class="tab-btn px-4 py-2 font-semibold text-gray-600 hover:text-blue-600 focus:outline-none" data-tab="known-faces">
                    <i class="fas fa-users mr-2"></i>Known Faces
                </button>
                <button onclick="switchTab('plates')" class="tab-btn px-4 py-2 font-semibold text-gray-600 hover:text-blue-600 focus:outline-none" data-tab="plates">
                    <i class="fas fa-car mr-2"></i>Authorized Plates
                </button>
            </div>

            <!-- Tab Contents -->
            <div id="attendance" class="tab-content">
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Plate</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                            </tr>
                        </thead>
                        <tbody id="attendance-table" class="bg-white divide-y divide-gray-200">
                            <!-- Dynamic content -->
                        </tbody>
                    </table>
                </div>
            </div>

            <div id="authorized-log" class="tab-content hidden">
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Event</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Details</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                            </tr>
                        </thead>
                        <tbody id="authorized-log-table" class="bg-white divide-y divide-gray-200">
                            <!-- Dynamic content -->
                        </tbody>
                    </table>
                </div>
            </div>

            <div id="activity" class="tab-content hidden">
                <div id="activity-feed" class="space-y-3 max-h-96 overflow-y-auto">
                    <!-- Dynamic content -->
                </div>
            </div>

            <div id="known-faces" class="tab-content hidden">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold">Known Faces</h3>
                    <div class="space-x-2">
                        <button onclick="openAddFace()" class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition duration-300">
                            <i class="fas fa-plus mr-2"></i>Add Face
                        </button>
                        <button onclick="reloadIndex()" class="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition duration-300">
                            <i class="fas fa-sync-alt mr-2"></i>Rebuild Index
                        </button>
                    </div>
                </div>
                <div id="known-faces-gallery" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
                    <!-- Dynamic content -->
                </div>
            </div>

            <div id="plates" class="tab-content hidden">
                <div class="mb-4">
                    <div class="flex flex-col md:flex-row md:items-end space-y-2 md:space-y-0 md:space-x-4">
                        <div class="flex-1">
                            <label class="block text-sm font-medium text-gray-700 mb-1">Plate Number</label>
                            <input id="plate-input" type="text" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="e.g. ABC-123">
                        </div>
                        <div class="flex-1">
                            <label class="block text-sm font-medium text-gray-700 mb-1">Label (Optional)</label>
                            <input id="plate-label" type="text" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="Vehicle owner">
                        </div>
                        <div class="flex space-x-2">
                            <button onclick="addPlateUI()" class="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition duration-300">
                                <i class="fas fa-plus mr-2"></i>Add
                            </button>
                            <button onclick="loadPlates()" class="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition duration-300">
                                <i class="fas fa-sync-alt mr-2"></i>Refresh
                            </button>
                        </div>
                    </div>
                </div>
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Plate</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Label</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th>
                            </tr>
                        </thead>
                        <tbody id="plates-table" class="bg-white divide-y divide-gray-200">
                            <!-- Dynamic content -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- Add Face Modal -->
    <div id="add-face-modal" class="fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center z-50">
        <div class="bg-white rounded-xl p-6 w-full max-w-md mx-4">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-xl font-bold">Add New Face</h3>
                <button onclick="closeAddFace()" class="text-gray-500 hover:text-gray-700">
                    <i class="fas fa-times text-xl"></i>
                </button>
            </div>
            <form id="add-face-form" enctype="multipart/form-data">
                <div class="mb-4">
                    <label class="block text-sm font-medium text-gray-700 mb-1">Name</label>
                    <input type="text" id="face-name" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" required>
                </div>
                <div class="mb-4">
                    <label class="block text-sm font-medium text-gray-700 mb-1">Face Image</label>
                    <input type="file" id="face-image" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" accept="image/*" required>
                </div>
                <button type="submit" class="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition duration-300">
                    Add Face
                </button>
            </form>
        </div>
    </div>

    <script>
        // Tab switching functionality
        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.add('hidden');
            });
            
            // Remove active state from all tab buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('text-blue-600', 'border-b-2', 'border-blue-600');
                btn.classList.add('text-gray-600');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.remove('hidden');
            
            // Add active state to clicked tab button
            const activeBtn = document.querySelector(`[data-tab="${tabName}"]`);
            activeBtn.classList.remove('text-gray-600');
            activeBtn.classList.add('text-blue-600', 'border-b-2', 'border-blue-600');
            
            // Load content for specific tabs
            if (tabName === 'attendance') updateAttendance();
            if (tabName === 'authorized-log') updateAuthorizedLog();
            if (tabName === 'activity') updateActivity();
            if (tabName === 'known-faces') loadKnownFaces();
            if (tabName === 'plates') loadPlates();
        }

        // Manual barrier open
        function manualOpenBarrier() {
            if (confirm('Are you sure you want to manually open the barrier?')) {
                fetch('/manual_open_barrier', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Barrier opened successfully!');
                            updateBarrierStatus();
                            updateAuthorizedLog();
                        } else {
                            alert('Failed to open barrier: ' + data.error);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Failed to open barrier');
                    });
            }
        }

        // Manual barrier close
        function manualCloseBarrier() {
            if (confirm('Are you sure you want to manually close the barrier?')) {
                fetch('/manual_close_barrier', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Barrier closed successfully!');
                            updateBarrierStatus();
                        } else {
                            alert('Failed to close barrier: ' + data.error);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Failed to close barrier');
                    });
            }
        }

        // Update functions
        function updateBarrierStatus() {
            fetch('/barrier_status')
                .then(r => r.json())
                .then(data => {
                    const indicator = document.getElementById('barrier-indicator');
                    const text = document.getElementById('barrier-text');
                    const statusDiv = document.getElementById('barrier-status');
                    const faceIcon = document.getElementById('face-auth-icon');
                    const faceText = document.getElementById('face-auth-text');
                    const plateIcon = document.getElementById('plate-auth-icon');
                    const plateText = document.getElementById('plate-auth-text');
                    
                    if (data.is_open) {
                        indicator.className = 'w-4 h-4 rounded-full bg-green-500 mr-2';
                        text.textContent = 'Barrier Open';
                        statusDiv.className = 'barrier-open text-white p-4 rounded-lg';
                    } else {
                        indicator.className = 'w-4 h-4 rounded-full bg-red-500 mr-2';
                        text.textContent = 'Barrier Closed';
                        statusDiv.className = 'barrier-closed text-white p-4 rounded-lg';
                    }
                    
                    // Update face auth status
                    if (data.face_authorized) {
                        faceIcon.innerHTML = '<i class="fas fa-user-check text-green-500"></i>';
                        faceText.textContent = data.face_name || 'Authorized';
                    } else {
                        faceIcon.innerHTML = '<i class="fas fa-user-times text-red-500"></i>';
                        faceText.textContent = data.face_name || 'Unauthorized';
                    }
                    
                    // Update plate auth status
                    if (data.plate_authorized) {
                        plateIcon.innerHTML = '<i class="fas fa-car text-green-500"></i>';
                        plateText.textContent = data.plate_number || 'Authorized';
                    } else {
                        plateIcon.innerHTML = '<i class="fas fa-car text-red-500"></i>';
                        plateText.textContent = data.plate_number || 'Unauthorized';
                    }
                })
                .catch(() => {});
        }

        function updateAuthorizedLog() {
            fetch('/barrier_logs')
                .then(r => r.json())
                .then(data => {
                    const tbody = document.getElementById('authorized-log-table');
                    tbody.innerHTML = '';
                    
                    if (!data.length) {
                        tbody.innerHTML = '<tr><td colspan="4" class="text-center py-4 text-gray-500">No barrier events</td></tr>';
                        return;
                    }
                    
                    data.forEach(event => {
                        const tr = document.createElement('tr');
                        let typeBadge = '';
                        let authInfo = '';
                        
                        if (event.manual) {
                            typeBadge = '<span class="px-2 py-1 text-xs font-semibold rounded-full bg-purple-100 text-purple-800">Manual</span>';
                            authInfo = 'Manual Trigger';
                        } else if (event.face_authorized && event.plate_authorized) {
                            typeBadge = '<span class="px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">Both</span>';
                            authInfo = 'Face & Plate Authorized';
                        } else if (event.face_authorized) {
                            typeBadge = '<span class="px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800">Face</span>';
                            authInfo = 'Face Authorized';
                        } else if (event.plate_authorized) {
                            typeBadge = '<span class="px-2 py-1 text-xs font-semibold rounded-full bg-yellow-100 text-yellow-800">Plate</span>';
                            authInfo = 'Plate Authorized';
                        }
                        
                        tr.innerHTML = `
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${event.timestamp}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Barrier Opened</td>
                            <td class="px-6 py-4 text-sm text-gray-900">
                                <div>${authInfo}</div>
                                <div class="text-xs text-gray-500">${event.reason || 'N/A'}</div>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap">${typeBadge}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                })
                .catch(() => {});
        }

        function refreshFaceFeed() {
            const img = document.getElementById('live-face');
            if (img) {
                img.src = '/video_feed_face?' + Date.now();
            }
            fetch('/fps_face')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('fps-face').textContent = 'FPS: ' + d.fps.toFixed(1);
                })
                .catch(() => {});
        }

        function refreshPlateFeed() {
            const img = document.getElementById('live-plate');
            if (img) {
                img.src = '/video_feed_plate?' + Date.now();
            }
            fetch('/fps_plate')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('fps-plate').textContent = 'FPS: ' + d.fps.toFixed(1);
                })
                .catch(() => {});
        }

        function updateFaceDetections() {
            fetch('/detections_face')
                .then(r => r.json())
                .then(data => {
                    const container = document.getElementById('face-dets');
                    container.innerHTML = '';
                    
                    if (!data.length) {
                        container.innerHTML = '<p class="text-gray-500 text-sm">No recent detections</p>';
                        return;
                    }
                    
                    data.slice(-5).reverse().forEach(item => {
                        const div = document.createElement('div');
                        div.className = 'flex items-center justify-between p-2 bg-gray-50 rounded';
                        
                        const badge = document.createElement('span');
                        badge.className = `px-2 py-1 text-xs font-semibold rounded-full ${
                            item.type === 'known' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        }`;
                        badge.textContent = item.type === 'known' ? 'Known' : 'Unknown';
                        
                        const name = document.createElement('span');
                        name.className = 'text-sm font-medium';
                        name.textContent = item.type === 'known' ? item.name : 'Unknown';
                        
                        const time = document.createElement('span');
                        time.className = 'text-xs text-gray-500';
                        time.textContent = new Date(item.timestamp * 1000).toLocaleTimeString();
                        
                        div.appendChild(badge);
                        div.appendChild(name);
                        div.appendChild(time);
                        container.appendChild(div);
                    });
                })
                .catch(() => {});
        }

        function updatePlateDetections() {
            fetch('/detections_plate')
                .then(r => r.json())
                .then(data => {
                    const container = document.getElementById('plate-dets');
                    container.innerHTML = '';
                    
                    if (!data.length) {
                        container.innerHTML = '<p class="text-gray-500 text-sm">No recent reads</p>';
                        return;
                    }
                    
                    data.slice(-5).reverse().forEach(item => {
                        const div = document.createElement('div');
                        div.className = 'flex items-center justify-between p-2 bg-gray-50 rounded';
                        
                        const badge = document.createElement('span');
                        badge.className = `px-2 py-1 text-xs font-semibold rounded-full ${
                            item.ok ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        }`;
                        badge.textContent = item.ok ? 'Authorized' : 'Unknown';
                        
                        const plate = document.createElement('span');
                        plate.className = 'text-sm font-medium';
                        plate.textContent = item.plate || '(blank)';
                        
                        const time = document.createElement('span');
                        time.className = 'text-xs text-gray-500';
                        time.textContent = new Date(item.timestamp * 1000).toLocaleTimeString();
                        
                        div.appendChild(badge);
                        div.appendChild(plate);
                        div.appendChild(time);
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
                        tbody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-gray-500">No attendance records</td></tr>';
                        return;
                    }
                    
                    rows.forEach(row => {
                        const tr = document.createElement('tr');
                        const typeBadge = row.entry_type === 'face' 
                            ? '<span class="px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800">Face</span>'
                            : '<span class="px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">Plate</span>';
                            
                        tr.innerHTML = `
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${row.id}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${row.name}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${row.plate_number || '-'}</td>
                            <td class="px-6 py-4 whitespace-nowrap">${typeBadge}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${row.timestamp}</td>
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
                    const container = document.getElementById('activity-feed');
                    container.innerHTML = '';
                    
                    if (!data.length) {
                        container.innerHTML = '<p class="text-gray-500 text-center py-4">No activity records</p>';
                        return;
                    }
                    
                    data.slice().reverse().forEach(item => {
                        const div = document.createElement('div');
                        div.className = 'flex items-start space-x-3 p-3 bg-gray-50 rounded-lg';
                        
                        const time = document.createElement('div');
                        time.className = 'text-xs text-gray-500 whitespace-nowrap';
                        time.textContent = new Date(item.timestamp).toLocaleTimeString();
                        
                        const content = document.createElement('div');
                        content.className = 'flex-1';
                        content.innerHTML = `
                            <div class="font-semibold text-sm">${item.event}</div>
                            <div class="text-sm text-gray-600">${item.details || ''}</div>
                        `;
                        
                        div.appendChild(time);
                        div.appendChild(content);
                        container.appendChild(div);
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
            fetch('/reload_index', { method: 'POST' })
                .then(r => r.json())
                .then(d => {
                    if (d.success) {
                        alert('Index rebuilt. Known faces: ' + d.count);
                        loadKnownFaces();
                        updateStats();
                    } else {
                        alert('Failed: ' + (d.error || 'unknown'));
                    }
                })
                .catch(() => alert('Request failed'));
        }

        function loadKnownFaces() {
            fetch('/known_faces')
                .then(r => r.json())
                .then(data => {
                    const gallery = document.getElementById('known-faces-gallery');
                    gallery.innerHTML = '';
                    
                    if (!data.length) {
                        gallery.innerHTML = '<p class="text-gray-500 col-span-full text-center py-4">No known faces</p>';
                        return;
                    }
                    
                    data.forEach(name => {
                        const div = document.createElement('div');
                        div.className = 'relative group';
                        div.innerHTML = `
                            <img src="/known/${name}.jpg" alt="${name}" class="w-full h-24 object-cover rounded-lg shadow-md group-hover:shadow-lg transition-shadow duration-300">
                            <div class="absolute bottom-0 left-0 right-0 bg-black bg-opacity-60 text-white text-xs p-1 rounded-b-lg text-center truncate">
                                ${name}
                            </div>
                        `;
                        gallery.appendChild(div);
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
                        tbody.innerHTML = '<tr><td colspan="4" class="text-center py-4 text-gray-500">No authorized plates</td></tr>';
                        return;
                    }
                    
                    rows.forEach(r => {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${r.id}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${r.plate}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${r.label || ''}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                                <button onclick="delPlate('${r.plate}')" class="text-red-600 hover:text-red-900">
                                    <i class="fas fa-trash"></i> Delete
                                </button>
                            </td>
                        `;
                        tbody.appendChild(tr);
                    });
                })
                .catch(() => {});
        }

        function addPlateUI() {
            const plate = document.getElementById('plate-input').value.trim();
            const label = document.getElementById('plate-label').value.trim();
            if (!plate) {
                alert('Enter a plate number');
                return;
            }
            fetch('/plates', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ plate: plate, label: label })
            })
                .then(r => r.json())
                .then(d => {
                    if (d.success) {
                        document.getElementById('plate-input').value = '';
                        document.getElementById('plate-label').value = '';
                        loadPlates();
                        updateStats();
                    } else {
                        alert('Failed: ' + (d.error || 'unknown'));
                    }
                })
                .catch(() => alert('Request failed'));
        }

        function delPlate(plate) {
            if (!confirm('Delete plate ' + plate + '?')) return;
            fetch('/plates/' + encodeURIComponent(plate), { method: 'DELETE' })
                .then(r => r.json())
                .then(d => {
                    if (d.success) {
                        loadPlates();
                        updateStats();
                    } else {
                        alert('Failed: ' + (d.error || 'unknown'));
                    }
                });
        }

        function openAddFace() {
            document.getElementById('add-face-modal').style.display = 'flex';
        }

        function closeAddFace() {
            document.getElementById('add-face-modal').style.display = 'none';
        }

        // Form submission
        document.getElementById('add-face-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const fd = new FormData();
            fd.append('name', document.getElementById('face-name').value);
            fd.append('image', document.getElementById('face-image').files[0]);
            fetch('/add_face', { method: 'POST', body: fd })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        alert('Face added successfully');
                        closeAddFace();
                        loadKnownFaces();
                        updateStats();
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(() => alert('Error adding face'));
        });

        // Auto-refresh intervals
        setInterval(refreshFaceFeed, 1000);
        setInterval(refreshPlateFeed, 1000);
        setInterval(updateFaceDetections, 4000);
        setInterval(updatePlateDetections, 4000);
        setInterval(updateStats, 10000);
        setInterval(updateBarrierStatus, 1000);
        setInterval(updateAuthorizedLog, 5000);

        // Initial load
        document.addEventListener('DOMContentLoaded', function() {
            refreshFaceFeed();
            refreshPlateFeed();
            updateFaceDetections();
            updatePlateDetections();
            updateAttendance();
            updateActivity();
            updateStats();
            updateBarrierStatus();
            updateAuthorizedLog();
            loadKnownFaces();
            loadPlates();
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
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
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

@app.route("/fps_face")
def fps_face():
    return jsonify({"fps": face_worker.get_fps()})

@app.route("/fps_plate")
def fps_plate():
    return jsonify({"fps": plate_worker.get_fps()})

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
    return jsonify(a[-200:])

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
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
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

@app.route("/manual_open_barrier", methods=["POST"])
def manual_open_barrier():
    try:
        safe_open_barrier(
            reason="Manual trigger",
            manual=True
        )
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

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/system_health")
def system_health():
    health_data = {
        "face_worker": {
            "fps": face_worker.get_fps(),
            "queue_size": unknown_face_queue.qsize(),
            "last_frame_age": time.time() - face_worker.last_process_time
        },
        "plate_worker": {
            "fps": plate_worker.get_fps(),
            "queue_size": unknown_plate_queue.qsize(),
            "last_frame_age": time.time() - plate_worker.last_process_time
        },
        "camera_face": {
            "connected": cam_face.cap is not None and cam_face.cap.isOpened(),
            "reconnect_attempts": cam_face.reconnect_attempts
        },
        "camera_plate": {
            "connected": cam_plate.cap is not None and cam_plate.cap.isOpened(),
            "reconnect_attempts": cam_plate.reconnect_attempts
        },
        "database": {
            "pool_size": len(DB_POOL),
            "total_attendance": get_total_attendance_count()
        }
    }
    return jsonify(health_data)

@app.route("/barrier_status")
def barrier_status_route():
    # Create a copy to avoid threading issues
    with barrier_lock:
        status = barrier_status.copy()
    
    # Format opened_at if it exists
    if status["opened_at"]:
        status["opened_at"] = datetime.fromtimestamp(status["opened_at"]).strftime("%Y-%m-%d %H:%M:%S")
    
    return jsonify(status)

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
    
    # Test barrier at startup
    if BARRIER_AVAILABLE:
        logger.info("Testing barrier at startup...")
        safe_open_barrier(reason="System startup test")
    
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)