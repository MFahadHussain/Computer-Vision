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
MIN_PLATE_ASPECT_RATIO = float(os.environ.get("MIN_PLATE_RATIO", "1.5"))
MAX_PLATE_ASPECT_RATIO = float(os.environ.get("MAX_PLATE_RATIO", "5.0"))

# Camera settings
RTSP_FACE_URL = os.environ.get("RTSP_FACE_URL", "rtsp://admin:afaqkhan-1@192.168.18.139:554/Streaming/channels/101")
RTSP_PLATE_URL = os.environ.get("RTSP_PLATE_URL", "rtsp://admin:afaqkhan-1@192.168.18.139:554/Streaming/channels/102")
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
AUTHORIZED_PLATE_PATTERN_MINLEN = int(os.environ.get("PLATE_MINLEN", 4))

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

def get_db_connection():
    if not DB_POOL:
        conn = sqlite3.connect(ATTENDANCE_DB, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        DB_POOL.append(conn)
    return DB_POOL.pop()

def return_db_connection(conn):
    if len(DB_POOL) < DB_POOL_SIZE:
        DB_POOL.append(conn)
    else:
        conn.close()

# ---------------- Barrier Control ----------------
try:
    from barrier_control import open_barrier
except Exception:
    def open_barrier():
        logger.info("[SIM] Barrier relay triggered")

last_barrier_open_time = 0.0
BARRIER_COOLDOWN = int(os.environ.get("BARRIER_COOLDOWN", 5))

def safe_open_barrier(reason: str = ""):
    global last_barrier_open_time
    now = time.time()
    if now - last_barrier_open_time >= BARRIER_COOLDOWN:
        try:
            open_barrier()
            last_barrier_open_time = now
            logger.info(f"Barrier opened ✅ {('— ' + reason) if reason else ''}")
        except Exception as e:
            logger.error(f"Barrier open failed: {e}")

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
        conn.commit()
    finally:
        return_db_connection(conn)

def insert_attendance(name: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO attendance (name, timestamp) VALUES (?,?)", (name, ts))
        conn.commit()
        return ts
    finally:
        return_db_connection(conn)

def get_all_attendance(limit: int = 500):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name, timestamp FROM attendance ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [{"id": r[0], "name": r[1], "timestamp": r[2]} for r in rows]
    finally:
        return_db_connection(conn)

def get_today_attendance_count():
    today_str = date.today().strftime("%Y-%m-%d")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM attendance WHERE date(timestamp)=?", (today_str,))
        c = cur.fetchone()[0]
        return int(c)
    finally:
        return_db_connection(conn)

def get_total_attendance_count():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM attendance")
        c = cur.fetchone()[0]
        return int(c)
    finally:
        return_db_connection(conn)

def list_plates() -> List[Dict[str, str]]:
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, plate, label FROM plates ORDER BY plate ASC")
        rows = cur.fetchall()
        return [{"id": r[0], "plate": r[1], "label": r[2] or ""} for r in rows]
    finally:
        return_db_connection(conn)

def add_plate(plate: str, label: str = "") -> Tuple[bool, str]:
    plate = plate.strip().upper()
    if len(plate) < AUTHORIZED_PLATE_PATTERN_MINLEN:
        return False, "Plate too short"
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO plates (plate, label) VALUES (?,?)", (plate, label))
        conn.commit()
        return True, ""
    except sqlite3.IntegrityError:
        return False, "Plate already exists"
    except Exception as e:
        return False, str(e)
    finally:
        return_db_connection(conn)

def remove_plate(plate: str) -> Tuple[bool, str]:
    plate = plate.strip().upper()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM plates WHERE plate=?", (plate,))
        conn.commit()
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        return_db_connection(conn)

def is_authorized_plate(plate: str) -> bool:
    plate = plate.strip().upper()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM plates WHERE plate=?", (plate,))
        r = cur.fetchone()
        return r is not None
    finally:
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
                time.sleep(RECONNECT_DELAY)
                self._open()
                continue
                
            with self.lock:
                self.last_frame = frame
                self.last_ts = time.time()
                self.frame_buffer.append((frame.copy(), time.time()))
                
        logger.info(f"Capture thread stopped: {self.name}")
        
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
            ts_str = insert_attendance(name)
            self.last_seen_ts[name] = now
            self._push_activity("attendance", f"{name} marked present at {ts_str}")
            safe_open_barrier(reason=f"face: {name}")
            
    def _detect_and_recognize(self, frame):
        public_dets = []
        vis = frame.copy()
        persons = []
        try:
            persons = face_app.get(frame)
        except Exception as e:
            logger.warning(f"InsightFace get() failed: {e}")
            
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
            else:
                self._maybe_log_attendance(label)
                
            public_dets.append(feed_item)
            
        return vis, public_dets
        
    def run(self):
        logger.info("FaceWorker started")
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
            vis, public_dets = self._detect_and_recognize(frame)
            with self.lock:
                self.curr_vis = vis
                self.latest_public_detections = public_dets
                
            # Performance monitoring
            self.monitor.track_performance(start_time)
            self.monitor.periodic_cleanup()
            
        logger.info("FaceWorker stopped")

class PlateWorker(WorkerBase):
    def __init__(self, cam: CameraThread):
        super().__init__(cam, "PlateWorker")
        self.last_plate_ts: Dict[str, float] = {}
        self._frame_skip = 0
        
    def _push_activity(self, event: str, details: str = ""):
        self.activity_log.append({"timestamp": int(time.time() * 1000), "event": event, "details": details})
        if len(self.activity_log) > 200:
            self.activity_log = self.activity_log[-200:]
            
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
            
        # Aspect ratio check
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        aspect = width / height if height > 0 else 0
        
        if aspect < MIN_PLATE_ASPECT_RATIO or aspect > MAX_PLATE_ASPECT_RATIO:
            return False
            
        # Character composition check
        if not any(c.isdigit() for c in text) or not any(c.isalpha() for c in text):
            return False
            
        return True
        
    def _ocr_plate(self, crop) -> str:
        if _easyocr_reader is None:
            return ""
            
        # Multiple preprocessing techniques
        methods = [
            lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2),
            lambda img: cv2.medianBlur(img, 3)
        ]
        
        best_text = ""
        best_conf = 0
        
        for method in methods:
            try:
                processed = method(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                results = _easyocr_reader.readtext(processed, detail=1)
                
                for _, text, conf in results:
                    if conf > best_conf and conf > PLATE_CONFIDENCE_THRESHOLD:
                        best_conf = conf
                        best_text = "".join(ch for ch in text if ch.isalnum()).upper()
            except Exception:
                continue
                
        return best_text
        
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
            
            if not text or not self._validate_plate(text, [x1, y1, x2, y2]):
                self._draw_plate(vis, (x1, y1, x2, y2), text, False)
                continue
                
            seen_any = True
            ok = is_authorized_plate(text)
            self._draw_plate(vis, (x1, y1, x2, y2), text, ok)
            public.append({"plate": text, "ok": ok, "timestamp": int(time.time())})
            
            ts_last = self.last_plate_ts.get(text, 0.0)
            now = time.time()
            if ok and (now - ts_last >= PLATE_DEDUP_WINDOW_SEC):
                self.last_plate_ts[text] = now
                safe_open_barrier(reason=f"plate: {text}")
                if SAVE_UNKNOWN_PLATES:
                    fname = f"plate_{text}_{int(now*1000)}.jpg"
                    try:
                        unknown_plate_queue.put_nowait((fname, crop))
                    except queue.Full:
                        logger.warning("Unknown plate queue full, dropping frame")
                self._push_activity("plate_ok", f"Authorized plate: {text}")
            elif not ok:
                if SAVE_UNKNOWN_PLATES:
                    fname = f"unknown_plate_{int(time.time()*1000)}.jpg"
                    try:
                        unknown_plate_queue.put_nowait((fname, crop))
                    except queue.Full:
                        logger.warning("Unknown plate queue full, dropping frame")
                self._push_activity("plate_unknown", f"Unknown plate: {text}")
                
        if not seen_any and plate_model is None:
            self._push_activity("plate_fallback", "OCR-only fallback used; no plate box")
            
        return vis, public
        
    def run(self):
        logger.info("PlateWorker started")
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
            vis, public = self._detect_and_read(frame)
            with self.lock:
                self.curr_vis = vis
                self.latest_public_detections = public[-10:]
                
            # Performance monitoring
            self.monitor.track_performance(start_time)
            self.monitor.periodic_cleanup()
            
        logger.info("PlateWorker stopped")

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
  <title>AI Security Dashboard – Face + Plate</title>
  <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap\" rel=\"stylesheet\">
  <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css\"> 
  <style>
    :root { --primary:#4361ee; --secondary:#3f37c9; --success:#4cc9f0; --danger:#f72585; --warning:#f8961e; --dark:#03071e; --light:#f8f9fa; --gray:#6c757d; --shadow:0 4px 6px rgba(0,0,0,.1); --shadow-lg:0 10px 15px rgba(0,0,0,.1);} *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:'Inter',sans-serif;background:#f0f2f5;color:var(--dark);line-height:1.6}
    .container{max-width:1500px;margin:0 auto;padding:20px}
    header{background:linear-gradient(135deg,var(--primary),var(--secondary));color:#fff;padding:25px 0;border-radius:10px;box-shadow:var(--shadow-lg);margin-bottom:30px;text-align:center}
    header h1{font-size:2.2rem;font-weight:700;margin-bottom:10px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}
    .panel{background:#fff;border-radius:10px;padding:16px;box-shadow:var(--shadow)}
    .panel-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #eee}
    .panel-title{font-size:1.1rem;font-weight:600}
    .status-indicator{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}
    .status-live{background:#2ecc71;box-shadow:0 0 8px #2ecc71;animation:pulse 1.5s infinite}
    @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(46,204,113,.7)}70%{box-shadow:0 0 0 10px rgba(46,204,113,0)}100%{box-shadow:0 0 0 0 rgba(46,204,113,0)}}
    .video-container{position:relative;width:100%;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:8px;background:#000}
    .video-container img{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover}
    .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:16px}
    .stat{background:#fff;border-radius:10px;padding:14px;display:flex;align-items:center;box-shadow:var(--shadow)}
    .stat .icon{width:50px;height:50px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-right:12px;font-size:20px}
    .icon.primary{background:rgba(67,97,238,.1);color:var(--primary)}
    .icon.success{background:rgba(76,201,240,.1);color:var(--success)}
    .icon.danger{background:rgba(247,37,133,.1);color:var(--danger)}
    .icon.warning{background:rgba(248,150,30,.1);color:var(--warning)}
    .table-container{overflow-x:auto}
    table{width:100%;border-collapse:collapse}
    th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #eee}
    th{background:#f8f9fa;font-weight:600}
    .face-gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));grid-gap:12px;margin-top:10px}
    .face-item{position:relative;border-radius:8px;overflow:hidden;box-shadow:var(--shadow)}
    .face-item img{width:100%;height:120px;object-fit:cover}
    .face-item .label{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.65);color:#fff;padding:4px;font-size:.8em;text-align:center}
    .pill{padding:2px 8px;border-radius:12px;font-size:.75em;font-weight:600;text-transform:uppercase}
    .ok{background:#d4edda;color:#155724}
    .bad{background:#f8d7da;color:#721c24}
    .flex{display:flex;gap:10px;align-items:center}
    .btn{display:inline-block;background:var(--primary);color:#fff;border:none;padding:7px 12px;border-radius:6px;cursor:pointer;font-size:.9rem}
    .btn:hover{opacity:.95}
    .btn-sm{padding:5px 9px;font-size:.8rem}
    .section{margin-top:18px}
    .input{padding:8px;border:1px solid #ddd;border-radius:6px}
  </style>
</head>
<body>
  <div class=\"container\">
    <header>
      <h1><i class=\"fas fa-shield-alt\"></i> AI Security Dashboard — Face & Plate</h1>
      <p>Barrier opens on recognized face <b>or</b> authorized plate</p>
    </header>
    <div class=\"stats\">
      <div class=\"stat\"><div class=\"icon primary\"><i class=\"fas fa-users\"></i></div><div><div id=\"known-count\" style=\"font-weight:700;font-size:1.2rem\">0</div><div>Known Faces</div></div></div>
      <div class=\"stat\"><div class=\"icon danger\"><i class=\"fas fa-user-slash\"></i></div><div><div id=\"unknown-count\" style=\"font-weight:700;font-size:1.2rem\">0</div><div>Unknown Faces (saved)</div></div></div>
      <div class=\"stat\"><div class=\"icon success\"><i class=\"fas fa-car\"></i></div><div><div id=\"plate-count\" style=\"font-weight:700;font-size:1.2rem\">0</div><div>Authorized Plates</div></div></div>
      <div class=\"stat\"><div class=\"icon warning\"><i class=\"fas fa-calendar-check\"></i></div><div><div id=\"today-count\" style=\"font-weight:700;font-size:1.2rem\">0</div><div>Today's Attendance</div></div></div>
    </div>
    <div class=\"grid\">
      <div class=\"panel\">
        <div class=\"panel-header\"><div class=\"panel-title\"><span class=\"status-indicator status-live\"></span>Face Camera</div><span id=\"fps-face\">FPS: 0</span></div>
        <div class=\"video-container\"><img id=\"live-face\" src=\"/video_feed_face\" alt=\"Face Feed\"></div>
        <div class=\"section\">
          <h4>Recent Face Detections</h4>
          <div id=\"face-dets\"></div>
        </div>
      </div>
      <div class=\"panel\">
        <div class=\"panel-header\"><div class=\"panel-title\"><span class=\"status-indicator status-live\"></span>Plate Camera</div><span id=\"fps-plate\">FPS: 0</span></div>
        <div class=\"video-container\"><img id=\"live-plate\" src=\"/video_feed_plate\" alt=\"Plate Feed\"></div>
        <div class=\"section\">
          <h4>Recent Plate Reads</h4>
          <div id=\"plate-dets\"></div>
        </div>
      </div>
      <div class=\"panel\">
        <div class=\"panel-header\"><div class=\"panel-title\">Attendance</div><button class=\"btn btn-sm\" onclick=\"updateAttendance()\"><i class=\"fas fa-sync-alt\"></i> Refresh</button></div>
        <div class=\"table-container\"><table><thead><tr><th>ID</th><th>Name</th><th>Timestamp</th></tr></thead><tbody id=\"attendance-table\"></tbody></table></div>
      </div>
      <div class=\"panel\">
        <div class=\"panel-header\"><div class=\"panel-title\">Activity Log</div><button class=\"btn btn-sm\" onclick=\"updateActivity()\"><i class=\"fas fa-sync-alt\"></i> Refresh</button></div>
        <div id=\"activity-feed\" style=\"max-height:320px;overflow:auto\"></div>
      </div>
      <div class=\"panel\" style=\"grid-column:1 / -1\">
        <div class=\"panel-header\"><div class=\"panel-title\">Known Faces</div><div class=\"flex\"><button class=\"btn btn-sm\" onclick=\"openAddFace()\"><i class=\"fas fa-plus\"></i> Add Face</button><button class=\"btn btn-sm\" style=\"margin-left:8px\" onclick=\"reloadIndex()\"><i class=\"fas fa-rotate\"></i> Rebuild Index</button></div></div>
        <div class=\"face-gallery\" id=\"known-faces-gallery\"></div>
      </div>
      <div class=\"panel\" style=\"grid-column:1 / -1\">
        <div class=\"panel-header\"><div class=\"panel-title\">Authorized Plates</div>
          <div class=\"flex\">
            <input id=\"plate-input\" class=\"input\" placeholder=\"Enter plate e.g. ABC123\"> 
            <input id=\"plate-label\" class=\"input\" placeholder=\"Optional label\"> 
            <button class=\"btn btn-sm\" onclick=\"addPlateUI()\"><i class=\"fas fa-plus\"></i> Add</button>
            <button class=\"btn btn-sm\" onclick=\"loadPlates()\"><i class=\"fas fa-sync-alt\"></i> Refresh</button>
          </div>
        </div>
        <div class=\"table-container\"><table><thead><tr><th>ID</th><th>Plate</th><th>Label</th><th>Action</th></tr></thead><tbody id=\"plates-table\"></tbody></table></div>
      </div>
    </div>
  </div>
  <div id=\"add-face-modal\" style=\"display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:1000;align-items:center;justify-content:center\"> 
    <div style=\"background:#fff;border-radius:10px;padding:16px;max-width:480px;width:92%\">
      <div style=\"display:flex;justify-content:space-between;align-items:center;margin-bottom:10px\"><h3>Add New Face</h3><button onclick=\"closeAddFace()\" class=\"btn btn-sm\">Close</button></div>
      <form id=\"add-face-form\" enctype=\"multipart/form-data\">
        <div style=\"margin-bottom:10px\"><label>Name</label><input type=\"text\" id=\"face-name\" class=\"input\" required></div>
        <div style=\"margin-bottom:10px\"><label>Face Image</label><input type=\"file\" id=\"face-image\" class=\"input\" accept=\"image/*\" required></div>
        <button type=\"submit\" class=\"btn\">Add Face</button>
      </form>
    </div>
  </div>
  <script>
    function openAddFace(){document.getElementById('add-face-modal').style.display='flex'}
    function closeAddFace(){document.getElementById('add-face-modal').style.display='none'}
    function refreshFaceFeed(){const img=document.getElementById('live-face');if(img){img.src='/video_feed_face?'+Date.now()}fetch('/fps_face').then(r=>r.json()).then(d=>{document.getElementById('fps-face').textContent='FPS: '+d.fps.toFixed(1)}).catch(()=>{})}
    function refreshPlateFeed(){const img=document.getElementById('live-plate');if(img){img.src='/video_feed_plate?'+Date.now()}fetch('/fps_plate').then(r=>r.json()).then(d=>{document.getElementById('fps-plate').textContent='FPS: '+d.fps.toFixed(1)}).catch(()=>{})}
    function updateFaceDetections(){fetch('/detections_face').then(r=>r.json()).then(data=>{const c=document.getElementById('face-dets');c.innerHTML='';if(!data.length){c.innerHTML='<p>No recent detections</p>';return}data.slice(-8).reverse().forEach(item=>{const div=document.createElement('div');div.style.display='flex';div.style.alignItems='center';div.style.gap='8px';div.style.marginBottom='6px';
      const span=document.createElement('span');span.className='pill '+(item.type==='known'?'ok':'bad');span.textContent=item.type;
      const name=document.createElement('span');name.textContent=item.type==='known'?item.name:'Unknown';
      const time=document.createElement('span');time.style.color='#6c757d';time.style.fontSize='.85em';time.textContent=new Date(item.timestamp*1000).toLocaleTimeString();
      div.appendChild(span);div.appendChild(name);div.appendChild(time);c.appendChild(div);});}).catch(()=>{})}
    function updatePlateDetections(){fetch('/detections_plate').then(r=>r.json()).then(data=>{const c=document.getElementById('plate-dets');c.innerHTML='';if(!data.length){c.innerHTML='<p>No recent reads</p>';return}data.slice(-8).reverse().forEach(item=>{const div=document.createElement('div');div.style.display='flex';div.style.alignItems='center';div.style.gap='8px';div.style.marginBottom='6px';
      const pill=document.createElement('span');pill.className='pill '+(item.ok?'ok':'bad');pill.textContent=item.ok?'OK':'Unknown';
      const text=document.createElement('span');text.textContent=item.plate||'(blank)';
      const time=document.createElement('span');time.style.color='#6c757d';time.style.fontSize='.85em';time.textContent=new Date(item.timestamp*1000).toLocaleTimeString();
      div.appendChild(pill);div.appendChild(text);div.appendChild(time);c.appendChild(div);});}).catch(()=>{})}
    function updateAttendance(){fetch('/attendance').then(r=>r.json()).then(rows=>{const tbody=document.getElementById('attendance-table');tbody.innerHTML='';if(!rows.length){tbody.innerHTML='<tr><td colspan="3">No attendance records</td></tr>';return}rows.forEach(row=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${row.id}</td><td>${row.name}</td><td>${row.timestamp}</td>`;tbody.appendChild(tr);})}).catch(()=>{})}
    function updateActivity(){fetch('/activity_all').then(r=>r.json()).then(data=>{const c=document.getElementById('activity-feed');c.innerHTML='';if(!data.length){c.innerHTML='<p>No activity records</p>';return}data.slice().reverse().forEach(item=>{const div=document.createElement('div');div.style.display='flex';div.style.gap='10px';div.style.borderBottom='1px solid #eee';div.style.padding='6px 0';
      const time=document.createElement('div');time.style.color='#6c757d';time.style.fontSize='.85em';time.textContent=new Date(item.timestamp).toLocaleTimeString();
      const content=document.createElement('div');content.innerHTML=`<b>${item.event}</b> — ${item.details||''}`;
      div.appendChild(time);div.appendChild(content);c.appendChild(div);});}).catch(()=>{})}
    function updateStats(){fetch('/stats').then(r=>r.json()).then(d=>{document.getElementById('known-count').textContent=d.known_faces;document.getElementById('unknown-count').textContent=d.unknown_faces;document.getElementById('plate-count').textContent=d.authorized_plates;document.getElementById('today-count').textContent=d.today_attendance}).catch(()=>{})}
    function reloadIndex(){fetch('/reload_index',{method:'POST'}).then(r=>r.json()).then(d=>{if(d.success){alert('Index rebuilt. Known faces: '+d.count);loadKnownFaces();updateStats();}else{alert('Failed: '+(d.error||'unknown'))}}).catch(()=>alert('Request failed'))}
    function loadKnownFaces(){fetch('/known_faces').then(r=>r.json()).then(data=>{const g=document.getElementById('known-faces-gallery');g.innerHTML='';if(!data.length){g.innerHTML='<p>No known faces</p>';return}data.forEach(name=>{const div=document.createElement('div');div.className='face-item';div.innerHTML=`<img src="/known/${name}.jpg" alt="${name}"><div class="label">${name}</div>`;g.appendChild(div);})}).catch(()=>{})}
    function loadPlates(){fetch('/plates').then(r=>r.json()).then(rows=>{const tbody=document.getElementById('plates-table');tbody.innerHTML='';if(!rows.length){tbody.innerHTML='<tr><td colspan="4">No authorized plates</td></tr>';return}rows.forEach(r=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${r.id}</td><td>${r.plate}</td><td>${r.label||''}</td><td><button class="btn btn-sm" onclick="delPlate('${r.plate}')">Delete</button></td>`;tbody.appendChild(tr);})}).catch(()=>{})}
    function addPlateUI(){const p=document.getElementById('plate-input').value.trim();const l=document.getElementById('plate-label').value.trim();if(!p){alert('Enter a plate');return}fetch('/plates',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({plate:p,label:l})}).then(r=>r.json()).then(d=>{if(d.success){document.getElementById('plate-input').value='';document.getElementById('plate-label').value='';loadPlates();updateStats();}else{alert('Failed: '+(d.error||'unknown'))}}).catch(()=>alert('Request failed'))}
    function delPlate(p){if(!confirm('Delete plate '+p+'?'))return;fetch('/plates/'+encodeURIComponent(p),{method:'DELETE'}).then(r=>r.json()).then(d=>{if(d.success){loadPlates();updateStats();}else{alert('Failed: '+(d.error||'unknown'))}})}
    document.getElementById('add-face-form').addEventListener('submit',function(e){e.preventDefault();const fd=new FormData();fd.append('name',document.getElementById('face-name').value);fd.append('image',document.getElementById('face-image').files[0]);fetch('/add_face',{method:'POST',body:fd}).then(r=>r.json()).then(data=>{if(data.success){alert('Face added');closeAddFace();loadKnownFaces();updateStats();}else{alert('Error: '+data.error)}}).catch(()=>alert('Error adding face'))});
    setInterval(refreshFaceFeed,1000);setInterval(refreshPlateFeed,1000);
    setInterval(updateFaceDetections,4000);setInterval(updatePlateDetections,4000);
    setInterval(updateStats,10000);
    document.addEventListener('DOMContentLoaded',function(){refreshFaceFeed();refreshPlateFeed();updateFaceDetections();updatePlateDetections();updateAttendance();updateActivity();loadKnownFaces();loadPlates();updateStats()});
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
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)