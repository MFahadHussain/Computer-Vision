#!/usr/bin/env python3
"""
AI Security Dashboard – Optimized (face-first, LPR-ready hooks)
- Faster, safer barrier logic (cooldown + dedup)
- Frame skipping + async unknown saver
- InsightFace primary, YOLO face fallback (optional)
- Clean index management + health endpoints
- Jetson-friendly settings (FFmpeg/GStreamer RTSP, CPU thread hints)

Drop-in replacement for your original script. Keep your existing folder structure:
  known/  unknown/  snapshots/  logs/  attendance.db

Author: refactor for speed + maintainability
"""
import os
import cv2
import time
import json
import queue
import signal
import logging
import sqlite3
import threading
from datetime import datetime, date
from typing import List, Dict, Tuple

from flask import Flask, Response, jsonify, send_from_directory, request

# ---- Optional: your barrier control module ----
try:
    from barrier_control import open_barrier as _hw_open_barrier
except Exception:
    _hw_open_barrier = None

# ---------------- Logging ----------------
USE_GPU = os.environ.get("USE_GPU", "1") not in ("0", "false", "False")
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("ai-sec-dashboard")

# ---------------- Config ----------------
RTSP_URL = os.environ.get(
    "RTSP_URL",
    "rtsp://admin:password@192.168.1.64:554/Streaming/channels/101",
)
# If you prefer GStreamer on Jetson, set RTSP_BACKEND=gst and supply a pipeline in RTSP_URL
RTSP_BACKEND = os.environ.get("RTSP_BACKEND", "ffmpeg").lower()  # "ffmpeg"|"gst"

CAM_READ_TIMEOUT = 5.0  # seconds
RECONNECT_DELAY = 0.5
FRAME_MAX_AGE = 2.0  # seconds
SAVE_UNKNOWN = True
SAVE_UNKNOWN_ASYNC = True
ATTENDANCE_DB = os.path.join(os.getcwd(), "attendance.db")
ATTENDANCE_DEDUP_WINDOW_SEC = int(os.environ.get("DEDUP_SEC", "60"))

BASE_DIR = os.getcwd()
KNOWN_DIR = os.path.join(BASE_DIR, "known")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown")
SNAP_DIR = os.path.join(BASE_DIR, "snapshots")
for d in (KNOWN_DIR, UNKNOWN_DIR, SNAP_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- Performance Tweaks ----------------
FRAME_PROCESS_EVERY = int(os.environ.get("FRAME_EVERY", "3"))  # process every Nth frame
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "50"))
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.45"))
BARRIER_COOLDOWN = int(os.environ.get("BARRIER_COOLDOWN", "5"))

# Optional: reduce OpenCV CPU thread contention (Jetson often benefits from small values)
try:
    cv2.setNumThreads(int(os.environ.get("CV_THREADS", "2")))
except Exception:
    pass

# ---------------- Barrier Cooldown ----------------
_last_barrier_open_time = 0.0


def safe_open_barrier():
    global _last_barrier_open_time
    now = time.time()
    if now - _last_barrier_open_time < BARRIER_COOLDOWN:
        logger.debug("Barrier cooldown active; skip trigger")
        return False
    try:
        if _hw_open_barrier:
            _hw_open_barrier()
            _last_barrier_open_time = now
            logger.info("Barrier triggered ✅")
            return True
        else:
            logger.warning("barrier_control.open_barrier() not available; dry-run")
            _last_barrier_open_time = now
            return True
    except Exception as e:
        logger.error(f"Barrier open failed: {e}")
        return False


# ---------------- Async Unknown Saver ----------------
unknown_save_queue: "queue.Queue[Tuple[str, any]]" = queue.Queue()


def _unknown_saver_worker():
    while True:
        item = unknown_save_queue.get()
        if item is None:
            break
        fname, crop = item
        try:
            fpath = os.path.join(UNKNOWN_DIR, fname)
            ok, enc = cv2.imencode(".jpg", crop)
            if ok:
                enc.tofile(fpath)
                logger.debug(f"Saved unknown face {fname}")
        except Exception as e:
            logger.error(f"Unknown save failed: {e}")


if SAVE_UNKNOWN_ASYNC:
    threading.Thread(target=_unknown_saver_worker, daemon=True).start()

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
        logger.warning(f"Could not query ONNX providers, forcing CPU. Reason: {e}")
        return ["CPUExecutionProvider"]


# ---------------- Models ----------------
try:
    from ultralytics import YOLO  # optional face fallback
except Exception as e:
    YOLO = None
    logger.warning(f"Ultralytics import failed or not installed: {e}")

insightface = None
face_app = None


def init_insightface():
    """Init InsightFace FaceAnalysis (detector+recognizer)."""
    global insightface, face_app
    import importlib

    insightface = importlib.import_module("insightface")
    from insightface.app import FaceAnalysis

    providers = get_onnx_providers()
    logger.info("Preparing InsightFace model (buffalo_l)...")
    face_app = FaceAnalysis(name="buffalo_l", providers=providers)
    ctx_id = 0 if ("CUDAExecutionProvider" in providers) else -1
    face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    try:
        face_app.models["detection"].threshold = 0.5  # slightly stricter
    except Exception:
        pass
    logger.info("InsightFace ready.")


# ---------------- Known face index ----------------
import numpy as np


class FaceIndex:
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.labels: List[str] = []
        self._lock = threading.Lock()

    def clear(self):
        with self._lock:
            self.embeddings = []
            self.labels = []

    def add(self, label: str, emb: np.ndarray):
        if emb is None:
            return
        with self._lock:
            self.embeddings.append(emb.astype(np.float32))
            self.labels.append(label)

    def build_from_dir(self, folder: str):
        self.clear()
        logger.info(f"Building index from: {folder}")
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
        with self._lock:
            count = len(self.labels)
        logger.info(f"Indexed {count} known faces.")

    def search(self, q_emb: np.ndarray) -> Tuple[str, float]:
        with self._lock:
            if not self.embeddings:
                return ("Unknown", 0.0)
            E = np.stack(self.embeddings, axis=0)  # (N, D)
            labels = list(self.labels)
        q = q_emb.astype(np.float32)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        qn = q / (np.linalg.norm(q) + 1e-8)
        sims = En @ qn
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        label = labels[idx] if best >= SIM_THRESHOLD else "Unknown"
        return (label, best)


face_index = FaceIndex()

# ---------------- Attendance (SQLite) ----------------

def init_db():
    con = sqlite3.connect(ATTENDANCE_DB)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    con.commit()
    con.close()


def insert_attendance(name: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    con = sqlite3.connect(ATTENDANCE_DB)
    cur = con.cursor()
    cur.execute("INSERT INTO attendance (name, timestamp) VALUES (?,?)", (name, ts))
    con.commit()
    con.close()
    # Barrier trigger moved behind cooldown guard
    safe_open_barrier()
    return ts


def get_all_attendance(limit: int = 500):
    con = sqlite3.connect(ATTENDANCE_DB)
    cur = con.cursor()
    cur.execute(
        "SELECT id, name, timestamp FROM attendance ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    con.close()
    return [{"id": r[0], "name": r[1], "timestamp": r[2]} for r in rows]


def get_today_attendance_count():
    today_str = date.today().strftime("%Y-%m-%d")
    con = sqlite3.connect(ATTENDANCE_DB)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM attendance WHERE date(timestamp)=?", (today_str,))
    c = cur.fetchone()[0]
    con.close()
    return int(c)


def get_total_attendance_count():
    con = sqlite3.connect(ATTENDANCE_DB)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM attendance")
    c = cur.fetchone()[0]
    con.close()
    return int(c)


# ---------------- Camera Thread ----------------
class CameraThread(threading.Thread):
    def __init__(self, url: str, backend: str = "ffmpeg"):
        super().__init__(daemon=True)
        self.url = url
        self.backend = backend
        self.cap = None
        self.lock = threading.Lock()
        self.last_frame = None
        self.last_ts = 0.0
        self.stop_event = threading.Event()

    def _open(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.backend == "gst":
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 25)
        except Exception:
            pass

    def run(self):
        logger.info("Starting capture thread...")
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                self._open()
                time.sleep(0.1)
                if self.cap is None or not self.cap.isOpened():
                    logger.warning("Capture not opened, retrying...")
                    time.sleep(RECONNECT_DELAY)
                    continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                logger.warning("Frame empty, reconnecting...")
                time.sleep(RECONNECT_DELAY)
                self._open()
                continue

            with self.lock:
                self.last_frame = frame
                self.last_ts = time.time()

        logger.info("Capture thread stopped.")

    def get_latest(self):
        with self.lock:
            return (
                None if self.last_frame is None else self.last_frame.copy(),
                self.last_ts,
            )

    def stop(self):
        self.stop_event.set()
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


cam = CameraThread(RTSP_URL, RTSP_BACKEND)


# ---------------- Detection/Recognition Worker ----------------
class Worker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.curr_vis = None
        self.lock = threading.Lock()
        self.fps = 0.0
        self._last_fps_time = time.time()
        self._frames = 0
        self._proc_counter = 0
        self.activity_log: List[Dict] = []  # latest 200 events
        self.last_seen_ts: Dict[str, float] = {}  # dedup attendance per name
        self.latest_public_detections: List[Dict] = []  # for /detections card feed

    def _draw_bbox(self, img, box, label, sim):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        caption = f"{label} ({sim:.2f})" if label != "Unknown" else "Unknown"
        cv2.putText(
            img,
            caption,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    def _save_unknown_crop(self, frame, box):
        if not SAVE_UNKNOWN:
            return None
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ts = int(time.time() * 1000)
        fname = f"unknown_{ts}.jpg"
        if SAVE_UNKNOWN_ASYNC:
            unknown_save_queue.put((fname, crop))
            return fname
        fpath = os.path.join(UNKNOWN_DIR, fname)
        ok, enc = cv2.imencode(".jpg", crop)
        if ok:
            enc.tofile(fpath)
            return fname
        return None

    def _push_activity(self, event: str, details: str = ""):
        self.activity_log.append(
            {"timestamp": int(time.time() * 1000), "event": event, "details": details}
        )
        if len(self.activity_log) > 200:
            self.activity_log = self.activity_log[-200:]

    def _maybe_log_attendance(self, name: str):
        now = time.time()
        last = self.last_seen_ts.get(name, 0.0)
        if now - last >= ATTENDANCE_DEDUP_WINDOW_SEC:
            ts_str = insert_attendance(name)
            self.last_seen_ts[name] = now
            self._push_activity(
                event="attendance", details=f"{name} marked present at {ts_str}"
            )

    def _detect_with_fallback(self, frame):
        persons = []
        try:
            persons = face_app.get(frame)
        except Exception as e:
            logger.warning(f"InsightFace get() failed: {e}")
        # YOLO face fallback (optional)
        if not persons and yolo_model is not None:
            try:
                res = yolo_model.predict(frame, verbose=False, conf=0.35, iou=0.45)[0]
                boxes = (
                    res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
                )
                for b in boxes:
                    x1, y1, x2, y2 = [int(v) for v in b]
                    if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                        continue
                    face_crop = frame[y1:y2, x1:x2]
                    try:
                        sub = face_app.get(face_crop)
                        if sub:
                            obj = sub[0]
                            obj.bbox = np.array([x1, y1, x2, y2])
                            persons.append(obj)
                    except Exception as e:
                        logger.debug(f"Embedding on crop failed: {e}")
            except Exception as e:
                logger.debug(f"YOLO fallback failed: {e}")
        return persons

    def _classify_and_annotate(self, frame, persons):
        public_dets = []
        vis = frame.copy()
        for p in persons:
            x1, y1, x2, y2 = p.bbox.astype(int).tolist()
            if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                continue
            label, sim = "Unknown", 0.0
            try:
                emb = p.embedding
                if emb is not None:
                    label, sim = face_index.search(emb)
            except Exception as e:
                logger.debug(f"search() failed: {e}")

            self._draw_bbox(vis, (x1, y1, x2, y2), label, sim)

            feed_item = {
                "type": "known" if label != "Unknown" else "unknown",
                "name": label if label != "Unknown" else None,
                "timestamp": int(time.time()),
            }
            if label == "Unknown":
                fname = self._save_unknown_crop(frame, (x1, y1, x2, y2))
                if fname:
                    feed_item["path"] = f"/unknown/{fname}"
                    self._push_activity("unknown", f"Unknown face captured: {fname}")
            else:
                self._maybe_log_attendance(label)

            public_dets.append(feed_item)
        return vis, public_dets

    def run(self):
        logger.info("Worker started.")
        while not self.stop_event.is_set():
            frame, ts = cam.get_latest()
            if frame is None:
                time.sleep(0.005)
                continue
            if time.time() - ts > FRAME_MAX_AGE:
                time.sleep(0.005)
                continue

            # Frame skipping (process every Nth)
            self._proc_counter = (self._proc_counter + 1) % FRAME_PROCESS_EVERY
            if self._proc_counter != 0:
                # still update preview with last annotated frame
                with self.lock:
                    if self.curr_vis is None:
                        self.curr_vis = frame
                # FPS accounting only counts processed frames; skip increment here
                continue

            persons = self._detect_with_fallback(frame)
            vis, public_dets = self._classify_and_annotate(frame, persons)

            with self.lock:
                self.curr_vis = vis
                self.latest_public_detections = public_dets

            # FPS calc (processed frames only)
            self._frames += 1
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self.fps = self._frames / (now - self._last_fps_time)
                self._frames = 0
                self._last_fps_time = now

        logger.info("Worker stopped.")

    def get_vis(self):
        with self.lock:
            if self.curr_vis is None:
                return None
            return self.curr_vis.copy()

    def get_public_detections(self):
        with self.lock:
            return list(self.latest_public_detections)

    def get_activity(self):
        with self.lock:
            return list(self.activity_log)

    def get_fps(self):
        return float(self.fps)


worker = Worker()

# ---------------- YOLO Load (optional) ----------------
yolo_model = None
if YOLO is not None:
    try:
        # Provide yolov8n-face.pt in working dir or adjust path
        yolo_model = YOLO("yolov8n-face.pt")
        logger.info("YOLOv8-face loaded (fallback detector).")
    except Exception as e:
        logger.warning(f"YOLOv8-face not available ({e}); using InsightFace only.")

# ---------------- Flask App ----------------
app = Flask(__name__)


@app.route("/")
def index():
    html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <title>AI Security Dashboard</title>
  <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap\" rel=\"stylesheet\">
  <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css\">
  <style>
    :root { --primary:#4361ee; --secondary:#3f37c9; --success:#4cc9f0; --danger:#f72585; --warning:#f8961e; --dark:#03071e; --light:#f8f9fa; --gray:#6c757d; --shadow:0 4px 6px rgba(0,0,0,.1); --shadow-lg:0 10px 15px rgba(0,0,0,.1);} *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:'Inter',sans-serif;background:#f0f2f5;color:var(--dark);line-height:1.6}
    .container{max-width:1400px;margin:0 auto;padding:20px}
    header{background:linear-gradient(135deg,var(--primary),var(--secondary));color:#fff;padding:25px 0;border-radius:10px;box-shadow:var(--shadow-lg);margin-bottom:30px;text-align:center}
    header h1{font-size:2.5rem;font-weight:700;margin-bottom:10px}
    header p{font-size:1.1rem;opacity:.9}
    .stats-container{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px;margin-bottom:30px}
    .stat-card{background:#fff;border-radius:10px;padding:20px;box-shadow:var(--shadow);display:flex;align-items:center;transition:transform .3s ease}
    .stat-card:hover{transform:translateY(-5px)}
    .stat-icon{width:60px;height:60px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-right:15px;font-size:24px}
    .stat-icon.primary{background:rgba(67,97,238,.1);color:var(--primary)}
    .stat-icon.success{background:rgba(76,201,240,.1);color:var(--success)}
    .stat-icon.danger{background:rgba(247,37,133,.1);color:var(--danger)}
    .stat-icon.warning{background:rgba(248,150,30,.1);color:var(--warning)}
    .stat-info h3{font-size:1.8rem;font-weight:700;margin-bottom:5px}
    .stat-info p{color:var(--gray);font-size:.9rem}
    .dashboard{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px}
    .panel{background:#fff;border-radius:10px;padding:20px;box-shadow:var(--shadow)}
    .panel-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid #eee}
    .panel-title{font-size:1.3rem;font-weight:600;color:var(--dark)}
    .status-indicator{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:8px}
    .status-live{background:#2ecc71;box-shadow:0 0 8px #2ecc71;animation:pulse 1.5s infinite}
    .status-offline{background:#e74c3c}
    @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(46,204,113,.7)}70%{box-shadow:0 0 0 10px rgba(46,204,113,0)}100%{box-shadow:0 0 0 0 rgba(46,204,113,0)}}
    .video-container{position:relative;width:100%;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:8px;background:#000}
    .video-container img{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover}
    .btn{display:inline-block;background:var(--primary);color:#fff;border:none;padding:8px 15px;border-radius:5px;cursor:pointer;font-size:.9rem;font-weight:500;transition:background .3s}
    .btn:hover{background:var(--secondary)}
    .btn-sm{padding:5px 10px;font-size:.8rem}
    .table-container{overflow-x:auto}
    table{width:100%;border-collapse:collapse}
    th,td{padding:12px 15px;text-align:left;border-bottom:1px solid #eee}
    th{background:#f8f9fa;font-weight:600;color:var(--dark)}
    tr:hover{background:#f8f9fa}
    .face-gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));grid-gap:15px;margin-top:15px}
    .face-item{position:relative;border-radius:8px;overflow:hidden;box-shadow:var(--shadow);transition:transform .3s ease}
    .face-item:hover{transform:scale(1.05)}
    .face-item img{width:100%;height:120px;object-fit:cover;display:block}
    .face-item .label{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.7);color:#fff;padding:5px;font-size:.8em;text-align:center}
    .detection-feed{max-height:300px;overflow-y:auto;margin-top:15px}
    .detection-item{display:flex;align-items:center;padding:10px;border-bottom:1px solid #eee}
    .detection-item img{width:50px;height:50px;border-radius:50%;object-fit:cover;margin-right:15px}
    .detection-info{flex-grow:1}
    .detection-name{font-weight:600;margin-bottom:3px}
    .detection-time{font-size:.8em;color:var(--gray)}
    .detection-type{padding:3px 8px;border-radius:12px;font-size:.7em;font-weight:600;text-transform:uppercase}
    .type-known{background:#d4edda;color:#155724}
    .type-unknown{background:#f8d7da;color:#721c24}
    .full-width{grid-column:1 / -1}
    .activity-item{display:flex;padding:10px 0;border-bottom:1px solid #eee}
    .activity-time{width:80px;color:#6c757d;font-size:.85rem}
    .activity-content{flex-grow:1}
    .activity-event{font-weight:500}
    .activity-details{font-size:.85rem;color:#6c757d}
    .footer{text-align:center;margin-top:30px;padding:20px;color:#6c757d;font-size:.9rem}
    .alert{padding:15px;margin-bottom:20px;border:1px solid transparent;border-radius:4px}
    .alert-danger{color:#721c24;background:#f8d7da;border-color:#f5c6cb}
    .alert-success{color:#155724;background:#d4edda;border-color:#c3e6cb}
  </style>
</head>
<body>
  <div class=\"container\">
    <header>
      <h1><i class=\"fas fa-user-check\"></i> AI Security Dashboard</h1>
      <p>Real-time monitoring and attendance tracking system</p>
    </header>
    <div class=\"stats-container\">
      <div class=\"stat-card\"><div class=\"stat-icon primary\"><i class=\"fas fa-users\"></i></div><div class=\"stat-info\"><h3 id=\"known-count\">0</h3><p>Known Faces</p></div></div>
      <div class=\"stat-card\"><div class=\"stat-icon danger\"><i class=\"fas fa-user-slash\"></i></div><div class=\"stat-info\"><h3 id=\"unknown-count\">0</h3><p>Unknown Faces</p></div></div>
      <div class=\"stat-card\"><div class=\"stat-icon success\"><i class=\"fas fa-user-check\"></i></div><div class=\"stat-info\"><h3 id=\"attendance-count\">0</h3><p>Attendance Records</p></div></div>
      <div class=\"stat-card\"><div class=\"stat-icon warning\"><i class=\"fas fa-calendar-day\"></i></div><div class=\"stat-info\"><h3 id=\"today-count\">0</h3><p>Today's Attendance</p></div></div>
    </div>
    <div class=\"dashboard\">
      <div class=\"panel\">
        <div class=\"panel-header\"><h2 class=\"panel-title\"><span class=\"status-indicator status-live\"></span>Live Video Feed</h2><span id=\"fps-counter\">FPS: 0</span></div>
        <div class=\"video-container\"><img id=\"live-feed\" src=\"/video_feed\" alt=\"Live Feed\"></div>
      </div>
      <div class=\"panel\">
        <div class=\"panel-header\"><h2 class=\"panel-title\">Recent Detections</h2><button class=\"btn btn-sm\" onclick=\"updateDetections()\"><i class=\"fas fa-sync-alt\"></i> Refresh</button></div>
        <div id=\"detection-feed\" class=\"detection-feed\"><p>Loading detections...</p></div>
      </div>
      <div class=\"panel\">
        <div class=\"panel-header\"><h2 class=\"panel-title\">Attendance Records</h2><button class=\"btn btn-sm\" onclick=\"updateAttendance()\"><i class=\"fas fa-sync-alt\"></i> Refresh</button></div>
        <div class=\"table-container\"><table><thead><tr><th>ID</th><th>Name</th><th>Timestamp</th></tr></thead><tbody id=\"attendance-table\"></tbody></table></div>
      </div>
      <div class=\"panel\">
        <div class=\"panel-header\"><h2 class=\"panel-title\">Activity Log</h2><button class=\"btn btn-sm\" onclick=\"updateActivity()\"><i class=\"fas fa-sync-alt\"></i> Refresh</button></div>
        <div id=\"activity-feed\" class=\"detection-feed\"></div>
      </div>
      <div class=\"panel full-width\">
        <div class=\"panel-header\"><h2 class=\"panel-title\">Known Faces</h2><button class=\"btn btn-sm\" onclick=\"document.getElementById('add-face-modal').style.display='flex'\"><i class=\"fas fa-plus\"></i> Add Face</button><button class=\"btn btn-sm\" style=\"margin-left:8px\" onclick=\"reloadIndex()\"><i class=\"fas fa-rotate\"></i> Rebuild Index</button></div>
        <div class=\"face-gallery\" id=\"known-faces-gallery\"></div>
      </div>
      <div class=\"panel full-width\">
        <div class=\"panel-header\"><h2 class=\"panel-title\">Unknown Faces</h2><button class=\"btn btn-sm\" onclick=\"updateUnknownFaces()\"><i class=\"fas fa-sync-alt\"></i> Refresh</button></div>
        <div class=\"face-gallery\" id=\"unknown-faces-gallery\"></div>
      </div>
    </div>
    <div class=\"footer\"><p>AI Security System &copy; 2025 | Powered by ZYPHER ENTERPRISE</p></div>
  </div>
  <div id=\"add-face-modal\" class=\"modal\" style=\"display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.5);z-index:1000;justify-content:center;align-items:center\">
    <div class=\"modal-content\" style=\"background:#fff;border-radius:10px;padding:20px;max-width:500px;width:90%\">
      <div class=\"modal-header\" style=\"display:flex;justify-content:space-between;align-items:center;margin-bottom:15px\"><h3 class=\"modal-title\">Add New Face</h3><button class=\"close-btn\" onclick=\"document.getElementById('add-face-modal').style.display='none'\" style=\"background:none;border:none;font-size:1.5rem;cursor:pointer;color:#6c757d\">&times;</button></div>
      <form id=\"add-face-form\" enctype=\"multipart/form-data\">
        <div class=\"form-group\"><label for=\"face-name\">Name</label><input type=\"text\" id=\"face-name\" class=\"form-control\" required></div>
        <div class=\"form-group\"><label for=\"face-image\">Face Image</label><input type=\"file\" id=\"face-image\" class=\"form-control\" accept=\"image/*\" required></div>
        <button type=\"submit\" class=\"btn\">Add Face</button>
      </form>
    </div>
  </div>
  <script>
    function updateLiveFeed(){const img=document.getElementById('live-feed');if(img){img.src='/video_feed?'+new Date().getTime()}fetch('/fps').then(r=>r.json()).then(d=>{document.getElementById('fps-counter').textContent=`FPS: ${d.fps.toFixed(1)}`}).catch(()=>{})}
    function updateDetections(){fetch('/detections').then(r=>r.json()).then(data=>{const c=document.getElementById('detection-feed');c.innerHTML='';if(!data.length){c.innerHTML='<p>No recent detections</p>';return}data.forEach(item=>{const div=document.createElement('div');div.className='detection-item';const img=document.createElement('img');if(item.type==='known'&&item.name){img.src='/known/'+item.name+'.jpg'}else if(item.path){img.src=item.path}else{img.src=''}const info=document.createElement('div');info.className='detection-info';const name=document.createElement('div');name.className='detection-name';name.textContent=item.type==='known'?item.name:'Unknown Face';const time=document.createElement('div');time.className='detection-time';time.textContent=new Date(item.timestamp*1000).toLocaleTimeString();const type=document.createElement('span');type.className='detection-type '+(item.type==='known'?'type-known':'type-unknown');type.textContent=item.type;info.appendChild(name);info.appendChild(time);info.appendChild(type);div.appendChild(img);div.appendChild(info);c.appendChild(div);})}).catch(()=>{})}
    function updateAttendance(){fetch('/attendance').then(r=>r.json()).then(rows=>{const tbody=document.getElementById('attendance-table');tbody.innerHTML='';if(!rows.length){tbody.innerHTML='<tr><td colspan="3">No attendance records</td></tr>';return}rows.forEach(row=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${row.id}</td><td>${row.name}</td><td>${row.timestamp}</td>`;tbody.appendChild(tr);})}).catch(()=>{})}
    function updateActivity(){fetch('/activity').then(r=>r.json()).then(data=>{const c=document.getElementById('activity-feed');c.innerHTML='';if(!data.length){c.innerHTML='<p>No activity records</p>';return}data.slice().reverse().forEach(item=>{const div=document.createElement('div');div.className='activity-item';const time=document.createElement('div');time.className='activity-time';time.textContent=new Date(item.timestamp).toLocaleTimeString();const content=document.createElement('div');content.className='activity-content';const event=document.createElement('div');event.className='activity-event';event.textContent=item.event;const details=document.createElement('div');details.className='activity-details';details.textContent=item.details||'';content.appendChild(event);content.appendChild(details);div.appendChild(time);div.appendChild(content);c.appendChild(div);})}).catch(()=>{})}
    function updateKnownFaces(){fetch('/known_faces').then(r=>r.json()).then(data=>{const g=document.getElementById('known-faces-gallery');g.innerHTML='';if(!data.length){g.innerHTML='<p>No known faces</p>';return}data.forEach(name=>{const div=document.createElement('div');div.className='face-item';div.innerHTML=`<img src="/known/${name}.jpg" alt="${name}"><div class="label">${name}</div>`;g.appendChild(div);})}).catch(()=>{})}
    function updateUnknownFaces(){fetch('/unknown_faces').then(r=>r.json()).then(data=>{const g=document.getElementById('unknown-faces-gallery');g.innerHTML='';if(!data.length){g.innerHTML='<p>No unknown faces</p>';return}data.forEach(fname=>{const div=document.createElement('div');div.className='face-item';div.innerHTML=`<img src="/unknown/${fname}" alt="Unknown Face"><div class="label">Unknown</div>`;g.appendChild(div);})}).catch(()=>{})}
    function updateStats(){fetch('/stats').then(r=>r.json()).then(d=>{document.getElementById('known-count').textContent=d.known_faces;document.getElementById('unknown-count').textContent=d.unknown_faces;document.getElementById('attendance-count').textContent=d.total_attendance;document.getElementById('today-count').textContent=d.today_attendance}).catch(()=>{})}
    function reloadIndex(){fetch('/reload_index',{method:'POST'}).then(r=>r.json()).then(d=>{if(d.success){alert('Index rebuilt. Known faces: '+d.count);updateKnownFaces();updateStats();}else{alert('Failed: '+(d.error||'unknown'))}}).catch(()=>alert('Request failed'))}
    document.getElementById('add-face-form').addEventListener('submit',function(e){e.preventDefault();const fd=new FormData();fd.append('name',document.getElementById('face-name').value);fd.append('image',document.getElementById('face-image').files[0]);fetch('/add_face',{method:'POST',body:fd}).then(r=>r.json()).then(data=>{if(data.success){alert('Face added successfully!');document.getElementById('add-face-modal').style.display='none';document.getElementById('add-face-form').reset();updateKnownFaces();updateStats();}else{alert('Error adding face: '+data.error)}}).catch(err=>alert('Error adding face'))});
    setInterval(updateLiveFeed,1000);setInterval(updateDetections,4000);setInterval(updateStats,10000);
    document.addEventListener('DOMContentLoaded',function(){updateLiveFeed();updateDetections();updateAttendance();updateActivity();updateKnownFaces();updateUnknownFaces();updateStats()});
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


# ---------------- MJPEG Stream ----------------
def mjpeg_generator():
    boundary = b"--frame"
    while True:
        vis = worker.get_vis()
        if vis is None:
            time.sleep(0.01)
            continue
        ok, buf = cv2.imencode(".jpg", vis)
        if not ok:
            continue
        frame = buf.tobytes()
        yield (
            boundary
            + b"\r\n"
            + b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(frame)}\r\n\r\n".encode()
            + frame
            + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/fps")
def get_fps():
    return jsonify({"fps": worker.get_fps()})


@app.route("/detections")
def get_detections():
    return jsonify(worker.get_public_detections())


@app.route("/activity")
def get_activity():
    return jsonify(worker.get_activity())


@app.route("/attendance")
def get_attendance():
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


@app.route("/unknown_faces")
def unknown_faces():
    fns = [f for f in os.listdir(UNKNOWN_DIR) if os.path.isfile(os.path.join(UNKNOWN_DIR, f))]
    fns.sort(reverse=True)
    return jsonify(fns[:200])


@app.route("/stats")
def get_stats():
    known_count = len([f for f in os.listdir(KNOWN_DIR) if os.path.isfile(os.path.join(KNOWN_DIR, f))])
    unknown_count = len([f for f in os.listdir(UNKNOWN_DIR) if os.path.isfile(os.path.join(UNKNOWN_DIR, f))])
    total_att = get_total_attendance_count()
    today_att = get_today_attendance_count()
    return jsonify(
        {
            "known_faces": known_count,
            "unknown_faces": unknown_count,
            "total_attendance": total_att,
            "today_attendance": today_att,
        }
    )


@app.route("/add_face", methods=["POST"])
def add_face():
    try:
        name = request.form.get("name", "").strip()
        imgfile = request.files.get("image")
        if not name or not imgfile:
            return jsonify({"success": False, "error": "Name and image are required"}), 400
        safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        out_path = os.path.join(KNOWN_DIR, f"{safe_name}.jpg")
        arr = np.frombuffer(imgfile.read(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400
        ok, enc = cv2.imencode(".jpg", img)
        if not ok:
            return jsonify({"success": False, "error": "Encode failed"}), 500
        enc.tofile(out_path)
        person = face_app.get(img)
        if not person:
            return jsonify({"success": False, "error": "No face found in image"}), 400
        face_index.add(safe_name, person[0].embedding)
        return jsonify({"success": True})
    except Exception as e:
        logger.exception("add_face failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/known/<path:filename>')
def serve_known(filename):
    fpath = os.path.join(KNOWN_DIR, filename)
    if not os.path.isfile(fpath):
        return ("Not found", 404)
    return send_from_directory(KNOWN_DIR, filename)


@app.route('/unknown/<path:filename>')
def serve_unknown(filename):
    fpath = os.path.join(UNKNOWN_DIR, filename)
    if not os.path.isfile(fpath):
        return ("Not found", 404)
    return send_from_directory(UNKNOWN_DIR, filename)


@app.route("/reload_index", methods=["POST"])
def reload_index():
    try:
        face_index.build_from_dir(KNOWN_DIR)
        return jsonify({"success": True, "count": len(face_index.labels)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health")
def health():
    # Basic component health
    cam_ok = cam.cap is not None and cam.cap.isOpened()
    return jsonify({
        "status": "ok", 
        "camera_open": bool(cam_ok), 
        "fps": worker.get_fps(),
        "cooldown_remaining": max(0, BARRIER_COOLDOWN - int(time.time() - _last_barrier_open_time))
    })


# ---------------- Graceful Shutdown ----------------
_stop_once = threading.Event()


def shutdown(*_):
    if _stop_once.is_set():
        return
    _stop_once.set()
    try:
        worker.stop_event.set()
        cam.stop()
        if SAVE_UNKNOWN_ASYNC:
            unknown_save_queue.put(None)
    except Exception:
        pass


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ---------------- Boot ----------------
if __name__ == "__main__":
    # Init DB + models
    init_db()
    init_insightface()
    # Build known index
    face_index.build_from_dir(KNOWN_DIR)
    # Start camera + worker
    cam.start()
    worker.start()
    # Run Flask
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
