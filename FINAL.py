#!/usr/bin/env python3
import os
import cv2
import time
import json
import math
import queue
import shutil
import signal
import logging
import threading
from datetime import datetime, timedelta

# ---- 3rd party ----
# pip install ultralytics easyocr opencv-python requests
from ultralytics import YOLO
import easyocr
import numpy as np
import requests

# =======================
# Configuration
# =======================
# RTSP source (set via env or hardcode below)
RTSP_URL = os.environ.get("RTSP_URL", "rtsp://username:password@192.168.1.64:554/Streaming/channels/101")

# If you trained a local plate model, put its path here; otherwise we use a small hub model:
YOLO_PLATE_WEIGHTS = os.environ.get("YOLO_PLATE_WEIGHTS", "license_plate_detector.pt")

# OCR languages
OCR_LANGS = os.environ.get("OCR_LANGS", "en").split(",")

# Plate authorization DB (simple dict here; swap with SQLite/REST if you like)
AUTHORIZED_PLATES = {
    "AFT042": "Fahad Hussain",
    "KHT9090": "Ali Akbar",
    "PES5566": "Talal Syed",
    "LEB5700": "Fahad Hussain",
}

# Require Face+Plate AND check?
ENABLE_FACE_AND_CHECK = True  # set False to allow plate-only
FACE_SERVICE_URL = os.environ.get("FACE_SERVICE_URL", "http://127.0.0.1:5000/detections")
FACE_VALID_WINDOW_SEC = int(os.environ.get("FACE_VALID_WINDOW_SEC", "8"))  # how recent a known face must be

# Snapshots
SNAP_DIR = os.environ.get("SNAP_DIR", "snapshots")
SNAP_FULL_DIR = os.path.join(SNAP_DIR, "full")
SNAP_PLATE_DIR = os.path.join(SNAP_DIR, "plates")
os.makedirs(SNAP_FULL_DIR, exist_ok=True)
os.makedirs(SNAP_PLATE_DIR, exist_ok=True)

# UI overlay & gate timing
SHOW_WINDOW = True  # set False if running headless
OPEN_DURATION_SEC = int(os.environ.get("OPEN_DURATION_SEC", "6"))  # how long to keep barrier open
DEBOUNCE_SEC = int(os.environ.get("DEBOUNCE_SEC", "10"))  # ignore repeated triggers within this window

# Plate text post-processing
MIN_OCR_CONFIDENCE = float(os.environ.get("MIN_OCR_CONFIDENCE", "0.35"))  # EasyOCR "prob" threshold
VALID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NORMALIZE_REPLACEMENTS = {
    " ": "", "-": "", "_": "", ":": "", ".": "", "/": "", "|": "",
    "O": "0", "I": "1", "B": "8",  # common OCR confusions
}

# GPIO pins (BCM numbering). Adjust for your relay module.
GPIO_GREEN_PIN = int(os.environ.get("GPIO_GREEN_PIN", "17"))
GPIO_BARRIER_PIN = int(os.environ.get("GPIO_BARRIER_PIN", "27"))
GPIO_BOLLARDS_PIN = int(os.environ.get("GPIO_BOLLARDS_PIN", "22"))

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("anpr-gate")

# =======================
# GPIO Controller (with fallback)
# =======================
class BaseGPIO:
    def setup(self, pin): raise NotImplementedError
    def set_high(self, pin): raise NotImplementedError
    def set_low(self, pin): raise NotImplementedError
    def cleanup(self): pass

class SimGPIO(BaseGPIO):
    def setup(self, pin):
        logger.info(f"(SIM GPIO) setup pin {pin}")
    def set_high(self, pin):
        logger.info(f"(SIM GPIO) pin {pin} -> HIGH")
    def set_low(self, pin):
        logger.info(f"(SIM GPIO) pin {pin} -> LOW")
    def cleanup(self):
        logger.info("(SIM GPIO) cleanup")

class RPiGPIO(BaseGPIO):
    def __init__(self):
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        self.GPIO.setmode(GPIO.BCM)
    def setup(self, pin):
        self.GPIO.setup(pin, self.GPIO.OUT)
        self.GPIO.output(pin, self.GPIO.LOW)  # default OFF
    def set_high(self, pin):
        self.GPIO.output(pin, self.GPIO.HIGH)
    def set_low(self, pin):
        self.GPIO.output(pin, self.GPIO.LOW)
    def cleanup(self):
        self.GPIO.cleanup()

def get_gpio():
    try:
        import RPi.GPIO as _  # just to test availability
        logger.info("RPi.GPIO available: using real GPIO")
        return RPiGPIO()
    except Exception:
        logger.info("RPi.GPIO not available: using SIMULATED GPIO")
        return SimGPIO()

GPIO = get_gpio()
for p in (GPIO_GREEN_PIN, GPIO_BARRIER_PIN, GPIO_BOLLARDS_PIN):
    GPIO.setup(p)

def signal_green_on():
    GPIO.set_high(GPIO_GREEN_PIN)
def signal_green_off():
    GPIO.set_low(GPIO_GREEN_PIN)

def barrier_up():
    GPIO.set_high(GPIO_BARRIER_PIN)
def barrier_down():
    GPIO.set_low(GPIO_BARRIER_PIN)

def bollards_down():
    GPIO.set_high(GPIO_BOLLARDS_PIN)
def bollards_up():
    GPIO.set_low(GPIO_BOLLARDS_PIN)

# =======================
# Face Recognition AND-check helper
# =======================
def recent_authorized_face_present() -> bool:
    """
    Query your existing face-recognition Flask app /detections endpoint and see if the latest
    detection is type='known' within FACE_VALID_WINDOW_SEC.
    Expected payload (from your app): [{'type':'known'|'unknown','name':..., 'timestamp': <epoch seconds>, ...}]
    """
    if not ENABLE_FACE_AND_CHECK:
        return True
    try:
        r = requests.get(FACE_SERVICE_URL, timeout=1.5)
        r.raise_for_status()
        dets = r.json() or []
        now = time.time()
        # Find a 'known' detection within window
        for d in dets[:10]:  # check a handful of recent items
            if d.get("type") == "known":
                ts = float(d.get("timestamp", 0))
                if (now - ts) <= FACE_VALID_WINDOW_SEC:
                    return True
        return False
    except Exception as e:
        logger.warning(f"Face service check failed: {e}")
        # Fail-safe: require plate-only if face service not reachable?
        return False

# =======================
# Utilities
# =======================
def normalize_plate_text(s: str) -> str:
    s = s.upper()
    for k, v in NORMALIZE_REPLACEMENTS.items():
        s = s.replace(k, v)
    s = "".join(ch for ch in s if ch in VALID_CHARS)
    return s

def save_snapshots(frame_bgr, plate_crop_bgr, plate_text):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(SNAP_FULL_DIR, f"{plate_text}_{ts}.jpg")
    plate_path = os.path.join(SNAP_PLATE_DIR, f"{plate_text}_{ts}.jpg")
    try:
        cv2.imwrite(full_path, frame_bgr)
        if plate_crop_bgr is not None and plate_crop_bgr.size > 0:
            cv2.imwrite(plate_path, plate_crop_bgr)
        logger.info(f"Snapshots saved: full={full_path}, plate={plate_path}")
    except Exception as e:
        logger.warning(f"Snapshot save failed: {e}")

def put_text(img, text, org, color=(255, 255, 255), scale=0.8, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# =======================
# ANPR Engine
# =======================
class ANPRGate:
    def __init__(self, rtsp_url, yolo_weights):
        self.rtsp_url = rtsp_url
        self.model = None
        self.ocr = easyocr.Reader(OCR_LANGS)
        self.last_trigger_time = 0.0

        # load YOLO (local file or hub path)
        try:
            if os.path.isfile(yolo_weights):
                self.model = YOLO(yolo_weights)
            else:
                self.model = YOLO(yolo_weights)  # hub model, will download once
            logger.info(f"YOLO plate model loaded: {yolo_weights}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{yolo_weights}': {e}")
            raise

        self.cap = None
        self.stop_event = threading.Event()

    def open_stream(self):
        # FFMPEG backend is more robust for RTSP
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # Optional performance hints
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 25)
        except Exception:
            pass

    def gate_trigger(self, frame, plate_text, reason="Authorized"):
        now = time.time()
        if (now - self.last_trigger_time) < DEBOUNCE_SEC:
            logger.info("Debounced: trigger skipped (too soon)")
            return
        self.last_trigger_time = now

        # Signals
        logger.info(f"=== GATE OPEN: {reason} | Plate={plate_text} ===")
        signal_green_on()
        barrier_up()
        bollards_down()

        save_snapshots(frame, None, plate_text)

        # Hold open
        time.sleep(OPEN_DURATION_SEC)

        # Close
        signal_green_off()
        barrier_down()
        bollards_up()
        logger.info("=== GATE CLOSED ===")

    def process_frame(self, frame):
        # Run detector
        results = self.model(frame, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return frame, None, None, None  # (overlay, text, conf, bbox)

        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        best_plate = None
        best_text = None
        best_conf = 0.0
        best_bbox = None

        for (x1, y1, x2, y2), det_conf in zip(boxes, confs):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            plate_roi = frame[y1:y2, x1:x2]
            if plate_roi.size == 0:
                continue

            # Preprocess for OCR
            roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.bilateralFilter(roi, 7, 40, 40)
            roi = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            ocr_results = self.ocr.readtext(roi, detail=1, paragraph=False)
            # pick the highest prob text-like result
            for (_bbox, raw_text, prob) in ocr_results:
                if prob < MIN_OCR_CONFIDENCE:
                    continue
                text = normalize_plate_text(raw_text)
                if len(text) < 4:
                    continue
                if prob > best_conf:
                    best_conf = prob
                    best_text = text
                    best_plate = plate_roi.copy()
                    best_bbox = (x1, y1, x2, y2)

        # Draw overlay if we have something
        overlay = frame.copy()
        if best_bbox is not None:
            (x1, y1, x2, y2) = best_bbox
            color = (0, 255, 0) if best_text in AUTHORIZED_PLATES else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            put_text(overlay, f"Plate: {best_text or '---'}", (x1, max(0, y1-35)), (255,255,0), 0.8, 2)
            put_text(overlay, f"OCR conf: {best_conf:.2f}", (x1, max(0, y1-10)), (200,200,200), 0.7, 2)

        return overlay, best_text, best_conf, best_plate

    def run(self):
        self.open_stream()
        if not self.cap or not self.cap.isOpened():
            logger.error("Cannot open RTSP stream.")
            return

        logger.info("ANPR gate started.")
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                logger.warning("RTSP read failed. Reconnecting in 1.0s...")
                time.sleep(1.0)
                self.open_stream()
                continue

            overlay, plate_text, ocr_conf, plate_crop = self.process_frame(frame)

            # Decide access
            if plate_text:
                authorized_plate = plate_text in AUTHORIZED_PLATES
                if ENABLE_FACE_AND_CHECK:
                    face_ok = recent_authorized_face_present()
                else:
                    face_ok = True

                status = "AUTHORIZED" if (authorized_plate and face_ok) else "DENIED"
                color = (0, 255, 0) if status == "AUTHORIZED" else (0, 0, 255)

                put_text(overlay, f"Plate Match: {'YES' if authorized_plate else 'NO'}", (20, 40), color, 0.9, 2)
                if ENABLE_FACE_AND_CHECK:
                    put_text(overlay, f"Face Match: {'YES' if face_ok else 'NO'}", (20, 70), color, 0.9, 2)
                put_text(overlay, f"Decision: {status}", (20, 100), color, 1.0, 3)

                # Trigger hardware if both conditions OK
                if authorized_plate and face_ok:
                    # Save both snapshots (full & plate)
                    save_snapshots(frame, plate_crop, plate_text)
                    self.gate_trigger(frame, plate_text, reason="Plate+Face authorized")

            # Display (optional)
            if SHOW_WINDOW:
                cv2.imshow("ANPR Gate (RTSP)", overlay if overlay is not None else frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        logger.info("Stopping ANPR gate...")
        try:
            self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    def stop(self):
        self.stop_event.set()

# =======================
# Main
# =======================
runner = None
def _graceful_shutdown(*_):
    logger.info("Signal received. Shutting down...")
    try:
        if runner:
            runner.stop()
    except Exception:
        pass
    try:
        GPIO.cleanup()
    except Exception:
        pass

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        runner = ANPRGate(RTSP_URL, YOLO_PLATE_WEIGHTS)
        runner.run()
    finally:
        try:
            GPIO.cleanup()
        except Exception:
            pass
