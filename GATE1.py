import cv2
import threading
import queue
import time
import os
import csv
import serial
from datetime import datetime
from ultralytics import YOLO
import easyocr

# ---------------- Config ----------------
RTSP_URL = "rtsp://admin:afaqkhan-1@192.168.18.116:554/Streaming/channels/101"

AUTHORIZED_PLATES = {
    "EW080": "Fahad Hussain",
    "KHT9090": "Ali Akbar",
    "PES5566": "Talal Syed"
}

SNAPSHOT_DIR = "snapshots"
CSV_FILE = "plate_log.csv"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Logging init
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Plate', 'Authorized', 'Snapshot'])

# ---------------- USB Relay ----------------
class USBRelay:
    def __init__(self, port="/dev/tty.usbserial-110", baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            print(f"[USBRelay] Connected to {port}")
        except Exception as e:
            print("[ERROR] Could not connect to relay:", e)
            self.ser = None

    def relay_on(self):
        if self.ser:
            # ⚠️ Example command, may differ per relay model!
            self.ser.write(b'\xA0\x01\x01\xA2')
            print("[USBRelay] Relay ON")

    def relay_off(self):
        if self.ser:
            # ⚠️ Example command, may differ per relay model!
            self.ser.write(b'\xA0\x01\x00\xA1')
            print("[USBRelay] Relay OFF")

    def pulse(self, duration=0.5):
        """Turn relay ON briefly, then OFF"""
        self.relay_on()
        time.sleep(duration)
        self.relay_off()

relay = USBRelay(port="/dev/tty.usbserial-110")  # update with your USB relay path

# ---------------- Threaded RTSP Reader ----------------
frame_queue = queue.Queue(maxsize=1)

def rtsp_reader(url):
    while True:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("[WARN] Cannot open stream, retrying...")
            time.sleep(2)
            continue
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to grab frame, reconnecting...")
                break
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)
        cap.release()
        time.sleep(1)

threading.Thread(target=rtsp_reader, args=(RTSP_URL,), daemon=True).start()

# ---------------- Load Models ----------------
yolo_model = YOLO("license_plate_detector.pt")  # replace with your trained YOLO model
reader = easyocr.Reader(['en'])

# ---------------- Helper Functions ----------------
def save_snapshot(frame, plate_text):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plate_text}_{timestamp}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, frame)
    return path

def log_plate(plate_text, authorized, snapshot):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, plate_text, authorized, snapshot])

# ---------------- Track plates ----------------
last_seen_plate = {}     # plate -> last snapshot time
last_trigger_time = {}   # plate -> last barrier trigger time
SNAP_INTERVAL = 10       # seconds between snapshots of same plate
TRIGGER_COOLDOWN = 15    # seconds between barrier triggers for same plate

# ---------------- Main Loop ----------------
while True:
    try:
        frame = frame_queue.get(timeout=2)
    except queue.Empty:
        print("[WARN] No frame received in 2 sec")
        continue

    results = yolo_model(frame, verbose=False)
    for r in results:
        if getattr(r, "boxes", None) is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        for (x1, y1, x2, y2) in boxes:
            # Crop with padding
            pad = 5
            x1_p = max(0, x1 - pad)
            y1_p = max(0, y1 - pad)
            x2_p = min(frame.shape[1], x2 + pad)
            y2_p = min(frame.shape[0], y2 + pad)
            plate_roi = frame[y1_p:y2_p, x1_p:x2_p]
            if plate_roi.size == 0:
                continue

            # OCR
            ocr_results = reader.readtext(plate_roi)
            plate_text = "".join([text.upper().replace(" ", "").replace(":", "")
                                  for (_, text, _) in ocr_results])
            if not plate_text:
                plate_text = "UNKNOWN"

            # Authorization check
            if plate_text in AUTHORIZED_PLATES:
                label = f"Authorized: {AUTHORIZED_PLATES[plate_text]}"
                color = (0, 255, 0)
                authorized = True
            else:
                label = f"Unauthorized ({plate_text})"
                color = (0, 0, 255)
                authorized = False
                print("[ALERT] Unauthorized vehicle:", plate_text)

            now = time.time()

            # Snapshot/logging
            if plate_text not in last_seen_plate or (now - last_seen_plate[plate_text] > SNAP_INTERVAL):
                snap_path = save_snapshot(frame, plate_text)
                log_plate(plate_text, authorized, snap_path)
                last_seen_plate[plate_text] = now

            # Barrier trigger (authorized only)
            if authorized:
                if plate_text not in last_trigger_time or (now - last_trigger_time[plate_text] > TRIGGER_COOLDOWN):
                    print(f"[BARRIER] Opening for {plate_text} ({AUTHORIZED_PLATES[plate_text]})")
                    threading.Thread(target=relay.pulse, args=(0.5,), daemon=True).start()
                    last_trigger_time[plate_text] = now

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("ANPR + Barrier Control", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit program
    if key == ord('q'):
        break

    # Manual relay test
    if key == ord('t'):
        print("[TEST] Manual relay pulse triggered")
        threading.Thread(target=relay.pulse, args=(0.5,), daemon=True).start()

cv2.destroyAllWindows()
