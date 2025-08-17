import cv2
import easyocr
import os
from datetime import datetime
from ultralytics import YOLO

# ------------------------------- Configuration -------------------------------
AUTHORIZED_PLATES = {
    "AFT042": "Fahad Hussain",
    "KHT9090": "Ali Akbar",
    "PES5566": "Talal Syed"
}

SNAPSHOT_DIR = "snapshots"
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

# Load YOLOv8 trained model
model = YOLO("license_plate_detector.pt")  # replace with your trained weights

# Initialize EasyOCR
reader = easyocr.Reader(['en'])


# ------------------------------- Helper Functions -------------------------------
def save_snapshot(frame, plate_text):
    """Save full car snapshot with timestamp and plate number."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SNAPSHOT_DIR}/{plate_text}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[INFO] Snapshot saved: {filename}")
    return filename


def recognize_plate(frame):
    """Detect license plate and extract text using OCR."""
    results = model(frame, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        for (x1, y1, x2, y2) in boxes:
            plate_roi = frame[y1:y2, x1:x2].copy()
            if plate_roi.size == 0:
                continue

            # OCR
            ocr_results = reader.readtext(plate_roi)
            plate_text = "".join([text.upper().replace(" ", "").replace(":", "") for (_, text, prob) in ocr_results])

            # Draw green box and plate text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Authorization check
            if plate_text in AUTHORIZED_PLATES:
                label = f"Authorized: {AUTHORIZED_PLATES[plate_text]}"
                color = (0, 255, 0)
            else:
                label = f"Unauthorized ({plate_text})"
                color = (0, 0, 255)
            cv2.putText(frame, label, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Save snapshot
            save_snapshot(frame, plate_text)

            return plate_text, (plate_text in AUTHORIZED_PLATES)

    return None, False


# ------------------------------- Video Stream -------------------------------
def main():
    cap = cv2.VideoCapture(0)  # or replace with RTSP/IP camera link

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        plate_text, authorized = recognize_plate(frame)

        if authorized:
            cv2.putText(frame, "Access Granted ✅", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Scanning...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        cv2.imshow("ANPR System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

