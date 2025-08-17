import cv2
import easyocr
import os
from datetime import datetime
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

# reuse your easyocr.Reader instance
# reader = easyocr.Reader(['en'])

MIN_OCR_CONFIDENCE = 0.35
VALID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NORMALIZE_REPLACEMENTS = {
    " ": "", "-": "", "_": "", ":": "", ".": "", "/": "", "|": "",
    "O": "0", "I": "1", "B": "8", "S": "5"  # common confusions - tweak per region
}

def normalize_plate_text(s: str) -> str:
    s = s.upper()
    for a, b in NORMALIZE_REPLACEMENTS.items():
        s = s.replace(a, b)
    s = "".join(ch for ch in s if ch in VALID_CHARS)
    return s

def ocr_multiline_plate(plate_bgr, reader):
    """
    plate_bgr : BGR image cropped to plate (from YOLO bbox)
    reader : easyocr.Reader
    returns: (combined_text, confidence)
    """
    try:
        # 1) Preprocess: gray, denoise, resize, binarize
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 75, 75)  # denoise while keeping edges
        h, w = gray.shape[:2]
        if h < 20 or w < 60:
            # tiny — upscale
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            h, w = gray.shape

        # adaptive threshold or Otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2) Horizontal projection & find split row around middle
        proj = np.sum(thresh == 255, axis=1).astype(np.int32)  # white pixel counts per row
        # smooth projection
        proj_smooth = cv2.GaussianBlur(proj.reshape(-1,1).astype(np.float32), (9,1), 0).flatten()
        # search for minimal valley in center region (25%..75%)
        top_search = int(h * 0.25)
        bottom_search = int(h * 0.75)
        if bottom_search <= top_search:
            split_row = h // 2
        else:
            rel = np.argmin(proj_smooth[top_search:bottom_search])
            split_row = top_search + int(rel)

        # Ensure split produces two non-empty regions
        if split_row <= 3 or split_row >= h - 3:
            # fallback: just split middle
            split_row = h // 2

        top_img = plate_bgr[0:split_row, :]
        bottom_img = plate_bgr[split_row:, :]

        # 3) OCR each half (use reader.readtext on grayscale or color crop)
        top_res = reader.readtext(top_img, detail=1, paragraph=False)
        bottom_res = reader.readtext(bottom_img, detail=1, paragraph=False)

        def best_text(res):
            best = ("", 0.0)
            for (_box, txt, prob) in res:
                if prob >= MIN_OCR_CONFIDENCE and len(txt.strip()) >= 1:
                    if prob > best[1]:
                        best = (txt.strip(), float(prob))
            return best

        top_txt, top_conf = best_text(top_res)
        bottom_txt, bottom_conf = best_text(bottom_res)

        # fallback: if any half empty, try whole plate
        if top_txt == "" and bottom_txt == "":
            whole_res = reader.readtext(plate_bgr, detail=1, paragraph=False)
            whole_txt, whole_conf = best_text(whole_res)
            combined = normalize_plate_text(whole_txt)
            return combined, whole_conf

        # normalize & combine
        top_norm = normalize_plate_text(top_txt)
        bottom_norm = normalize_plate_text(bottom_txt)
        # assembly rule: letters on top, numbers bottom (Pakistan) -> join with space
        candidate = (top_norm + " " + bottom_norm).strip()
        # overall confidence (weighted)
        confs = [c for c in (top_conf, bottom_conf) if c > 0]
        overall_conf = (sum(confs) / len(confs)) if confs else 0.0

        return candidate, overall_conf

    except Exception as e:
        # fallback try whole
        try:
            res = reader.readtext(plate_bgr, detail=1, paragraph=False)
            if res:
                txts = sorted(res, key=lambda r: r[2], reverse=True)
                txt = txts[0][1]
                return normalize_plate_text(txt), float(txts[0][2])
        except Exception:
            pass
        return "", 0.0

# -------------------------------
# Configuration
# -------------------------------
AUTHORIZED_PLATES = {
    "AFT042": "Fahad Hussain",
    "KHT9090": "Ali Akbar",
    "PES5566": "Talal Syed"
}

SNAPSHOT_DIR = "snapshots"
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

# Load YOLOv8 license plate detector
model = YOLO("license_plate_detector.pt")  # replace with your trained/custom model

# Initialize OCR
reader = easyocr.Reader(['en'])


# -------------------------------
# Helper Functions
# -------------------------------
def save_snapshot(frame, plate_text):
    """Save full car snapshot with timestamp and plate number."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SNAPSHOT_DIR}/{plate_text}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[INFO] Full car snapshot saved: {filename}")
    return filename


def recognize_plate(frame):
    """Detect license plate and extract text using OCR."""
    results = model(frame, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)  # bounding boxes

        for (x1, y1, x2, y2) in boxes:
            plate_roi = frame[y1:y2, x1:x2].copy()
            if plate_roi.size == 0:
                continue

            # OCR recognition
            ocr_results = reader.readtext(plate_roi)
            for (_, text, prob) in ocr_results:
                text = text.upper().replace(" ", "").replace(":", "").strip()

                # Draw bounding box around plate
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Authorization check
                if text in AUTHORIZED_PLATES:
                    label = f"Authorized: {AUTHORIZED_PLATES[text]}"
                    color = (0, 255, 0)
                else:
                    label = f"Unauthorized ({text})"
                    color = (0, 0, 255)

                # Display info on frame
                cv2.putText(frame, f"Plate: {text}", (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Save full car snapshot
                save_snapshot(frame, text)

                return text, (text in AUTHORIZED_PLATES)

    return None, False


def main():
    cap = cv2.VideoCapture(0)  # webcam / replace with IP camera link

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        plate_text, authorized = recognize_plate(frame)

        if authorized:
            cv2.putText(frame, "Access Granted ✅", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Scanning...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        cv2.imshow("ANPR System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
