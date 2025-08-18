import time
import serial
import logging

# ------------------------------
# CONFIGURATION
# ------------------------------
SERIAL_PORT = "/dev/tty.usbserial-1420"   # Replace with your actual device
BAUD_RATE = 9600                          # Adjust if barrier uses different baud
BARRIER_COOLDOWN = 5                      # Seconds between triggers

# Simulated authorized entries
AUTHORIZED_PLATES = {"EW080": "Fahad Hussain", "ABC123": "Test User"}
AUTHORIZED_FACES = {"Fahad Bangash": "Resident", "Ali Khan": "Staff"}

# ------------------------------
# LOGGER SETUP
# ------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AccessControl")

# ------------------------------
# SERIAL CONNECTION
# ------------------------------
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    logger.info(f"Connected to barrier on {SERIAL_PORT} at {BAUD_RATE} baud")
except Exception as e:
    ser = None
    logger.error(f"Failed to open serial port: {e}")


# ------------------------------
# ACCESS CONTROL CLASS
# ------------------------------
class AccessControlSystem:
    def __init__(self):
        self.last_triggered = 0

    def trigger_barrier(self, reason: str):
        """Send command to barrier over serial if cooldown passed."""
        now = time.time()
        if now - self.last_triggered < BARRIER_COOLDOWN:
            logger.warning("Barrier trigger ignored (cooldown active).")
            return

        self.last_triggered = now
        logger.info(f"BARRIER TRIGGERED: {reason}")

        if ser and ser.is_open:
            try:
                # ⚠️ Replace with actual command your barrier expects
                ser.write(b"OPEN\n")
                logger.info("Sent OPEN command to barrier via serial")
            except Exception as e:
                logger.error(f"Serial error: {e}")
        else:
            logger.warning("Serial port not available, barrier not triggered")

    def check_plate(self, plate_text: str):
        """Check license plate against authorized list."""
        if plate_text in AUTHORIZED_PLATES:
            owner = AUTHORIZED_PLATES[plate_text]
            self.trigger_barrier(f"Authorized plate: {plate_text} ({owner})")
        else:
            logger.info(f"Unauthorized plate detected: {plate_text}")

    def check_face(self, face_name: str):
        """Check face recognition result against authorized list."""
        if face_name in AUTHORIZED_FACES:
            role = AUTHORIZED_FACES[face_name]
            self.trigger_barrier(f"Authorized face: {face_name} ({role})")
        else:
            logger.info(f"Unauthorized face detected: {face_name}")

    def manual_trigger(self):
        """Manually trigger the barrier for testing."""
        self.trigger_barrier("Manual test trigger")


# ------------------------------
# MAIN LOOP (Simulation + Manual CLI)
# ------------------------------
if __name__ == "__main__":
    system = AccessControlSystem()

    print("Access Control System Ready 🚦")
    print("Commands:")
    print("  plate <TEXT>  → Simulate license plate detection")
    print("  face <NAME>   → Simulate face detection")
    print("  open          → Manually open the barrier")
    print("  exit          → Quit\n")

    while True:
        cmd = input("> ").strip().split(maxsplit=1)

        if not cmd:
            continue

        action = cmd[0].lower()

        if action == "plate" and len(cmd) > 1:
            system.check_plate(cmd[1].upper())

        elif action == "face" and len(cmd) > 1:
            system.check_face(cmd[1])

        elif action == "open":
            system.manual_trigger()

        elif action == "exit":
            print("Exiting...")
            break

        else:
            print("Unknown command. Try: plate <TEXT>, face <NAME>, open, exit")
