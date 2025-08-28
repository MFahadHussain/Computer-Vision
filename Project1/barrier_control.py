import serial
import time
import logging

# ---------------- Configuration ----------------
RELAY_PORT = "COM3"   # Adjust depending on your system
BAUD_RATE = 9600

# Track states separately
ENTRANCE_STATE = "closed"
EXIT_STATE = "closed"

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Core Function ----------------
def send_command(cmd: bytes, delay=0.2):
    """Send a command to the CH9102 relay with proper reset/flush."""
    try:
        with serial.Serial(RELAY_PORT, BAUD_RATE, timeout=1) as ser:
            ser.setDTR(False)
            ser.setRTS(False)
            time.sleep(delay)

            ser.write(cmd)
            ser.flush()
            time.sleep(delay)
        return True

    except Exception as e:
        logger.error(f"❌ Serial error: {e}")
        return False


# ---------------- Entrance Barrier ----------------
def open_barrier():
    """Open the entrance barrier."""
    global ENTRANCE_STATE
    if send_command(b'1'):
        logger.info("✅ Entrance Barrier OPENED")
        ENTRANCE_STATE = "open"

def close_barrier():
    """Close the entrance barrier."""
    global ENTRANCE_STATE
    if send_command(b'2'):
        logger.info("⛔ Entrance Barrier CLOSED")
        ENTRANCE_STATE = "closed"


# ---------------- Exit Barrier ----------------
def open_exit_barrier():
    """Open the exit barrier."""
    global EXIT_STATE
    if send_command(b'3'):   # different relay channel
        logger.info("✅ Exit Barrier OPENED")
        EXIT_STATE = "open"

def close_exit_barrier():
    """Close the exit barrier."""
    global EXIT_STATE
    if send_command(b'4'):   # different relay channel
        logger.info("⛔ Exit Barrier CLOSED")
        EXIT_STATE = "closed"


# ---------------- Status Check ----------------
def get_barrier_status():
    """Return both entrance and exit barrier status."""
    return {
        "entrance": ENTRANCE_STATE,
        "exit": EXIT_STATE
    }


# ---------------- Test ----------------
if __name__ == "__main__":
    logger.info("🔄 Testing Barriers")

    # Entrance cycle
    open_barrier()
    time.sleep(2)
    close_barrier()

    # Exit cycle
    open_exit_barrier()
    time.sleep(2)
    close_exit_barrier()

    logger.info(f"📌 Final Status: {get_barrier_status()}")
