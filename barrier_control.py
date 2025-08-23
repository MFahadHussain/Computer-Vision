import serial
import time

RELAY_PORT = "COM3"
BAUD_RATE = 9600

def send_command(cmd: bytes, delay=0.2):
    """Send a command to the CH9102 relay with proper reset/flush."""
    try:
        with serial.Serial(RELAY_PORT, BAUD_RATE, timeout=1) as ser:
            # reset lines
            ser.setDTR(False)
            ser.setRTS(False)
            time.sleep(delay)

            # send command
            ser.write(cmd)
            ser.flush()
            time.sleep(delay)  # allow relay to react

    except Exception as e:
        print(f"❌ Serial error: {e}")


def open_barrier(duration=3):
    send_command(b'1')
    print("✅ Relay ON - Barrier opening...")
    time.sleep(duration)
    send_command(b'2')
    print("⛔ Relay OFF - Barrier closed.")


# ---------------- Loop Test ----------------
if __name__ == "__main__":
    for i in range(5):
        print(f"\n🔄 Cycle {i+1}")
        open_barrier(3)
        time.sleep(2)
