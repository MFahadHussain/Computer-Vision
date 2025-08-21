# Configuration for AI Security System

# RTSP Camera URLs
FACE_CAMERA_RTSP = "rtsp://username:password@camera-ip:554/stream1"
PLATE_CAMERA_RTSP = "rtsp://username:password@camera-ip:554/stream2"

# Recognition settings
FACE_RECOGNITION_MODEL = "buffalo_l"
FACE_RECOGNITION_THRESHOLD = 0.6
FACE_RECOGNITION_COOLDOWN = 5  # seconds

PLATE_CONFIDENCE_THRESHOLD = 0.7
PLATE_RECOGNITION_COOLDOWN = 5  # seconds

# Database settings
DATABASE_PATH = "security.db"

# Hardware settings (for Raspberry Pi)
BARRIER_GPIO_PIN = 18
GREEN_LED_GPIO_PIN = 23
RED_LED_GPIO_PIN = 24

# Snapshot settings
SNAPSHOT_DIRECTORY = "snapshots"