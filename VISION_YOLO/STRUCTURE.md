# Project Structure

## Modular Architecture

The codebase has been refactored into a clean, modular structure:

```
VISION_YOLO/
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── STRUCTURE.md                 # This file
│
├── detection/                   # Detection module
│   ├── __init__.py             # Module exports
│   └── detector.py             # YOLOWorldROIDetector class
│
├── tracking/                    # Tracking module
│   ├── __init__.py             # Module exports
│   └── tracker.py              # BoTSORTWrapper class
│
├── video/                       # Video I/O module
│   ├── __init__.py             # Module exports
│   ├── video_reader.py         # ThreadSafeVideoReader class
│   └── video_writer.py         # ThreadSafeVideoWriter class
│
├── ui/                          # UI components
│   ├── __init__.py             # Module exports
│   └── polygon_selector.py    # EnhancedPolygonSelector class
│
├── models/                      # Data models
│   ├── __init__.py             # Module exports
│   └── data_models.py         # PolygonROI, ROIConfig, Detection classes
│
└── utils/                       # Utility modules
    ├── __init__.py             # Module exports
    ├── model_checker.py       # Model file validation
    └── logger_config.py       # Logging configuration
```

## Module Responsibilities

### `main.py`
- Entry point for the application
- Command-line argument parsing
- Initializes detector and processes video

### `detection/`
- **detector.py**: Main detector class
  - Model loading and initialization
  - ROI selection and cropping
  - Detection and tracking coordination
  - Annotation drawing
  - Video processing pipeline

### `tracking/`
- **tracker.py**: Tracking wrapper
  - BoT-SORT tracker initialization
  - ByteTrack fallback support
  - Parameter mapping for Supervision 0.27.0+
  - Detection-to-tracking conversion

### `video/`
- **video_reader.py**: Thread-safe video reading
  - Background frame reading
  - Error handling
  - Video property extraction
  
- **video_writer.py**: Thread-safe video writing
  - Background frame writing
  - Error handling
  - Output directory creation

### `ui/`
- **polygon_selector.py**: Interactive ROI selection
  - Polygon drawing and editing
  - Undo/redo functionality
  - Grid snapping
  - Point manipulation
  - Help overlay

### `models/`
- **data_models.py**: Data structures
  - `PolygonROI`: Single polygon ROI with mask computation
  - `ROIConfig`: Multiple ROI configuration
  - `Detection`: Detection result with tracking info

### `utils/`
- **model_checker.py**: Model file validation
  - Checks for corrupted model files
  - Cleans invalid model files
  
- **logger_config.py**: Logging setup
  - Configures logging format and level

## Import Structure

All modules use absolute imports from the project root:

```python
# Example imports
from models.data_models import PolygonROI, ROIConfig, Detection
from ui.polygon_selector import EnhancedPolygonSelector
from tracking.tracker import BoTSORTWrapper
from video.video_reader import ThreadSafeVideoReader
from video.video_writer import ThreadSafeVideoWriter
from utils.model_checker import check_and_clean_model_files
from utils.logger_config import setup_logger
```

## Usage

Run from the project root:

```bash
python main.py -i video.mp4 -o output.mp4
```

The project structure ensures:
- ✅ Clear separation of concerns
- ✅ Easy to maintain and extend
- ✅ Reusable components
- ✅ Testable modules
- ✅ Clean imports

