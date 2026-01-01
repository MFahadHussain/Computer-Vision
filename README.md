# YOLO-World V2 Video Detector

Enhanced YOLO-World V2 Video Detector with Multiple Polygon ROI Cropping, Editing, and BoT-SORT Tracking with ReID

> **Note**: If you see dependency conflict warnings during installation, they're usually harmless. See [DEPENDENCY_CONFLICTS.md](DEPENDENCY_CONFLICTS.md) for details.

## Project Structure

```
VISION_YOLO/
├── main.py                 # Main entry point
├── detection/              # Detection module
│   ├── __init__.py
│   └── detector.py         # YOLO-World detector class
├── tracking/               # Tracking module
│   ├── __init__.py
│   └── tracker.py          # BoT-SORT wrapper
├── video/                  # Video I/O module
│   ├── __init__.py
│   ├── video_reader.py     # Thread-safe video reader
│   └── video_writer.py     # Thread-safe video writer
├── ui/                     # UI components
│   ├── __init__.py
│   └── polygon_selector.py # Enhanced polygon ROI selector
├── models/                 # Data models
│   ├── __init__.py
│   └── data_models.py      # PolygonROI, ROIConfig, Detection
└── utils/                  # Utility modules
    ├── __init__.py
    ├── model_checker.py    # Model file validation
    └── logger_config.py    # Logging configuration
```

## Features

- **Multiple Polygon ROI Selection**: Define and edit multiple polygon regions of interest
- **ROI Save/Load**: Save and reuse ROI configurations across sessions
- **Enhanced UI**: Undo/redo, grid snap, point insertion/deletion
- **BoT-SORT Tracking with ReID**: Robust object tracking with re-identification support using boxmot library
- **Multi-Feed Processing**: Process multiple video feeds simultaneously with synchronized tracking
- **Click-to-Track**: Select an object by clicking to track it across all feeds
- **Thread-Safe Video Processing**: Efficient video I/O with background threads
- **Error Handling**: Comprehensive error handling throughout the pipeline

## Installation

### Recommended: Use Virtual Environment

To avoid dependency conflicts, use a virtual environment:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Direct Installation

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install ultralytics opencv-python boxmot torch numpy
```

**Note**: If you encounter dependency conflicts, see [INSTALLATION.md](INSTALLATION.md) for troubleshooting.

## Usage

### Basic Usage

```bash
python main.py -i video.mp4 -o output.mp4
```

### With Custom Classes

```bash
python main.py -i video.mp4 -o output.mp4 -c "person" "car" "dog"
```

### Without BoT-SORT Tracking

```bash
python main.py -i video.mp4 -o output.mp4 --no-botsort
```

### Multi-Feed Processing with Click-to-Track

```bash
python main.py -i feed1.mp4 feed2.mp4 feed3.mp4 -o out1.mp4 out2.mp4 out3.mp4
```

### Save and Load ROI Configuration

```bash
# Save ROI after selection
python main.py -i video.mp4 -o output.mp4 --save-roi roi_config.json

# Load saved ROI
python main.py -i video.mp4 -o output.mp4 --load-roi roi_config.json
```

### Without Preview Window

```bash
python main.py -i video.mp4 -o output.mp4 --no-preview
```

### Custom Confidence Threshold

```bash
python main.py -i video.mp4 -o output.mp4 --conf 0.5
```

## Command Line Arguments

- `-i, --input`: Path(s) to input video file(s). Use multiple paths for multi-feed processing (required)
- `-o, --output`: Path(s) for output annotated video(s). Must match number of inputs (required)
- `-m, --model`: Path to YOLO-World model (default: yolov8l-worldv2.pt)
- `-c, --classes`: Custom classes to detect (space-separated)
- `--conf`: Confidence threshold (default: 0.3)
- `--no-preview`: Disable real-time preview window
- `--no-botsort`: Disable BoT-SORT tracking and use default tracking
- `--load-roi`: Path to load ROI configuration JSON file
- `--save-roi`: Path to save ROI configuration JSON file

## ROI Selection Controls

### Selection Mode
- **Click**: Add points to the polygon
- **SPACE/ENTER**: Finish current polygon
- **n**: Start a new polygon
- **e**: Enter edit mode
- **f**: Finish all polygons
- **c**: Cancel
- **r**: Reset current polygon
- **z**: Undo
- **x**: Redo
- **g**: Toggle grid snap
- **h**: Toggle help

### Edit Mode
- **Click on polygon**: Select polygon
- **Click and drag points**: Edit point positions
- **i**: Insert point
- **d**: Delete selected polygon
- **r**: Delete selected point
- **s**: Exit edit mode
- **z**: Undo
- **x**: Redo
- **g**: Toggle grid snap
- **h**: Toggle help
- **ESC**: Cancel edit mode

## Module Documentation

### Detection Module
The `detection` module contains the main `YOLOWorldROIDetector` class that handles:
- Model loading and initialization
- ROI selection and cropping
- Detection and tracking
- Annotation drawing
- Video processing

### Tracking Module
The `tracking` module provides the `BoTSORTWrapper` class that:
- Wraps BoT-SORT tracker from boxmot library
- Supports ReID (re-identification) for better tracking across occlusions
- Uses Sparse Optical Flow (SOF) for camera motion compensation
- Handles parameter configuration for detection threshold, max age, and frame rate

### Video Module
The `video` module provides thread-safe video I/O:
- `ThreadSafeVideoReader`: Background frame reading
- `ThreadSafeVideoWriter`: Background frame writing

### UI Module
The `ui` module contains the `EnhancedPolygonSelector` class for:
- Interactive polygon ROI selection
- Point editing and manipulation
- Undo/redo functionality
- Grid snapping

### Models Module
The `models` module defines data structures:
- `PolygonROI`: Single polygon ROI configuration
- `ROIConfig`: Multiple ROI configuration
- `Detection`: Detection result with tracking info

### Utils Module
The `utils` module provides utility functions:
- Model file validation and cleaning
- Logging configuration

## License

This project uses YOLO-World V2 from Ultralytics and boxmot library for tracking.

