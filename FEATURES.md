# New Features Documentation

## Overview

This document describes the new features added to the YOLO-World V2 Video Detector:

1. **Polygon ROI Save/Load**: Save and reuse ROI configurations
2. **Multi-Feed Processing**: Process multiple video feeds simultaneously
3. **Click-to-Track**: Select an object by clicking to track it across all feeds

## 1. Polygon ROI Save/Load

### Save ROI Configuration

After selecting polygon ROIs, you can save them to a JSON file for reuse:

```bash
python main.py -i video.mp4 -o output.mp4 --save-roi roi_config.json
```

### Load ROI Configuration

Load a previously saved ROI configuration:

```bash
python main.py -i video.mp4 -o output.mp4 --load-roi roi_config.json
```

### ROI Configuration Format

The ROI configuration is saved as JSON with the following structure:

```json
{
  "polygons": [
    {
      "id": 0,
      "points": [[x1, y1], [x2, y2], [x3, y3], ...],
      "min_x": 0,
      "min_y": 0,
      "width": 1920,
      "height": 1080
    }
  ]
}
```

## 2. Multi-Feed Processing

Process multiple video feeds simultaneously with synchronized tracking:

```bash
python main.py -i feed1.mp4 feed2.mp4 feed3.mp4 -o out1.mp4 out2.mp4 out3.mp4
```

### Features:
- **Synchronized Processing**: All feeds are processed frame-by-frame in sync
- **Shared ROI**: ROI selected on the first feed is applied to all feeds
- **Unified Tracking**: Target object is tracked across all feeds
- **Multi-Feed Preview**: View all feeds in a grid layout

### Requirements:
- Number of input videos must match number of output videos
- All videos should have similar frame rates for best results

## 3. Click-to-Track

### How It Works

1. **ROI Selection**: First, select polygon ROIs on the first feed (or load from file)
2. **Initial Detection**: System detects objects in the first frame
3. **Object Selection**: Click on the object you want to track
4. **Multi-Feed Tracking**: The selected object is tracked across all video feeds

### Object Selection Methods

#### Method 1: Click on Detected Object
- Detected objects are shown with bounding boxes
- Click directly on a detected object to select it

#### Method 2: Draw Bounding Box
- Click and drag to draw a bounding box around the object
- The system will find the best matching detection

### Tracking Features

- **ReID Support**: Uses BoT-SORT with ReID for robust tracking
- **Cross-Feed Tracking**: Same object tracked across multiple camera angles
- **Visual Feedback**: 
  - Green box: Target object is being tracked
  - Red text: Target object lost
  - Track ID displayed on each feed

### Example Workflow

```bash
# Step 1: Process multiple feeds with click-to-track
python main.py \
  -i camera1.mp4 camera2.mp4 camera3.mp4 \
  -o tracked1.mp4 tracked2.mp4 tracked3.mp4 \
  --load-roi my_roi.json

# The system will:
# 1. Load ROI configuration
# 2. Show initial detections
# 3. Wait for you to click on target object
# 4. Track that object across all 3 feeds
```

## Technical Details

### Object Tracker

The `ObjectSpecificTracker` class:
- Initializes with target bounding box and optional class ID
- Uses IoU (Intersection over Union) and center distance for matching
- Maintains separate tracker instances for each feed
- Updates target position based on best matching detection

### Multi-Feed Processor

The `MultiFeedProcessor` class:
- Manages multiple video readers and writers
- Synchronizes frame reading across feeds
- Applies same ROI to all feeds
- Creates unified preview grid
- Tracks target object independently in each feed

### ROI Manager

The `ROIManager` class:
- Saves ROI configurations to JSON
- Loads ROI configurations from JSON
- Validates ROI data integrity
- Handles file I/O errors gracefully

## Usage Tips

1. **ROI Selection**: 
   - Select ROI on the first feed carefully
   - Save ROI for reuse across sessions
   - ROI is applied to all feeds in multi-feed mode

2. **Object Selection**:
   - Select object in first frame for best results
   - Choose objects with clear visual features
   - Avoid selecting objects that are too small

3. **Multi-Feed Setup**:
   - Use videos with similar resolutions
   - Ensure videos are synchronized (same start time)
   - Consider frame rate differences

4. **Performance**:
   - Multi-feed processing is more CPU/GPU intensive
   - Consider processing fewer feeds if performance is an issue
   - Use `--no-preview` for faster processing

## Troubleshooting

### Object Not Tracked
- Ensure object is clearly visible in first frame
- Check that object is within ROI boundaries
- Try selecting a larger bounding box

### Multi-Feed Sync Issues
- Verify all videos have similar frame rates
- Check that videos start at the same time
- Ensure all videos have the same resolution

### ROI Load Errors
- Verify JSON file format is correct
- Check that ROI coordinates are valid
- Ensure file path is correct


