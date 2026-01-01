# Installation Guide

## Recommended: Use a Virtual Environment

To avoid dependency conflicts with other packages in your system, it's **strongly recommended** to use a virtual environment.

### Create Virtual Environment

```bash
# Using venv (Python 3.3+)
python -m venv venv

# Or using conda
conda create -n vision_yolo python=3.9
conda activate vision_yolo
```

### Activate Virtual Environment

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate

# With conda
conda activate vision_yolo
```

### Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install minimal dependencies (if you have conflicts)
pip install -r requirements-minimal.txt
```

## Troubleshooting Dependency Conflicts

If you see dependency conflict warnings (like the ones shown in the terminal), they are usually **warnings, not errors**. The YOLO-World detector should still work fine.

However, if you encounter actual runtime errors, try:

### Option 1: Use Virtual Environment (Recommended)
This isolates the project dependencies from your system packages.

### Option 2: Update Conflicting Packages
If you need to keep other packages, you can try updating them:

```bash
# Update streamlit (if you use it)
pip install --upgrade streamlit

# Update other packages as needed
pip install --upgrade packaging rich matplotlib scipy seaborn
```

### Option 3: Install Specific Versions
If you need specific versions for compatibility:

```bash
pip install ultralytics==8.0.196 opencv-python==4.8.1.78 boxmot==10.0.0 torch==2.0.0 numpy==1.24.0
```

## Verify Installation

After installation, verify that everything works:

```bash
python -c "from ultralytics import YOLOWorld; print('✓ Ultralytics OK')"
python -c "import cv2; print('✓ OpenCV OK')"
python -c "from boxmot import BotSort; print('✓ BoxMot OK')"
python -c "import torch; print('✓ PyTorch OK')"
```

## Common Issues

### Issue: `boxmot` installation fails
**Solution**: Make sure you have the latest pip and setuptools:
```bash
pip install --upgrade pip setuptools wheel
pip install boxmot
```

### Issue: CUDA/GPU not detected
**Solution**: Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: OpenCV import errors
**Solution**: Reinstall opencv-python:
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python
```

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional but recommended for faster processing (CUDA-compatible)

## Next Steps

After successful installation, you can:

1. Test with a sample video:
   ```bash
   python main.py -i your_video.mp4 -o output.mp4
   ```

2. Read the [README.md](README.md) for usage examples

3. Check [FEATURES.md](FEATURES.md) for advanced features


