"""Model file checking and cleaning utilities."""

import os
import logging

logger = logging.getLogger(__name__)


def check_and_clean_model_files():
    """
    Checks for model files smaller than 1MB, which usually indicates a corrupted
    or interrupted download. Deletes them to prevent PytorchStreamReader errors.
    """
    # Common YOLO-World V2 model names
    model_files = [
        'yolov8l-worldv2.pt',
        'yolov8s-worldv2.pt',
        'yolov8m-worldv2.pt',
        'yolov8x-worldv2.pt'
    ]
    
    for model_name in model_files:
        if os.path.exists(model_name):
            try:
                file_size = os.path.getsize(model_name)
                if file_size < 1024 * 1024:  # Less than 1MB
                    logger.warning(f"Found potentially corrupted model file: {model_name} (Size: {file_size} bytes). Deleting...")
                    os.remove(model_name)
                    logger.info(f"Deleted {model_name}. Please re-download the model.")
            except Exception as e:
                logger.error(f"Error checking file {model_name}: {str(e)}")

