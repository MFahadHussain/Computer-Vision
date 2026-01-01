"""Detection modules for YOLO-World V2 Video Detector."""

from .detector import YOLOWorldROIDetector
from .multi_feed_processor import MultiFeedProcessor

__all__ = ['YOLOWorldROIDetector', 'MultiFeedProcessor']



