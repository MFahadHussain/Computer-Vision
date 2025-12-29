"""Video I/O modules for YOLO-World V2 Video Detector."""

from .video_reader import ThreadSafeVideoReader
from .video_writer import ThreadSafeVideoWriter

__all__ = ['ThreadSafeVideoReader', 'ThreadSafeVideoWriter']

