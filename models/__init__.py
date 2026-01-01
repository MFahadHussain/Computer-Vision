"""Data models for YOLO-World V2 Video Detector."""

from .data_models import PolygonROI, ROIConfig, Detection
from .roi_manager import ROIManager

__all__ = ['PolygonROI', 'ROIConfig', 'Detection', 'ROIManager']



