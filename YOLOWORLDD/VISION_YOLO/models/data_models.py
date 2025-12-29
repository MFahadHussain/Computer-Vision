"""Data models for ROI configuration and detections."""

from typing import Tuple, Optional, List
from dataclasses import dataclass, field
import numpy as np
import cv2


@dataclass
class PolygonROI:
    """Configuration for a single Polygon Region of Interest."""
    points: List[Tuple[int, int]] = field(default_factory=list)
    min_x: int = 0
    min_y: int = 0
    width: int = 0
    height: int = 0
    mask: Optional[np.ndarray] = None
    id: int = 0
    
    def is_valid(self) -> bool:
        """Check if ROI has valid dimensions."""
        return len(self.points) >= 3 and self.width > 0 and self.height > 0
    
    def compute_bounds(self):
        """Compute bounding box and mask from polygon points."""
        if not self.points:
            return
            
        # Calculate bounding box
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        self.min_x = min(x_coords)
        self.min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)
        self.width = max_x - self.min_x
        self.height = max_y - self.min_y
        
        # Create mask
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Adjust points to be relative to the top-left of the bounding box
        adjusted_points = [(p[0] - self.min_x, p[1] - self.min_y) for p in self.points]
        
        # Fill polygon in the mask
        cv2.fillPoly(self.mask, [np.array(adjusted_points, dtype=np.int32)], 255)


@dataclass
class ROIConfig:
    """Configuration for multiple Regions of Interest."""
    polygons: List[PolygonROI] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if at least one ROI has valid dimensions."""
        return len(self.polygons) > 0 and any(p.is_valid() for p in self.polygons)
    
    def add_polygon(self, polygon: PolygonROI):
        """Add a polygon ROI to the configuration."""
        self.polygons.append(polygon)
    
    def remove_polygon(self, polygon_id: int):
        """Remove a polygon ROI by ID."""
        self.polygons = [p for p in self.polygons if p.id != polygon_id]
    
    def compute_bounds(self):
        """Compute bounds for all polygons."""
        for polygon in self.polygons:
            polygon.compute_bounds()


@dataclass
class Detection:
    """Represents a single detection with global coordinates."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    roi_id: Optional[int] = None

