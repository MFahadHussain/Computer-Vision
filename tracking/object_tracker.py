"""Object-specific tracker for click-to-track functionality."""

import numpy as np
from typing import Tuple, Optional, List, Dict
import logging
import cv2

from tracking.tracker import BoTSORTWrapper
from models.data_models import Detection

logger = logging.getLogger(__name__)


class ObjectSpecificTracker:
    """Tracker for tracking a specific object across multiple feeds."""
    
    def __init__(self, target_bbox: Tuple[int, int, int, int], 
                 target_class_id: Optional[int] = None,
                 tracker_config: Optional[Dict] = None):
        """Initialize tracker for a specific object.
        
        Args:
            target_bbox: Initial bounding box of the target object (x1, y1, x2, y2)
            target_class_id: Optional class ID to filter detections
            tracker_config: Optional tracker configuration
        """
        self.target_bbox = target_bbox
        self.target_class_id = target_class_id
        self.tracker = BoTSORTWrapper(tracker_config)
        self.target_track_id: Optional[int] = None
        self.is_initialized = False
        
        # Calculate target center and size for matching
        x1, y1, x2, y2 = target_bbox
        self.target_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.target_size = ((x2 - x1) * (y2 - y1))
        
    def find_best_match(self, detections: List[Dict], frame: np.ndarray) -> Optional[Dict]:
        """Find the best matching detection for the target object.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame (for visual features if needed)
            
        Returns:
            Best matching detection or None
        """
        if not detections:
            return None
        
        # Filter by class if specified
        if self.target_class_id is not None:
            detections = [d for d in detections if d.get('class_id') == self.target_class_id]
        
        if not detections:
            return None
        
        # Calculate IoU and center distance for each detection
        best_match = None
        best_score = 0.0
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate IoU with target bbox
            iou = self._calculate_iou(self.target_bbox, bbox)
            
            # Calculate center distance
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            center_dist = np.sqrt(
                (center[0] - self.target_center[0])**2 + 
                (center[1] - self.target_center[1])**2
            )
            
            # Normalize center distance (use frame diagonal as reference)
            h, w = frame.shape[:2]
            max_dist = np.sqrt(w**2 + h**2)
            normalized_dist = center_dist / max_dist if max_dist > 0 else 1.0
            
            # Combined score (IoU weighted more)
            score = 0.7 * iou + 0.3 * (1.0 - normalized_dist)
            
            if score > best_score:
                best_score = score
                best_match = det
        
        return best_match if best_score > 0.3 else None  # Minimum threshold
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> Optional[Detection]:
        """Update tracker and return tracked target object.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame
            
        Returns:
            Detection object for the tracked target, or None if not found
        """
        if not self.is_initialized:
            # First frame - find best match and initialize tracking
            best_match = self.find_best_match(detections, frame)
            if best_match:
                # Initialize with the best match
                init_detections = [best_match]
                tracked = self.tracker.update(init_detections, frame)
                if tracked:
                    self.target_track_id = tracked[0]['track_id']
                    self.is_initialized = True
                    # Update target bbox
                    self.target_bbox = tracked[0]['bbox']
                    x1, y1, x2, y2 = self.target_bbox
                    self.target_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    self.target_size = ((x2 - x1) * (y2 - y1))
                    
                    return Detection(
                        bbox=tuple(map(int, self.target_bbox)),
                        confidence=tracked[0]['confidence'],
                        class_id=tracked[0]['class_id'],
                        class_name=f"target_{self.target_track_id}",
                        track_id=self.target_track_id,
                        roi_id=None
                    )
            return None
        
        # Update with all detections
        tracked_objects = self.tracker.update(detections, frame)
        
        # Find the tracked object with our target track ID
        for obj in tracked_objects:
            if obj['track_id'] == self.target_track_id:
                # Update target bbox
                self.target_bbox = obj['bbox']
                x1, y1, x2, y2 = self.target_bbox
                self.target_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                self.target_size = ((x2 - x1) * (y2 - y1))
                
                return Detection(
                    bbox=tuple(map(int, obj['bbox'])),
                    confidence=obj['confidence'],
                    class_id=obj['class_id'],
                    class_name=f"target_{self.target_track_id}",
                    track_id=obj['track_id'],
                    roi_id=None
                )
        
        # Target not found in this frame
        return None


