"""BoT-SORT wrapper using boxmot library for integration with YOLO-World.

This wrapper uses the BoT-SORT implementation from the boxmot library,
which provides a modern, well-maintained tracking solution with re-identification.
"""

import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import logging
import traceback

logger = logging.getLogger(__name__)

# Try to import BoT-SORT from boxmot library
try:
    from boxmot import BotSort
    import torch
    BOTSORT_AVAILABLE = True
    logger.info("BoT-SORT imported successfully from boxmot library")
except ImportError:
    logger.warning("boxmot not available. Install with: pip install boxmot")
    BOTSORT_AVAILABLE = False


class BoTSORTWrapper:
    """BoT-SORT wrapper using boxmot library for integration with YOLO-World.
    
    This wrapper uses the BoT-SORT implementation from the boxmot library,
    which provides a modern, well-maintained tracking solution with re-identification.
    """
    
    def __init__(self, tracker_args: Optional[Dict] = None):
        """Initialize BoT-SORT tracker with optional configuration.
        
        Args:
            tracker_args: Optional dictionary with tracker parameters:
                - det_thresh: Detection confidence threshold (default: 0.25)
                - max_age: Maximum frames to keep lost tracks (default: 30)
                - frame_rate: Video frame rate (default: 30)
        """
        if not BOTSORT_AVAILABLE:
            raise ImportError("BoT-SORT is not available. Install with: pip install boxmot")
        
        # Default configuration for boxmot's BoT-SORT
        default_args = {
            'det_thresh': 0.25,
            'max_age': 30,
            'frame_rate': 30,
        }
        
        # Update with user-provided config
        if tracker_args:
            default_args.update(tracker_args)
        
        self.tracker_args = default_args
        self.tracker = None
        self.frame_count = 0
        
        # Determine device
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.half = torch.cuda.is_available()
        
        # ReID weights path
        self.reid_weights = Path("osnet_x0_25_msmt17.pt")
        
        try:
            # Initialize BoT-SORT tracker from boxmot with ReID
            self.tracker = BotSort(
                reid_weights=self.reid_weights,
                device=self.device,
                half=self.half,
                det_thresh=self.tracker_args.get('det_thresh', 0.25),
                max_age=self.tracker_args.get('max_age', 30),
                frame_rate=self.tracker_args.get('frame_rate', 30),
                cmc_method="sof",  # Sparse Optical Flow - faster
                with_reid=True,  # Enable ReID
            )
            logger.info("BoT-SORT tracker (boxmot) with ReID initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BoT-SORT tracker: {str(e)}")
            raise
    
    def update(self, detections: List[Dict], frame: Optional[np.ndarray] = None) -> List[Dict]:
        """Update tracker with new detections and return tracked objects.
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'confidence', 'class_id'
            frame: Frame for ReID features (required by boxmot)
            
        Returns:
            List of tracked object dictionaries with 'bbox', 'track_id', 'class_id', 'confidence'
        """
        if self.tracker is None:
            return []
        
        try:
            if len(detections) == 0:
                # Empty detections array
                dets = np.empty((0, 6), dtype=float)
            else:
                # Convert detections to boxmot format: [x1, y1, x2, y2, conf, cls]
                dets = np.array([
                    [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 
                     d['confidence'], d.get('class_id', 0)]
                    for d in detections
                ], dtype=float)
            
            # Update tracker (boxmot requires the frame for ReID)
            tracked = self.tracker.update(dets, frame)
            
            # Convert tracked detections back to our format
            # boxmot returns: [x1, y1, x2, y2, track_id, conf, cls, ...]
            tracked_objects = []
            for t in tracked:
                try:
                    x1 = float(t[0])
                    y1 = float(t[1])
                    x2 = float(t[2])
                    y2 = float(t[3])
                    track_id = int(t[4])
                    conf = float(t[5]) if len(t) > 5 else 0.5
                    cls_id = int(t[6]) if len(t) > 6 else 0
                    
                    tracked_objects.append({
                        'bbox': (x1, y1, x2, y2),
                        'track_id': track_id,
                        'class_id': cls_id,
                        'confidence': conf
                    })
                except Exception as e:
                    logger.warning(f"Error parsing track: {e}")
                    continue
            
            self.frame_count += 1
            return tracked_objects
            
        except Exception as e:
            logger.error(f"Error updating BoT-SORT tracker: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def reset(self):
        """Reset the tracker state."""
        self.frame_count = 0
        # Reinitialize tracker with ReID
        try:
            self.tracker = BotSort(
                reid_weights=self.reid_weights,
                device=self.device,
                half=self.half,
                det_thresh=self.tracker_args.get('det_thresh', 0.25),
                max_age=self.tracker_args.get('max_age', 30),
                frame_rate=self.tracker_args.get('frame_rate', 30),
                cmc_method="sof",
                with_reid=True,
            )
        except Exception as e:
            logger.error(f"Error resetting tracker: {e}")

