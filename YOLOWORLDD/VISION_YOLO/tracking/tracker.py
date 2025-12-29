"""Robust wrapper for BoT-SORT using supervision library."""

import numpy as np
from typing import Optional, List, Dict
import logging
import traceback

logger = logging.getLogger(__name__)

# Import Supervision with Fallbacks
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
    # 0.27.0 Specific Imports
    try:
        # Try importing BoTSORT (New capitalization in 0.27.0)
        from supervision import BoTSORT
        BOTSORT_AVAILABLE = True
        logger.info("BoT-SORT (supervision 0.27.0+) imported successfully")
    except ImportError:
        logger.warning("BoT-SORT not found in supervision 0.27.0+ format. Checking legacy paths.")
        try:
            from supervision.tracker.botsort.core import BoTSORT
            BOTSORT_AVAILABLE = True
        except ImportError:
            logger.warning("BoT-SORT not available. Install with: pip install supervision")
            BOTSORT_AVAILABLE = False

    # Import ByteTrack for fallback
    try:
        from supervision import ByteTrack
        BYTETRACK_AVAILABLE = True
    except ImportError:
        logger.warning("ByteTrack not available.")
        BYTETRACK_AVAILABLE = False

except ImportError:
    logger.warning("Supervision library not found. Install with: pip install supervision")
    SUPERVISION_AVAILABLE = False
    BOTSORT_AVAILABLE = False
    BYTETRACK_AVAILABLE = False


class BoTSORTWrapper:
    """
    Robust wrapper for BoT-SORT using supervision library.
    
    Features:
    - Supports Supervision 0.27.0+ parameter naming (track_high_thresh, track_buffer).
    - Fallback to ByteTrack if BoTSORT fails (missing lapx/cython-bbox).
    """
    
    def __init__(self, tracker_args: Optional[Dict] = None):
        """Initialize tracker with optional configuration."""
        
        # Default configuration for Supervision 0.27.0+ BoTSORT
        default_args = {
            'track_high_thresh': 0.25,    # Renamed from track_activation_threshold
            'track_buffer': 30,           # Renamed from lost_track_buffer
            'match_thresh': 0.8,
            'frame_rate': 30,
        }
        
        # Update with user-provided config
        if tracker_args:
            # Legacy parameter mapping for backwards compatibility with user input
            legacy_map = {
                'track_activation_threshold': 'track_high_thresh',
                'lost_track_buffer': 'track_buffer',
            }
            
            # Map legacy keys to new keys
            normalized_args = {}
            for key, value in tracker_args.items():
                if key in legacy_map:
                    normalized_args[legacy_map[key]] = value
                else:
                    normalized_args[key] = value
            
            default_args.update(normalized_args)
        
        self.tracker_args = default_args
        self.tracker = None
        self.tracker_type = "None"
        
        # Attempt to initialize BoTSORT
        if BOTSORT_AVAILABLE:
            try:
                logger.info("Initializing BoTSORT...")
                self.tracker = BoTSORT(
                    track_high_thresh=self.tracker_args.get('track_high_thresh', 0.25),
                    track_buffer=self.tracker_args.get('track_buffer', 30),
                    match_thresh=self.tracker_args.get('match_thresh', 0.8),
                    frame_rate=self.tracker_args.get('frame_rate', 30),
                )
                self.tracker_type = "BoTSORT"
                logger.info("BoTSORT tracker initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize BoTSORT: {str(e)}")
                logger.warning("This usually means missing C++ dependencies (lapx, cython-bbox).")
                self.tracker = None
        else:
            logger.warning("BoTSORT is not installed or not importable.")
        
        # Fallback to ByteTrack if BoTSORT failed
        if self.tracker is None:
            if BYTETRACK_AVAILABLE:
                try:
                    logger.info("Falling back to ByteTrack...")
                    # ByteTrack uses slightly different params in supervision 0.27.0
                    # but accepts generic args similar to BoTSORT
                    bt_args = {
                        'track_activation_threshold': self.tracker_args.get('track_high_thresh', 0.25),
                        'lost_track_buffer': self.tracker_args.get('track_buffer', 30),
                        'minimum_matching_threshold': self.tracker_args.get('match_thresh', 0.8),
                        'frame_rate': self.tracker_args.get('frame_rate', 30),
                    }
                    self.tracker = ByteTrack(**bt_args)
                    self.tracker_type = "ByteTrack"
                    logger.info("ByteTrack initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize ByteTrack: {str(e)}")
                    raise
    
    def update(self, detections: List[Dict], frame: Optional[np.ndarray] = None) -> List[Dict]:
        """Update tracker with new detections."""
        if self.tracker is None:
            return []
        
        try:
            if len(detections) == 0:
                # Create empty detections and update tracker
                empty_detections = sv.Detections.empty()
                # Supervision 0.27.0 update
                if hasattr(self.tracker, 'update_with_detections'):
                    self.tracker.update_with_detections(empty_detections)
                return []
            
            # Convert detections to supervision Detections format
            xyxy = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3]] 
                            for d in detections])
            confidence = np.array([d['confidence'] for d in detections])
            class_id = np.array([d['class_id'] for d in detections])
            
            # Create supervision Detections object
            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )
            
            # Update tracker and get tracked detections
            # Supervision 0.27.0 standard method
            tracked_detections = self.tracker.update_with_detections(sv_detections)
            
            # Convert tracked detections back to our format
            tracked_objects = []
            if tracked_detections.tracker_id is not None:
                for i in range(len(tracked_detections)):
                    x1, y1, x2, y2 = tracked_detections.xyxy[i]
                    
                    tracked_objects.append({
                        'bbox': (float(x1), float(y1), float(x2), float(y2)),
                        'track_id': int(tracked_detections.tracker_id[i]),
                        'class_id': int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else 0,
                        'confidence': float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 0.5
                    })
            
            return tracked_objects
            
        except Exception as e:
            logger.error(f"Error updating {self.tracker_type} tracker: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def reset(self):
        """Reset the tracker state."""
        if self.tracker is not None and hasattr(self.tracker, 'reset'):
            self.tracker.reset()

