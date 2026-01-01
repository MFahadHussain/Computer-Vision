"""Multi-feed video processor with click-to-track functionality."""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import time
from pathlib import Path

from video.video_reader import ThreadSafeVideoReader
from video.video_writer import ThreadSafeVideoWriter
from detection.detector import YOLOWorldROIDetector
from tracking.object_tracker import ObjectSpecificTracker
from ui.object_selector import ObjectSelector
from models.roi_manager import ROIManager

logger = logging.getLogger(__name__)


class MultiFeedProcessor:
    """Processor for handling multiple video feeds with click-to-track."""
    
    def __init__(self, detector: YOLOWorldROIDetector):
        """Initialize multi-feed processor.
        
        Args:
            detector: YOLO-World detector instance
        """
        self.detector = detector
        self.object_trackers: Dict[int, ObjectSpecificTracker] = {}
        self.target_object_bbox: Optional[Tuple[int, int, int, int]] = None
        self.target_object_class: Optional[int] = None
        
    def select_target_object(self, frame: np.ndarray, detections: List) -> bool:
        """Allow user to select target object for tracking.
        
        Args:
            frame: First frame from the first video
            detections: Initial detections from the frame
            
        Returns:
            True if object selected, False if cancelled
        """
        # Convert detections to format expected by ObjectSelector
        det_dicts = []
        for det in detections:
            if hasattr(det, 'bbox'):
                det_dicts.append({
                    'bbox': det.bbox,
                    'class_id': det.class_id,
                    'class_name': det.class_name,
                    'confidence': det.confidence
                })
        
        selector = ObjectSelector("Select Object to Track")
        selected_bbox = selector.select_object(frame, det_dicts)
        
        if selected_bbox:
            self.target_object_bbox = selected_bbox
            
            # Find the detection that best matches the selected bbox
            best_det = None
            best_iou = 0.0
            
            for det in detections:
                if hasattr(det, 'bbox'):
                    iou = self._calculate_iou(selected_bbox, det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_det = det
            
            if best_det:
                self.target_object_class = best_det.class_id
                logger.info(f"Target object selected: {best_det.class_name} at {selected_bbox}")
                return True
        
        return False
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def process_multiple_feeds(self, 
                              input_paths: List[str],
                              output_paths: List[str],
                              roi_config_path: Optional[str] = None,
                              save_roi_path: Optional[str] = None,
                              show_preview: bool = True) -> int:
        """Process multiple video feeds simultaneously.
        
        Args:
            input_paths: List of input video paths
            output_paths: List of output video paths (must match input_paths length)
            roi_config_path: Optional path to load ROI configuration
            save_roi_path: Optional path to save ROI configuration
            show_preview: Whether to show preview window
            
        Returns:
            0 on success, 1 on error
        """
        if len(input_paths) != len(output_paths):
            logger.error("Number of input and output paths must match")
            return 1
        
        num_feeds = len(input_paths)
        
        try:
            # Initialize video readers
            readers = []
            writers = []
            video_props = []
            
            print(f"\nInitializing {num_feeds} video feeds...")
            for i, input_path in enumerate(input_paths):
                reader = ThreadSafeVideoReader(input_path)
                readers.append(reader)
                
                video_props.append({
                    'fps': reader.fps,
                    'width': reader.width,
                    'height': reader.height,
                    'total_frames': reader.total_frames
                })
                
                print(f"  Feed {i+1}: {Path(input_path).name} - {reader.width}x{reader.height} @ {reader.fps:.2f} fps")
            
            # Read first frames for ROI selection
            first_frames = []
            for reader in readers:
                ret, frame = reader.read_first_frame()
                if not ret or frame is None:
                    raise ValueError(f"Cannot read first frame from {reader.video_path}")
                first_frames.append(frame)
            
            # Select ROI (use first frame)
            if roi_config_path and Path(roi_config_path).exists():
                print(f"\nLoading ROI configuration from {roi_config_path}")
                roi_config = ROIManager.load_roi_config(roi_config_path)
                if roi_config:
                    self.detector.roi = roi_config
                    print(f"Loaded {len(roi_config.polygons)} ROI(s)")
            else:
                print("\nSelect ROI on first feed (will be applied to all feeds)...")
                self.detector.select_roi(first_frames[0])
                
                # Save ROI if requested
                if save_roi_path:
                    ROIManager.save_roi_config(self.detector.roi, save_roi_path)
                    print(f"ROI configuration saved to {save_roi_path}")
            
            # Get initial detections for object selection
            print("\nGetting initial detections for object selection...")
            cropped_frames = self.detector.crop_frame(first_frames[0])
            initial_detections = self.detector.detect_and_track(cropped_frames)
            
            # Select target object
            print("\nSelect the object you want to track across all feeds...")
            if not self.select_target_object(first_frames[0], initial_detections):
                print("No object selected. Exiting.")
                return 1
            
            # Initialize video writers
            print(f"\nInitializing output videos...")
            for i, (output_path, props) in enumerate(zip(output_paths, video_props)):
                writer = ThreadSafeVideoWriter(
                    output_path, props['fps'], props['width'], props['height']
                )
                writers.append(writer)
                print(f"  Output {i+1}: {Path(output_path).name}")
            
            # Initialize object trackers for each feed
            for i in range(num_feeds):
                tracker = ObjectSpecificTracker(
                    target_bbox=self.target_object_bbox,
                    target_class_id=self.target_object_class
                )
                self.object_trackers[i] = tracker
            
            # Processing loop
            frame_count = 0
            start_time = time.time()
            fps_display = 0.0
            preview_fps = 10  # Target FPS for preview updates (lower = better performance)
            preview_frame_counter = 0
            preview_cache = None  # Cache for preview dimensions
            
            # Calculate frame skip based on target preview FPS
            # If video is 30fps and we want 10fps preview, skip every 3rd frame
            min_fps = min(r.fps for r in readers) if readers else 30
            preview_frame_skip = max(1, int(min_fps / preview_fps)) if min_fps > 0 else 2
            
            if show_preview:
                # Create multi-feed preview window
                cv2.namedWindow("Multi-Feed Tracking", cv2.WINDOW_NORMAL)
            
            print("\nStarting multi-feed processing...")
            print("Press 'q' to quit, ESC to stop")
            print("Note: Preview updates every {} frames (~{} FPS) for better performance\n".format(
                preview_frame_skip, preview_fps))
            
            # Start all readers and writers
            for reader, writer in zip(readers, writers):
                reader.start()
                writer.start()
            
            try:
                while True:
                    # Read frames from all feeds
                    frames = []
                    frame_indices = []
                    all_done = False
                    
                    for i, reader in enumerate(readers):
                        ret, frame_idx, frame = reader.read()
                        if not ret or frame is None:
                            all_done = True
                            break
                        frames.append(frame)
                        frame_indices.append(frame_idx)
                    
                    if all_done:
                        break
                    
                    # Process each feed
                    all_annotated_frames = []
                    target_detections = []
                    
                    for i, frame in enumerate(frames):
                        # Crop to ROI
                        cropped_frames = self.detector.crop_frame(frame)
                        
                        # Detect and track
                        detections = self.detector.detect_and_track(cropped_frames)
                        
                        # Track target object specifically
                        target_tracker = self.object_trackers[i]
                        target_det = target_tracker.update(
                            [{'bbox': d.bbox, 'confidence': d.confidence, 'class_id': d.class_id}
                             for d in detections],
                            frame
                        )
                        
                        if target_det:
                            target_detections.append(target_det)
                            # Highlight target object
                            detections.append(target_det)
                        
                        # Draw annotations
                        annotated = self.detector.draw_annotations(frame, detections, draw_roi=(i == 0))
                        
                        # Add feed label
                        cv2.putText(annotated, f"Feed {i+1}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        
                        # Add target tracking status
                        if target_det:
                            status = f"TARGET TRACKED: ID {target_det.track_id}"
                            cv2.putText(annotated, status, (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            status = "TARGET LOST"
                            cv2.putText(annotated, status, (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        all_annotated_frames.append(annotated)
                        
                        # Write to output
                        writers[i].write(annotated)
                    
                    # Create multi-feed preview (skip frames for better performance)
                    preview_frame_counter += 1
                    if show_preview and all_annotated_frames and (preview_frame_counter % preview_frame_skip == 0):
                        # Arrange feeds in a grid (use cached dimensions if available)
                        preview = self._create_multi_feed_preview(all_annotated_frames, target_detections, preview_cache)
                        
                        # Cache preview dimensions for next time
                        if preview_cache is None:
                            preview_cache = preview.shape[:2]
                        
                        # Resize for display (only if needed)
                        h, w = preview.shape[:2]
                        max_display = 1280  # Reduced from 1920 for better performance
                        if max(h, w) > max_display:
                            scale = max_display / max(h, w)
                            preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                        
                        cv2.imshow("Multi-Feed Tracking", preview)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            print("\n\nProcessing interrupted by user.")
                            break
                    
                    # Update statistics
                    frame_count += 1
                    if frame_count % 10 == 0:
                        elapsed = time.time() - start_time
                        fps_display = frame_count / elapsed
                        print(f"\rProgress: FPS: {fps_display:.1f} | "
                              f"Target tracked in {len(target_detections)}/{num_feeds} feeds", 
                              end="", flush=True)
            finally:
                # Stop all readers and writers
                for reader, writer in zip(readers, writers):
                    reader.stop()
                    writer.stop()
            
            if show_preview:
                cv2.destroyAllWindows()
            
            # Final statistics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            print(f"\n\n{'='*60}")
            print("MULTI-FEED PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Processed frames: {frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Number of feeds: {num_feeds}")
            for i, output_path in enumerate(output_paths):
                print(f"  Output {i+1}: {output_path}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"Error processing multiple feeds: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 1
        
        return 0
    
    def _create_multi_feed_preview(self, frames: List[np.ndarray], 
                                   target_detections: List,
                                   cache: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Create a grid preview of multiple feeds with different resolutions.
        
        Args:
            frames: List of frames to display
            target_detections: List of target detections (not used but kept for compatibility)
            cache: Cached dimensions (height, width) to avoid recalculating
        """
        import cv2
        
        num_feeds = len(frames)
        
        if num_feeds == 1:
            return frames[0]
        
        # Use cached dimensions if available, otherwise calculate
        if cache:
            common_h, common_w = cache
        else:
            # Find the maximum dimensions to use as base size
            max_h = max(f.shape[0] for f in frames)
            max_w = max(f.shape[1] for f in frames)
            
            # Use a common size for all frames (limit to reasonable size for performance)
            # Cap at 720p for preview to improve performance
            common_h = min(max_h, 720)
            common_w = min(max_w, 1280)
        
        # Resize all frames to common size (use faster interpolation for preview)
        resized_frames = []
        for frame in frames:
            if frame.shape[:2] != (common_h, common_w):
                # Use INTER_AREA for downscaling (faster and better quality)
                resized_frame = cv2.resize(frame, (common_w, common_h), 
                                         interpolation=cv2.INTER_AREA if 
                                         (frame.shape[0] > common_h or frame.shape[1] > common_w) 
                                         else cv2.INTER_LINEAR)
                resized_frames.append(resized_frame)
            else:
                resized_frames.append(frame)
        
        # Arrange in grid (2 columns max)
        cols = min(2, num_feeds)
        rows = (num_feeds + cols - 1) // cols
        
        # Create grid canvas
        grid_h = rows * common_h
        grid_w = cols * common_w
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for i, frame in enumerate(resized_frames):
            row = i // cols
            col = i % cols
            y_start = row * common_h
            x_start = col * common_w
            grid[y_start:y_start+common_h, x_start:x_start+common_w] = frame
        
        return grid

