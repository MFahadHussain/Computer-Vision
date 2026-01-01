"""YOLO-World V2 detector with enhanced error handling, polygon ROI processing, and BoT-SORT tracking."""

import cv2
import numpy as np
from ultralytics import YOLOWorld
from typing import Tuple, Optional, List
import logging
import traceback

from models.data_models import PolygonROI, ROIConfig, Detection
from models.roi_manager import ROIManager
from ui.polygon_selector import EnhancedPolygonSelector
from tracking.tracker import BoTSORTWrapper

logger = logging.getLogger(__name__)


class YOLOWorldROIDetector:
    """YOLO-World V2 detector with enhanced error handling, polygon ROI processing, and BoT-SORT tracking."""
    
    # Color palette for tracking visualization (BGR format)
    COLORS = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
        (0, 128, 255),  # Light Orange
        (128, 255, 0),  # Light Green
    ]
    
    def __init__(self, model_path: str = 'yolov8l-worldv2.pt', 
                 custom_classes: Optional[List[str]] = None,
                 confidence_threshold: float = 0.3,
                 use_botsort: bool = True,
                 tracker_config: Optional[dict] = None):
        """Initialize the YOLO-World detector with BoT-SORT tracking option."""
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.roi: Optional[ROIConfig] = None
        self.use_botsort = use_botsort
        self.tracker = None
        
        try:
            print(f"Loading YOLO-World model: {model_path}")
            self.model = YOLOWorld(model_path)
            
            # Set custom classes if provided (YOLO-World supports open-vocabulary)
            if custom_classes:
                print(f"Setting custom classes: {custom_classes}")
                self.model.set_classes(custom_classes)
                
        except Exception as e:
            logger.error(f"Error loading YOLO-World model: {str(e)}")
            logger.error("Please check that the model path is correct and the model file is valid.")
            logger.error("You can download a pre-trained model from the Ultralytics repository.")
            raise
        
        # Initialize tracker
        if self.use_botsort:
            try:
                self.tracker = BoTSORTWrapper(tracker_config)
                print("BoT-SORT tracker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize BoT-SORT tracker: {str(e)}")
                logger.warning("Falling back to default tracking")
                self.use_botsort = False
    
    def select_roi(self, frame: np.ndarray, load_path: Optional[str] = None, 
                   save_path: Optional[str] = None) -> ROIConfig:
        """Allow user to select and edit multiple polygon ROIs with enhanced UI.
        
        Args:
            frame: Frame to display for ROI selection
            load_path: Optional path to load existing ROI configuration
            save_path: Optional path to save ROI configuration after selection
        """
        # Try to load existing ROI configuration
        if load_path:
            roi_config = ROIManager.load_roi_config(load_path)
            if roi_config:
                self.roi = roi_config
                print(f"Loaded ROI configuration from {load_path}")
                print(f"  {len(roi_config.polygons)} polygon ROI(s) loaded")
                return self.roi
        
        print("\n" + "="*60)
        print("ENHANCED MULTIPLE POLYGON ROI SELECTION AND EDITING")
        print("="*60)
        print("Click to add points to define your regions of interest.")
        print("Enhanced features:")
        print("  - Undo/Redo functionality (z/x)")
        print("  - Grid snap option (g)")
        print("  - Point insertion/deletion (i/r)")
        print("  - Interactive help overlay (h)")
        print("="*60 + "\n")
        
        # Create a window for ROI selection
        window_name = "Enhanced Polygon ROI Editor"
        
        # Resize window if frame is too large
        max_display_size = 1280
        h, w = frame.shape[:2]
        if max(h, w) > max_display_size:
            scale = max_display_size / max(h, w)
            display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            display_frame = frame.copy()
        
        # Let user select and edit multiple polygon ROIs with enhanced UI
        selector = EnhancedPolygonSelector(window_name, snap_to_grid=False, grid_size=10)
        polygons = selector.select_polygons(display_frame)
        
        # Create ROIConfig
        self.roi = ROIConfig()
        
        # If no valid selection, use full frame
        if len(polygons) == 0:
            print("No valid polygon ROIs selected. Using full frame.")
            h_full, w_full = frame.shape[:2]
            # Create a rectangular polygon for the full frame
            full_frame_polygon = PolygonROI(
                points=[(0, 0), (w_full, 0), (w_full, h_full), (0, h_full)],
                min_x=0,
                min_y=0,
                width=w_full,
                height=h_full,
                id=0
            )
            full_frame_polygon.compute_bounds()
            self.roi.add_polygon(full_frame_polygon)
        else:
            # Scale points back to original frame size if it was resized
            if max(h, w) > max_display_size:
                scale = max(h, w) / max_display_size
                for polygon in polygons:
                    polygon.points = [(int(p[0] * scale), int(p[1] * scale)) for p in polygon.points]
                    polygon.compute_bounds()
                    self.roi.add_polygon(polygon)
            else:
                for polygon in polygons:
                    polygon.compute_bounds()
                    self.roi.add_polygon(polygon)
            
            print(f"Selected {len(self.roi.polygons)} polygon ROIs")
            for i, polygon in enumerate(self.roi.polygons):
                print(f"  ROI {i+1}: bounding box - x={polygon.min_x}, y={polygon.min_y}, "
                      f"w={polygon.width}, h={polygon.height}")
        
        self.roi.compute_bounds()
        
        # Save ROI configuration if requested
        if save_path:
            if ROIManager.save_roi_config(self.roi, save_path):
                print(f"ROI configuration saved to {save_path}")
        
        return self.roi
    
    def crop_frame(self, frame: np.ndarray) -> List[np.ndarray]:
        """Crop the frame using the stored polygon ROIs with error handling."""
        if self.roi is None:
            raise ValueError("ROI not set. Call select_roi() first.")
        
        cropped_frames = []
        
        try:
            for polygon in self.roi.polygons:
                # Extract the bounding box of the ROI
                x, y, w, h = polygon.min_x, polygon.min_y, polygon.width, polygon.height
                
                # Validate coordinates
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    logger.warning(f"Invalid ROI dimensions: x={x}, y={y}, w={w}, h={h}")
                    continue
                
                # Check if ROI is within frame bounds
                frame_h, frame_w = frame.shape[:2]
                if x + w > frame_w or y + h > frame_h:
                    logger.warning(f"ROI extends beyond frame bounds: frame={frame_w}x{frame_h}, ROI={x+w}x{y+h}")
                    # Adjust ROI to fit within frame
                    w = min(w, frame_w - x)
                    h = min(h, frame_h - y)
                
                if w <= 0 or h <= 0:
                    logger.warning(f"Adjusted ROI has invalid dimensions: w={w}, h={h}")
                    continue
                
                cropped = frame[y:y+h, x:x+w].copy()
                
                # Apply the polygon mask
                if polygon.mask is not None and polygon.mask.shape[:2] == (h, w):
                    # Create a 3-channel mask
                    mask_3d = cv2.cvtColor(polygon.mask, cv2.COLOR_GRAY2BGR)
                    # Apply mask to the cropped image
                    cropped = cv2.bitwise_and(cropped, mask_3d)
                
                cropped_frames.append(cropped)
        except Exception as e:
            logger.error(f"Error cropping frame: {str(e)}")
            # Return empty list if cropping fails
            return []
        
        return cropped_frames
    
    def transform_to_global_coords(self, bbox: Tuple[float, float, float, float],
                                   roi_id: int) -> Tuple[int, int, int, int]:
        """Transform bounding box from crop-space to global frame-space with error handling."""
        if self.roi is None:
            raise ValueError("ROI not set. Call select_roi() first.")
        
        if roi_id >= len(self.roi.polygons):
            raise ValueError(f"Invalid ROI ID: {roi_id}")
        
        try:
            polygon = self.roi.polygons[roi_id]
            crop_x1, crop_y1, crop_x2, crop_y2 = bbox
            
            # Apply ROI offset transformation
            global_x1 = int(crop_x1 + polygon.min_x)
            global_y1 = int(crop_y1 + polygon.min_y)
            global_x2 = int(crop_x2 + polygon.min_x)
            global_y2 = int(crop_y2 + polygon.min_y)
            
            return (global_x1, global_y1, global_x2, global_y2)
        except Exception as e:
            logger.error(f"Error transforming coordinates: {str(e)}")
            # Return original bbox if transformation fails
            return tuple(map(int, bbox))
    
    def detect_and_track(self, cropped_frames: List[np.ndarray]) -> List[Detection]:
        """Run YOLO-World detection with tracking on cropped frames."""
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        all_detections = []
        
        try:
            for roi_id, cropped_frame in enumerate(cropped_frames):
                if cropped_frame is None or cropped_frame.size == 0:
                    logger.warning(f"Empty cropped frame for ROI {roi_id}")
                    continue
                
                # Run YOLO-World detection with optimizations
                # Limit image size for faster processing (640 is a good balance)
                results = self.model(
                    cropped_frame,
                    conf=self.confidence_threshold,
                    verbose=False,
                    imgsz=640  # Fixed size for consistent performance
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        # Prepare detections for tracking
                        detections = []
                        for box in boxes:
                            try:
                                # Get bounding box in crop-space
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                
                                # Get other detection properties
                                conf = float(box.conf[0].cpu().numpy())
                                cls_id = int(box.cls[0].cpu().numpy())
                                
                                detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': conf,
                                    'class_id': cls_id
                                })
                            except Exception as e:
                                logger.warning(f"Error processing detection: {str(e)}")
                                continue
                        
                        # Apply tracking
                        if self.use_botsort and self.tracker:
                            tracked_objects = self.tracker.update(detections, cropped_frame)
                            
                            # Convert tracked objects to our Detection format
                            for obj in tracked_objects:
                                try:
                                    # Transform to global coordinates
                                    global_bbox = self.transform_to_global_coords(obj['bbox'], roi_id)
                                    
                                    cls_name = self.model.names.get(obj['class_id'], f"class_{obj['class_id']}")
                                    
                                    detection = Detection(
                                        bbox=global_bbox,
                                        confidence=obj['confidence'],
                                        class_id=obj['class_id'],
                                        class_name=cls_name,
                                        track_id=obj['track_id'],
                                        roi_id=roi_id
                                    )
                                    all_detections.append(detection)
                                except Exception as e:
                                    logger.warning(f"Error processing tracked object: {str(e)}")
                                    continue
                        else:
                            # Use default tracking (no persistent IDs)
                            for det in detections:
                                try:
                                    # Transform to global coordinates
                                    global_bbox = self.transform_to_global_coords(det['bbox'], roi_id)
                                    
                                    cls_name = self.model.names.get(det['class_id'], f"class_{det['class_id']}")
                                    
                                    detection = Detection(
                                        bbox=global_bbox,
                                        confidence=det['confidence'],
                                        class_id=det['class_id'],
                                        class_name=cls_name,
                                        track_id=None,
                                        roi_id=roi_id
                                    )
                                    all_detections.append(detection)
                                except Exception as e:
                                    logger.warning(f"Error processing detection: {str(e)}")
                                    continue
        except Exception as e:
            logger.error(f"Error in detection and tracking: {str(e)}")
        
        return all_detections
    
    def draw_annotations(self, frame: np.ndarray, detections: List[Detection],
                         draw_roi: bool = True) -> np.ndarray:
        """Draw bounding boxes and annotations on the frame with error handling."""
        try:
            annotated = frame.copy()
            
            # Draw ROI boundaries
            if draw_roi and self.roi is not None:
                # Draw polygon ROIs
                for i, polygon in enumerate(self.roi.polygons):
                    if len(polygon.points) >= 3:
                        cv2.polylines(
                            annotated,
                            [np.array(polygon.points, dtype=np.int32)],
                            True,  # Closed polygon
                            (0, 255, 255),  # Yellow
                            2
                        )
                        
                        # Add ROI label
                        centroid_x = sum(p[0] for p in polygon.points) // len(polygon.points)
                        centroid_y = sum(p[1] for p in polygon.points) // len(polygon.points)
                        cv2.putText(
                            annotated, f"ROI {i+1}",
                            (centroid_x - 20, centroid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                        )
            
            # Draw detections
            for det in detections:
                try:
                    x1, y1, x2, y2 = det.bbox
                    
                    # Select color based on track ID
                    if det.track_id is not None:
                        color = self.COLORS[det.track_id % len(self.COLORS)]
                    else:
                        color = self.COLORS[det.class_id % len(self.COLORS)]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label
                    if det.track_id is not None:
                        label = f"ID:{det.track_id} {det.class_name} {det.confidence:.2f}"
                    else:
                        label = f"{det.class_name} {det.confidence:.2f}"
                    
                    # Add ROI ID to label if multiple ROIs
                    if self.roi and len(self.roi.polygons) > 1:
                        label = f"[ROI{det.roi_id+1}] {label}"
                    
                    # Draw label background
                    (label_w, label_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        annotated,
                        (x1, y1 - label_h - 10),
                        (x1 + label_w, y1),
                        color, -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                except Exception as e:
                    logger.warning(f"Error drawing detection: {str(e)}")
                    continue
            
            return annotated
        except Exception as e:
            logger.error(f"Error drawing annotations: {str(e)}")
            return frame
    
    def process_video(self, input_path: str, output_path: str,
                      show_preview: bool = True,
                      custom_classes: Optional[List[str]] = None):
        """Process an entire video file with enhanced error handling."""
        from video.video_reader import ThreadSafeVideoReader
        from video.video_writer import ThreadSafeVideoWriter
        import time
        
        try:
            # Update classes if specified
            if custom_classes and self.model:
                print(f"Setting detection classes: {custom_classes}")
                self.model.set_classes(custom_classes)
            
            # Initialize video reader
            print(f"\nOpening video: {input_path}")
            reader = ThreadSafeVideoReader(input_path)
            
            print(f"Video properties:")
            print(f"  - Resolution: {reader.width}x{reader.height}")
            print(f"  - FPS: {reader.fps:.2f}")
            print(f"  - Total frames: {reader.total_frames}")
            
            # Read first frame for ROI selection
            ret, first_frame = reader.read_first_frame()
            if not ret or first_frame is None:
                raise ValueError("Cannot read first frame from video")
            
            # Let user select and edit multiple polygon ROIs
            self.select_roi(first_frame)
            
            # Initialize video writer
            print(f"\nInitializing output: {output_path}")
            writer = ThreadSafeVideoWriter(
                output_path, reader.fps, reader.width, reader.height
            )
            
            # Processing statistics
            frame_count = 0
            start_time = time.time()
            fps_display = 0.0
            
            if show_preview:
                cv2.namedWindow("YOLO-World Detection", cv2.WINDOW_NORMAL)
            
            print("\nStarting video processing...")
            print(f"Tracking method: {'BoT-SORT' if self.use_botsort else 'Default'}")
            print("Press 'q' to quit preview, ESC to stop processing\n")
            
            with reader, writer:
                while True:
                    ret, frame_idx, frame = reader.read()
                    
                    if not ret or frame is None:
                        break
                    
                    # Crop frame to ROIs
                    cropped_frames = self.crop_frame(frame)
                    
                    # Run detection and tracking
                    detections = self.detect_and_track(cropped_frames)
                    
                    # Draw annotations on original frame
                    annotated = self.draw_annotations(frame, detections)
                    
                    # Add frame counter and FPS
                    info_text = f"Frame: {frame_idx}/{reader.total_frames} | FPS: {fps_display:.1f} | Objects: {len(detections)}"
                    cv2.putText(
                        annotated, info_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )
                    
                    # Write to output
                    writer.write(annotated)
                    
                    # Update statistics
                    frame_count += 1
                    if frame_count % 10 == 0:
                        elapsed = time.time() - start_time
                        fps_display = frame_count / elapsed
                        progress = (frame_idx / reader.total_frames) * 100
                        print(f"\rProgress: {progress:.1f}% | FPS: {fps_display:.1f} | "
                              f"Detections: {len(detections)}", end="", flush=True)
                    
                    # Show preview
                    if show_preview:
                        # Resize for display if needed
                        display_frame = annotated
                        max_display = 1280
                        h, w = display_frame.shape[:2]
                        if max(h, w) > max_display:
                            scale = max_display / max(h, w)
                            display_frame = cv2.resize(
                                display_frame, None, fx=scale, fy=scale
                            )
                        
                        cv2.imshow("YOLO-World Detection", display_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            print("\n\nProcessing interrupted by user.")
                            break
            
            if show_preview:
                cv2.destroyAllWindows()
            
            # Final statistics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            print(f"\n\n{'='*60}")
            print("PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Processed frames: {frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Tracking method: {'BoT-SORT' if self.use_botsort else 'Default'}")
            print(f"Output saved to: {output_path}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"\nError processing video: {str(e)}")
            print("Please check the input video file and try again.")
            return 1
        
        return 0

