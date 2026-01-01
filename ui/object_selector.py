"""Object selector for click-to-track functionality."""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ObjectSelector:
    """Interactive object selector for click-to-track."""
    
    def __init__(self, window_name: str = "Select Object to Track"):
        self.window_name = window_name
        self.selected_bbox: Optional[Tuple[int, int, int, int]] = None
        self.selected_point: Optional[Tuple[int, int]] = None
        self.drawing = False
        self.finished = False
        self.start_point: Optional[Tuple[int, int]] = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse event handler for object selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True
            self.selected_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Preview the selection box
                img = param.copy()
                if self.start_point:
                    cv2.rectangle(img, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                # Finalize the bounding box
                x1 = min(self.start_point[0], x)
                y1 = min(self.start_point[1], y)
                x2 = max(self.start_point[0], x)
                y2 = max(self.start_point[1], y)
                
                # Ensure minimum size
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.selected_bbox = (x1, y1, x2, y2)
                    self.finished = True
                else:
                    # Single click - create small box around point
                    self.selected_bbox = (x - 20, y - 20, x + 20, y + 20)
                    self.selected_point = (x, y)
                    self.finished = True
                self.drawing = False
    
    def select_object(self, frame: np.ndarray, detections: List[dict] = None) -> Optional[Tuple[int, int, int, int]]:
        """Allow user to select an object by clicking or drawing a box.
        
        Args:
            frame: Frame to display for selection
            detections: Optional list of detections to show (for clicking on detected objects)
            
        Returns:
            Bounding box (x1, y1, x2, y2) or None if cancelled
        """
        self.selected_bbox = None
        self.selected_point = None
        self.finished = False
        self.drawing = False
        
        display_frame = frame.copy()
        
        # Draw detections if provided
        if detections:
            for det in detections:
                if 'bbox' in det:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    if 'class_name' in det and 'confidence' in det:
                        label = f"{det['class_name']} {det['confidence']:.2f}"
                        cv2.putText(display_frame, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback, display_frame)
        
        instructions = [
            "OBJECT SELECTION",
            "Click and drag to draw a bounding box",
            "Or click on a detected object",
            "Press SPACE/ENTER to confirm selection",
            "Press ESC to cancel"
        ]
        
        while not self.finished:
            img = display_frame.copy()
            
            # Draw instructions
            for i, text in enumerate(instructions):
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(img, text, (10, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw current selection if drawing
            if self.drawing and self.start_point and self.selected_point:
                cv2.rectangle(img, self.start_point, self.selected_point, (0, 255, 0), 2)
            
            # Draw selected bbox if available
            if self.selected_bbox:
                x1, y1, x2, y2 = self.selected_bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, "SELECTED", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32 or key == 13:  # SPACE or ENTER
                if self.selected_bbox:
                    self.finished = True
            elif key == 27:  # ESC
                self.selected_bbox = None
                self.finished = True
        
        cv2.destroyWindow(self.window_name)
        return self.selected_bbox


