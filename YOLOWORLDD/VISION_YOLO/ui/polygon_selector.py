"""Enhanced polygon selector for ROI definition."""

import cv2
import numpy as np
from typing import List, Tuple
from models.data_models import PolygonROI


class EnhancedPolygonSelector:
    """Enhanced helper class for selecting and editing multiple polygon ROIs."""
    
    def __init__(self, window_name: str, snap_to_grid: bool = False, grid_size: int = 10):
        self.window_name = window_name
        self.polygons = []
        self.current_points = []
        self.drawing = False
        self.finished = False
        self.editing_mode = False
        self.selected_polygon_id = None
        self.selected_point_idx = None
        self.dragging_point = False
        self.snap_to_grid = snap_to_grid
        self.grid_size = grid_size
        self.history = []  # For undo/redo functionality
        self.history_idx = -1
        self.show_help = True
        self.help_alpha = 0.7  # Transparency of help overlay
        
    def save_state(self):
        """Save current state for undo functionality."""
        state = {
            'polygons': [PolygonROI(points=p.points.copy(), id=p.id) for p in self.polygons],
            'current_points': self.current_points.copy()
        }
        self.history = self.history[:self.history_idx+1]
        self.history.append(state)
        self.history_idx += 1
        
    def undo(self):
        """Undo last action."""
        if self.history_idx > 0:
            self.history_idx -= 1
            state = self.history[self.history_idx]
            self.polygons = [PolygonROI(points=p.points.copy(), id=p.id) for p in state['polygons']]
            self.current_points = state['current_points'].copy()
            return True
        return False
    
    def redo(self):
        """Redo last undone action."""
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            state = self.history[self.history_idx]
            self.polygons = [PolygonROI(points=p.points.copy(), id=p.id) for p in state['polygons']]
            self.current_points = state['current_points'].copy()
            return True
        return False
    
    def snap_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Snap point to grid if enabled."""
        if not self.snap_to_grid:
            return point
        x, y = point
        x = round(x / self.grid_size) * self.grid_size
        y = round(y / self.grid_size) * self.grid_size
        return (int(x), int(y))
    
    def find_closest_edge(self, polygon: PolygonROI, point: Tuple[int, int]) -> int:
        """Find the closest edge in a polygon to insert a new point."""
        if len(polygon.points) < 2:
            return 0
        
        min_dist = float('inf')
        insert_idx = 0
        
        for i in range(len(polygon.points)):
            p1 = polygon.points[i]
            p2 = polygon.points[(i+1) % len(polygon.points)]
            
            # Calculate distance from point to line segment
            line_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            point_vec = np.array([point[0] - p1[0], point[1] - p1[1]])
            line_len = np.linalg.norm(line_vec)
            
            if line_len > 0:
                line_unitvec = line_vec / line_len
                point_vec_scaled = point_vec / line_len
                
                t = np.dot(line_unitvec, point_vec_scaled)
                t = max(0.0, min(1.0, t))
                
                nearest = line_vec * t
                dist = np.linalg.norm(point_vec - nearest)
                
                if dist < min_dist:
                    min_dist = dist
                    insert_idx = i + 1
        
        return insert_idx
    
    def mouse_callback(self, event, x, y, flags, param):
        """Enhanced mouse event handler with better visual feedback."""
        # Snap point to grid if enabled
        x, y = self.snap_point((x, y))
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.editing_mode:
                # Check if clicking on a point
                if self.selected_polygon_id is not None:
                    polygon = self.polygons[self.selected_polygon_id]
                    for i, point in enumerate(polygon.points):
                        dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
                        if dist < 10:  # Threshold for point selection
                            self.selected_point_idx = i
                            self.dragging_point = True
                            self.save_state()
                            return
                
                # Check if clicking inside a polygon to select it
                for i, polygon in enumerate(self.polygons):
                    if cv2.pointPolygonTest(np.array(polygon.points, dtype=np.int32), (x, y), False) >= 0:
                        self.selected_polygon_id = i
                        self.selected_point_idx = None
                        return
                
                # Clicking outside any polygon deselects
                self.selected_polygon_id = None
                self.selected_point_idx = None
            else:
                # Adding points to new polygon
                self.current_points.append((x, y))
                self.drawing = True
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.editing_mode and self.dragging_point and self.selected_point_idx is not None:
                # Update the position of the selected point
                self.polygons[self.selected_polygon_id].points[self.selected_point_idx] = (x, y)
            elif self.drawing and not self.editing_mode:
                # Preview the line being drawn
                img = param.copy()
                if len(self.current_points) > 0:
                    cv2.line(img, self.current_points[-1], (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_point:
                self.dragging_point = False
            elif self.drawing and not self.editing_mode:
                self.drawing = False
                
    def draw_grid(self, img: np.ndarray):
        """Draw grid on image if snap_to_grid is enabled."""
        if not self.snap_to_grid:
            return
            
        h, w = img.shape[:2]
        grid_color = (50, 50, 50)  # Dark gray
        
        # Draw vertical lines
        for x in range(0, w, self.grid_size):
            cv2.line(img, (x, 0), (x, h), grid_color, 1)
            
        # Draw horizontal lines
        for y in range(0, h, self.grid_size):
            cv2.line(img, (0, y), (w, y), grid_color, 1)
    
    def draw_help_overlay(self, img: np.ndarray):
        """Draw help overlay on image."""
        if not self.show_help:
            return
            
        h, w = img.shape[:2]
        overlay = img.copy()
        
        # Create semi-transparent overlay
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.help_alpha, img, 1 - self.help_alpha, 0, img)
        
        # Prepare instructions based on mode
        if self.editing_mode:
            instructions = [
                "EDIT MODE",
                "Click on a polygon to select it",
                "Click and drag points to edit",
                "Press 'i' to insert a point",
                "Press 'd' to delete selected polygon",
                "Press 'r' to delete selected point",
                "Press 's' to exit edit mode",
                "Press 'h' to toggle help",
                "Press 'z' to undo, 'x' to redo",
                "Press 'g' to toggle grid snap",
                "Press ESC to cancel"
            ]
        else:
            instructions = [
                "SELECTION MODE",
                "Click to add points to the polygon",
                "Press SPACE or ENTER to finish current polygon",
                "Press 'n' to start a new polygon",
                "Press 'e' to enter edit mode",
                "Press 'f' to finish all polygons",
                "Press 'c' to cancel",
                "Press 'r' to reset current polygon",
                "Press 'h' to toggle help",
                "Press 'z' to undo, 'x' to redo",
                "Press 'g' to toggle grid snap"
            ]
        
        # Draw instructions
        for i, text in enumerate(instructions):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(img, text, (20, 30 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def select_polygons(self, frame: np.ndarray) -> List[PolygonROI]:
        """Enhanced polygon selection with better UI."""
        img = frame.copy()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback, img)
        
        # Initialize history
        self.save_state()
        
        while not self.finished:
            # Draw grid if enabled
            display_img = img.copy()
            self.draw_grid(display_img)
            
            # Draw completed polygons
            for i, polygon in enumerate(self.polygons):
                if len(polygon.points) > 1:
                    # Highlight selected polygon
                    color = (0, 255, 0) if i == self.selected_polygon_id else (0, 255, 255)
                    thickness = 3 if i == self.selected_polygon_id else 2
                    
                    cv2.polylines(display_img, [np.array(polygon.points, dtype=np.int32)], 
                                  True, color, thickness)
                
                # Draw points and their coordinates
                for j, point in enumerate(polygon.points):
                    # Highlight selected point
                    if i == self.selected_polygon_id and j == self.selected_point_idx:
                        cv2.circle(display_img, point, 8, (0, 0, 255), -1)
                    else:
                        cv2.circle(display_img, point, 5, (0, 0, 255), -1)

                    # Draw coordinates for the selected polygon
                    if i == self.selected_polygon_id:
                        coord_text = f"({point[0]}, {point[1]})"
                        cv2.putText(display_img, coord_text, (point[0] + 10, point[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Label polygon
                if len(polygon.points) > 0:
                    centroid_x = sum(p[0] for p in polygon.points) // len(polygon.points)
                    centroid_y = sum(p[1] for p in polygon.points) // len(polygon.points)
                    cv2.putText(display_img, f"ROI {i+1}", (centroid_x, centroid_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw current polygon being created
            if len(self.current_points) > 1:
                cv2.polylines(display_img, [np.array(self.current_points, dtype=np.int32)], 
                              False, (0, 255, 255), 2)
            
            # Draw points for current polygon and their coordinates
            for i, point in enumerate(self.current_points):
                cv2.circle(display_img, point, 5, (0, 0, 255), -1)
                cv2.putText(display_img, str(i+1), (point[0]+5, point[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # Also show coordinates for the current polygon being drawn
                coord_text = f"({point[0]}, {point[1]})"
                cv2.putText(display_img, coord_text, (point[0] + 10, point[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Draw mode indicator
            mode_text = "EDIT MODE" if self.editing_mode else "SELECTION MODE"
            mode_color = (0, 0, 255) if self.editing_mode else (0, 255, 0)
            cv2.putText(display_img, mode_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
            
            # Draw grid status
            grid_text = f"Grid: {'ON' if self.snap_to_grid else 'OFF'} ({self.grid_size}px)"
            cv2.putText(display_img, grid_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw help overlay if needed
            self.draw_help_overlay(display_img)
            
            cv2.imshow(self.window_name, display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if not self.editing_mode:
                if key == 32 or key == 13:  # SPACE or ENTER
                    if len(self.current_points) >= 3:
                        # Save current polygon
                        polygon = PolygonROI(points=self.current_points.copy(), id=len(self.polygons))
                        self.polygons.append(polygon)
                        self.current_points = []
                        self.save_state()
                        print(f"Polygon {len(self.polygons)} created. Press 'n' for new polygon, 'e' to edit, or 'f' to finish.")
                    else:
                        print("Need at least 3 points to form a polygon")
                elif key == ord('n'):  # New polygon
                    if len(self.current_points) >= 3:
                        # Save current polygon
                        polygon = PolygonROI(points=self.current_points.copy(), id=len(self.polygons))
                        self.polygons.append(polygon)
                        self.current_points = []
                        self.save_state()
                        print(f"Polygon {len(self.polygons)} created. Starting new polygon.")
                    else:
                        print("Need at least 3 points to save current polygon")
                elif key == ord('e'):  # Enter edit mode
                    if len(self.polygons) > 0:
                        self.editing_mode = True
                        print("Entered edit mode. Click on a polygon to select it and see coordinates.")
                    else:
                        print("No polygons to edit. Create at least one polygon first.")
                elif key == ord('f'):  # Finish all polygons
                    if len(self.current_points) >= 3:
                        # Save current polygon
                        polygon = PolygonROI(points=self.current_points.copy(), id=len(self.polygons))
                        self.polygons.append(polygon)
                    self.finished = True
                elif key == ord('c'):  # Cancel
                    self.polygons = []
                    self.current_points = []
                    self.finished = True
                elif key == ord('r'):  # Reset current polygon
                    self.current_points = []
                elif key == ord('z'):  # Undo
                    if self.undo():
                        print("Undo successful")
                elif key == ord('x'):  # Redo
                    if self.redo():
                        print("Redo successful")
                elif key == ord('g'):  # Toggle grid
                    self.snap_to_grid = not self.snap_to_grid
                    print(f"Grid snap {'enabled' if self.snap_to_grid else 'disabled'}")
                elif key == ord('h'):  # Toggle help
                    self.show_help = not self.show_help
            else:  # Edit mode
                if key == ord('s'):  # Exit edit mode
                    self.editing_mode = False
                    self.selected_polygon_id = None
                    self.selected_point_idx = None
                    print("Exited edit mode. Back to selection mode.")
                elif key == ord('d'):  # Delete selected polygon
                    if self.selected_polygon_id is not None:
                        self.save_state()
                        deleted_id = self.selected_polygon_id
                        self.polygons.pop(deleted_id)
                        print(f"Deleted ROI {deleted_id+1}")
                        self.selected_polygon_id = None
                        self.selected_point_idx = None
                    else:
                        print("No polygon selected for deletion")
                elif key == ord('r'):  # Delete selected point
                    if self.selected_polygon_id is not None and self.selected_point_idx is not None:
                        self.save_state()
                        polygon = self.polygons[self.selected_polygon_id]
                        if len(polygon.points) > 3:  # Keep at least 3 points
                            polygon.points.pop(self.selected_point_idx)
                            print(f"Deleted point from ROI {self.selected_polygon_id+1}")
                            self.selected_point_idx = None
                        else:
                            print("Cannot delete point. Polygon must have at least 3 points.")
                    else:
                        print("No point selected for deletion")
                elif key == ord('i'):  # Insert point
                    if self.selected_polygon_id is not None:
                        print("Click on the polygon edge where you want to insert a new point")
                        # The actual point insertion will happen in the mouse callback
                        # when a click is detected
                    else:
                        print("No polygon selected for inserting a point")
                elif key == ord('z'):  # Undo
                    if self.undo():
                        print("Undo successful")
                elif key == ord('x'):  # Redo
                    if self.redo():
                        print("Redo successful")
                elif key == ord('g'):  # Toggle grid
                    self.snap_to_grid = not self.snap_to_grid
                    print(f"Grid snap {'enabled' if self.snap_to_grid else 'disabled'}")
                elif key == ord('h'):  # Toggle help
                    self.show_help = not self.show_help
                elif key == 27:  # ESC
                    self.editing_mode = False
                    self.selected_polygon_id = None
                    self.selected_point_idx = None
                    print("Cancelled edit mode")
                
        cv2.destroyWindow(self.window_name)
        return self.polygons

