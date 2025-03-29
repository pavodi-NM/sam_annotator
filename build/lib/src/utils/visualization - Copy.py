import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
import os

class VisualizationManager:
    """Enhanced visualization manager with additional features."""
    
    def __init__(self):
        """Initialize visualization manager."""
        # Display settings
        self.mask_opacity = 0.5
        self.box_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.text_thickness = 2
        
        self.text_color = (255, 255, 0)  # Yellow
        self.outline_color = (0, 0, 0)    # Black
        
        # Colors
        self.colors = {
            'text': (255, 255, 0),     # Yellow
            'outline': (0, 0, 0),       # Black
            'box': (0, 255, 0),         # Green
            'selected_box': (255, 165, 0),  # Orange
            'mask': (0, 0, 255),        # Red
            'contour': (0, 0, 255),     # Red
            'grid': (128, 128, 128)     # Gray
        }
        
        # Visualization options
        self.show_grid = False
        self.grid_size = 50
        self.show_minimap = False
        self.minimap_scale = 0.2
        
    def set_color_scheme(self, scheme: str = 'default') -> None:
        """Change color scheme."""
        if scheme == 'dark':
            self.colors.update({
                'text': (200, 200, 200),
                'outline': (50, 50, 50),
                'box': (0, 200, 0),
                'selected_box': (200, 120, 0),
                'mask': (200, 0, 0),
                'contour': (200, 0, 0),
                'grid': (100, 100, 100)
            })
        else:  # default scheme
            self.colors.update({
                'text': (255, 255, 0),
                'outline': (0, 0, 0),
                'box': (0, 255, 0),
                'selected_box': (255, 165, 0),
                'mask': (0, 0, 255),
                'contour': (0, 0, 255),
                'grid': (128, 128, 128)
            })
            
    def set_mask_opacity(self, opacity: float) -> None:
        """Set opacity value for mask visualization."""
        self.mask_opacity = max(0.0, min(1.0, opacity))  # Clamp between 0 and 1
            
    def add_grid_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add grid overlay to image."""
        if not self.show_grid:
            return image
            
        overlay = image.copy()
        h, w = image.shape[:2]
        
        # Draw vertical lines
        for x in range(0, w, self.grid_size):
            cv2.line(overlay, (x, 0), (x, h), self.colors['grid'], 1)
            
        # Draw horizontal lines
        for y in range(0, h, self.grid_size):
            cv2.line(overlay, (0, y), (w, y), self.colors['grid'], 1)
            
        return overlay
    
    def create_minimap(self, image: np.ndarray, 
                      view_rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Create minimap with current view rectangle."""
        if not self.show_minimap:
            return None
            
        # Scale down the image
        h, w = image.shape[:2]
        minimap_w = int(w * self.minimap_scale)
        minimap_h = int(h * self.minimap_scale)
        minimap = cv2.resize(image, (minimap_w, minimap_h))
        
        # Draw view rectangle if provided
        if view_rect:
            x1, y1, x2, y2 = view_rect
            x1_scaled = int(x1 * self.minimap_scale)
            y1_scaled = int(y1 * self.minimap_scale)
            x2_scaled = int(x2 * self.minimap_scale)
            y2_scaled = int(y2 * self.minimap_scale)
            cv2.rectangle(minimap, (x1_scaled, y1_scaled), 
                        (x2_scaled, y2_scaled), 
                        self.colors['selected_box'], 1)
            
        return minimap
    
    def create_side_by_side_view(self, 
                                original: np.ndarray, 
                                annotated: np.ndarray) -> np.ndarray:
        """Create side-by-side view of original and annotated images."""
        h, w = original.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = original
        combined[:, w:] = annotated
        
        # Add dividing line
        cv2.line(combined, (w, 0), (w, h), self.colors['grid'], 2)
        
        return combined
    
    def highlight_overlapping_regions(self, 
                                    image: np.ndarray,
                                    annotations: List[Dict]) -> np.ndarray:
        """Highlight regions where annotations overlap."""
        if not annotations:
            return image
            
        h, w = image.shape[:2]
        overlap_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create binary masks for each annotation
        for annotation in annotations:
            current_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(current_mask, [annotation['contour_points']], -1, 1, -1)
            overlap_mask += current_mask
            
        # Highlight areas where overlap_mask > 1
        display = image.copy()
        overlap_regions = overlap_mask > 1
        display[overlap_regions] = (display[overlap_regions] * 0.5 + 
                                  np.array([0, 0, 255]) * 0.5).astype(np.uint8)
        
        return display
    
    def create_measurement_overlay(self, 
                                 image: np.ndarray,
                                 start_point: Tuple[int, int],
                                 end_point: Tuple[int, int]) -> np.ndarray:
        """Create measurement overlay with distance and angle information."""
        display = image.copy()
        
        # Draw measurement line
        cv2.line(display, start_point, end_point, self.colors['text'], 2)
        
        # Calculate distance
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Calculate angle
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360
            
        # Draw measurement text
        mid_point = ((start_point[0] + end_point[0])//2,
                    (start_point[1] + end_point[1])//2)
        cv2.putText(display, f"{distance:.1f}px", 
                   (mid_point[0] + 10, mid_point[1]),
                   self.font, self.font_scale, self.colors['text'], 2)
        cv2.putText(display, f"{angle:.1f}°",
                   (mid_point[0] + 10, mid_point[1] + 25),
                   self.font, self.font_scale, self.colors['text'], 2)
                   
        return display
    
    def create_annotation_preview(self,
                                image: np.ndarray,
                                mask: np.ndarray,
                                class_id: int,
                                class_name: str) -> np.ndarray:
        """Create preview of annotation before adding it."""
        preview = image.copy()
        
        # Create semi-transparent mask overlay
        mask_overlay = np.zeros_like(preview)
        mask_overlay[mask] = self.colors['mask']
        preview = cv2.addWeighted(preview, 1, mask_overlay, self.mask_opacity, 0)
        
        # Get mask contour
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Draw contour
            cv2.drawContours(preview, contours, -1, self.colors['contour'], 2)
            
            # Add class label near contour centroid
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"Class {class_id}: {class_name}"
                
                # Draw text with outline
                cv2.putText(preview, label, (cx, cy),
                          self.font, self.font_scale,
                          self.colors['outline'],
                          self.text_thickness + 1)
                cv2.putText(preview, label, (cx, cy),
                          self.font, self.font_scale,
                          self.colors['text'],
                          self.text_thickness)
        
        return preview
    
    def create_composite_view(self,
                        image: np.ndarray,
                        annotations: List[dict],
                        current_mask: Optional[np.ndarray] = None,
                        box_start: Optional[Tuple[int, int]] = None,
                        box_end: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Create a composite view with all visual elements."""
        display = image.copy()
        
        # Draw all saved annotations
        for annotation in annotations:
            # Draw contour
            cv2.drawContours(display, [annotation['contour_points']], -1, 
                        (0, 0, 255), 2)  # Red contour
            
            # Draw bounding box
            box = annotation['box']
            cv2.rectangle(display, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (0, 255, 0), 2)  # Green box
            
            # Draw class label
            cv2.putText(display, f"Class: {annotation['class_id']}", 
                    (int(box[0]), int(box[1] - 10)),
                    self.font, 0.6, (0, 255, 0), 2)
        
        # Add current mask if exists
        if current_mask is not None:
            mask_overlay = np.zeros_like(display)
            mask_overlay[current_mask] = [0, 0, 255]  # Red mask
            cv2.addWeighted(display, 1, mask_overlay, self.mask_opacity, 0, display)
        
        # Add current box if drawing
        if box_start and box_end:
            cv2.rectangle(display, box_start, box_end, (0, 255, 0), 2)
        
        return display

    def add_status_overlay(self, image: np.ndarray, 
                          status: str = "",
                          current_class: str = "",
                          current_class_id: int = 0,
                          current_image_path: Optional[str] = None,
                          current_idx: Optional[int] = None,
                          total_images: Optional[int] = None,
                          num_annotations: int = 0) -> np.ndarray:
        """Add status text overlay to the image."""
        overlay = image.copy()
        
        def put_text_with_outline(img, text, position):
            """Helper function to put text with outline."""
            cv2.putText(img, text, position, self.font, self.font_scale, 
                       self.outline_color, self.box_thickness + 1)
            cv2.putText(img, text, position, self.font, self.font_scale, 
                       self.text_color, self.box_thickness)

        # Add class information
        if current_class:
            class_text = f"Current Class: {current_class} (ID: {current_class_id})"
            put_text_with_outline(overlay, class_text, (10, 30))
        
        # Add image counter
        if current_image_path and current_idx is not None and total_images is not None:
            current_img_name = os.path.basename(current_image_path)
            counter_text = f"Image: {current_idx + 1}/{total_images} - {current_img_name}"
            put_text_with_outline(overlay, counter_text, (10, 60))
        
        # Add annotation counter
        annotations_text = f"Current annotations: {num_annotations}"
        put_text_with_outline(overlay, annotations_text, (10, 90))
        
        # Add status message
        if status:
            h, w = image.shape[:2]
            put_text_with_outline(overlay, status, (w//2 - 100, 30))
            
        return overlay