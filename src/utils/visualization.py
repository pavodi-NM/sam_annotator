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
        
        self.colors = self._generate_colors(100)  # Pre-generate colors for classes
        
        # Colors
        # self.colors = {
        #     'text': (255, 255, 0),     # Yellow
        #     'outline': (0, 0, 0),       # Black
        #     'box': (0, 255, 0),         # Green
        #     'selected_box': (255, 165, 0),  # Orange
        #     'mask': (0, 0, 255),        # Red
        #     'contour': (0, 0, 255),     # Red
        #     'grid': (128, 128, 128)     # Gray
        # }
        
        # Visualization options
        self.show_grid = False
        self.grid_size = 50
        self.show_minimap = False
        self.minimap_scale = 0.2
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization."""
        colors = []
        for i in range(n):
            hue = i / n
            sat = 0.7 + (i % 3) * 0.1
            val = 0.9 + (i % 2) * 0.1
            
            # Convert HSV to RGB
            rgb = cv2.cvtColor(np.uint8([[[
                hue * 179,
                sat * 255,
                val * 255
            ]]]), cv2.COLOR_HSV2BGR)[0][0]
            
            colors.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
        return colors
        
    def _draw_mask(self, image: np.ndarray, mask: np.ndarray, 
                   color: Tuple[int, int, int]) -> np.ndarray:
        """Draw a single mask on the image."""
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        return cv2.addWeighted(image, 1.0,
                             colored_mask, self.mask_opacity, 0)
    
    def _draw_box(self, image: np.ndarray, box: List[int], 
                  color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
        """Draw a bounding box on the image."""
        x1, y1, x2, y2 = map(int, box)
        return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    def _draw_label(self, image: np.ndarray, text: str, position: Tuple[int, int],
                    color: Tuple[int, int, int]) -> np.ndarray:
        """Draw text label with background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness)
            
        # Draw background rectangle
        x, y = position
        cv2.rectangle(image,
                     (x, y - text_height - baseline),
                     (x + text_width, y),
                     color, -1)
                     
        # Draw text
        cv2.putText(image, text,
                    (x, y - baseline),
                    font, scale, (255, 255, 255),
                    thickness)
        return image
    
    def _draw_points(self, image: np.ndarray, contour_points: np.ndarray,
                     color: Tuple[int, int, int]) -> np.ndarray:
        """Draw contour points on the image."""
        for point in contour_points:
            x, y = point[0]
            cv2.circle(image, (int(x), int(y)), 2, color, -1)
        return image
       
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
                            annotations: List[Dict],
                            current_mask: Optional[np.ndarray] = None,
                            box_start: Optional[Tuple[int, int]] = None,
                            box_end: Optional[Tuple[int, int]] = None,
                            show_masks: bool = True,
                            show_boxes: bool = True,
                            show_labels: bool = True,
                            show_points: bool = True) -> np.ndarray:
        """Create composite view with all visualizations."""
        display = image.copy()
        
        # Draw saved annotations
        for annotation in annotations:
            class_id = annotation['class_id']
            color = self.colors[class_id % len(self.colors)]
            
            # Draw mask from contour points
            if show_masks and 'contour_points' in annotation:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                #cv2.drawContours(mask, [annotation['contour_points']], -1, 255, -1) #   cv2.drawContours(mask, [annotation['contour_points']], -1, 255, -1)
                cv2.drawContours(mask, [annotation['contour_points']], -1, 255, -1)
                display = self._draw_mask(display, mask, color)
            
            # Draw bounding box
            if show_boxes and 'box' in annotation:
                display = self._draw_box(display, annotation['box'], color)
            
            # Draw class label
            if show_labels and 'box' in annotation:
                label_pos = (int(annotation['box'][0]), int(annotation['box'][1]) - 5)
                display = self._draw_label(display,
                                         f"Class: {class_id}",
                                         label_pos, color)
            
            # Draw contour points
            if show_points and 'contour_points' in annotation:
                display = self._draw_points(display,
                                          annotation['contour_points'],
                                          color)
        
        # Draw current selection
        if current_mask is not None and show_masks:
            display = self._draw_mask(display, current_mask, (0, 255, 0))
            
        if box_start and box_end and show_boxes:
            current_box = [
                min(box_start[0], box_end[0]),
                min(box_start[1], box_end[1]),
                max(box_start[0], box_end[0]),
                max(box_start[1], box_end[1])
            ]
            display = self._draw_box(display, current_box, (0, 255, 0))
            
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