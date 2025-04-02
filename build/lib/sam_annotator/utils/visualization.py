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
        
        # Point colors for prompts
        self.foreground_point_color = (0, 255, 0)  # Green for foreground points
        self.background_point_color = (255, 0, 0)  # Red for background points
        self.point_radius = 5
        self.point_thickness = 2
        
        # Visualization options
        self.show_grid = False
        self.grid_size = 50
        self.show_minimap = False
        self.minimap_scale = 0.2
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate visually distinct colors using improved HSV spacing."""
        colors = []
        # Use golden ratio for better distribution
        golden_ratio = 0.618033988749895
        
        # Predefined colors for first few classes to ensure distinctiveness
        base_colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 128, 0),   # Orange
            (128, 0, 255),   # Purple
            (0, 255, 128),   # Spring Green
            (255, 0, 128),   # Rose
        ]
        
        colors.extend(base_colors)
        
        # Generate additional colors if needed
        if n > len(base_colors):
            hue = 0.0
            for i in range(n - len(base_colors)):
                hue = (hue + golden_ratio) % 1.0
                sat = 0.8 + (i % 3) * 0.1  # Vary saturation
                val = 0.9 + (i % 2) * 0.1   # Vary value
                
                # Convert HSV to RGB
                rgb = cv2.cvtColor(np.uint8([[[
                    hue * 179,
                    sat * 255,
                    val * 255
                ]]]), cv2.COLOR_HSV2BGR)[0][0]
                
                colors.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
        
        return colors[:n]    
    
    def _get_text_color(self, background_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Determine text color based on background brightness."""
        # Calculate perceived brightness using weighted RGB values
        brightness = (0.299 * background_color[0] + 
                    0.587 * background_color[1] + 
                    0.114 * background_color[2])
        
        return (0, 0, 0) if brightness > 127 else (255, 255, 255)
     
   
   
   
   
   
   
   
   
   
   
    def _draw_mask(self, image: np.ndarray, mask: np.ndarray, 
                color: Tuple[int, int, int]) -> np.ndarray:
        """Draw a single mask on the image."""
        # Resize mask if dimensions don't match
        img_h, img_w = image.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        
        if mask_h != img_h or mask_w != img_w:
            mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        return cv2.addWeighted(image, 1.0, colored_mask, self.mask_opacity, 0)

    
    
    
    
    
    
    
    
    
    
    
    
    def _draw_box(self, image: np.ndarray, box: List[int], 
                  color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
        """Draw a bounding box on the image."""
        x1, y1, x2, y2 = map(int, box)
        return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    def _draw_label(self, image: np.ndarray, text: str, position: Tuple[int, int],
                    color: Tuple[int, int, int]) -> np.ndarray:
        """Draw text label with semi-transparent dark background."""
        return self._add_text_with_background(image, text, position)
    
    def _draw_points(self, image: np.ndarray, contour_points: np.ndarray,
                     color: Tuple[int, int, int]) -> np.ndarray:
        """Draw contour points on the image."""
        for point in contour_points:
            x, y = point[0]
            cv2.circle(image, (int(x), int(y)), 2, color, -1)
        return image
    
    def draw_input_points(self, image: np.ndarray, points: List[List[int]], 
                         point_labels: List[int]) -> np.ndarray:
        """Draw input points for point-based annotation.
        
        Args:
            image: The image to draw on
            points: List of [x, y] point coordinates
            point_labels: List of corresponding point labels (0=background, 1=foreground)
            
        Returns:
            Image with input points drawn
        """
        if not points or len(points) == 0:
            return image
            
        display = image.copy()
        
        for idx, (point, label) in enumerate(zip(points, point_labels)):
            x, y = point
            
            # Draw point with different color based on label
            color = self.foreground_point_color if label == 1 else self.background_point_color
            
            # Draw outer circle
            cv2.circle(display, (int(x), int(y)), self.point_radius, self.outline_color, self.point_thickness)
            
            # Draw inner circle with label color
            cv2.circle(display, (int(x), int(y)), self.point_radius - 2, color, -1)
            
            # Add small number to indicate click order
            cv2.putText(display, str(idx + 1), (int(x) + 7, int(y) + 7), 
                        self.font, 0.5, self.outline_color, 2)
            cv2.putText(display, str(idx + 1), (int(x) + 7, int(y) + 7), 
                        self.font, 0.5, (255, 255, 255), 1)
            
        return display
       
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
    
    def _add_text_with_background(self, image: np.ndarray, text: str, position: Tuple[int, int], 
                               font_scale: float = 0.6, bg_alpha: float = 0.5) -> np.ndarray:
        """Add text with semi-transparent dark background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        padding = 8
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness)
        
        # Calculate background rectangle coordinates with padding
        x, y = position
        rect_x1 = x - padding
        rect_y1 = y - text_height - padding
        rect_x2 = x + text_width + padding
        rect_y2 = y + padding
        
        # Create overlay for semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay,
                    (rect_x1, rect_y1),
                    (rect_x2, rect_y2),
                    (0, 0, 0),  # Black background
                    -1)
        
        # Apply transparency to background
        cv2.addWeighted(overlay, bg_alpha, image, 1 - bg_alpha, 0, image)
        
        # Draw text in white
        cv2.putText(image, text,
                    (x, y - baseline//2),
                    font, font_scale, (255, 255, 255), thickness)
        
        return image
    
   
    
    
    
    
    def create_composite_view(self,
                            image: np.ndarray,
                            annotations: List[Dict],
                            current_mask: Optional[np.ndarray] = None,
                            box_start: Optional[Tuple[int, int]] = None,
                            box_end: Optional[Tuple[int, int]] = None,
                            input_points: Optional[List[List[int]]] = None,
                            input_point_labels: Optional[List[int]] = None,
                            show_masks: bool = True,
                            show_boxes: bool = True,
                            show_labels: bool = True,
                            show_points: bool = True) -> np.ndarray:
        
        """Create composite view with all visualizations."""
        try:
            display = image.copy()
            img_h, img_w = display.shape[:2]
            
            # Draw saved annotations
            for annotation in annotations:
                class_id = annotation['class_id']
                color = self.colors[class_id % len(self.colors)]
                
                # Draw mask from contour points (simplified approach from original)
                if show_masks and 'contour_points' in annotation:
                    # Create mask from contour points
                    mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    cv2.drawContours(mask, [annotation['contour_points']], -1, 255, -1)
                    display = self._draw_mask(display, mask, color)
                    
                    # Calculate bounding box from contour for consistent display
                    if show_boxes or show_labels:
                        x, y, w, h = cv2.boundingRect(annotation['contour_points'])
                        box = [x, y, x + w, y + h]
                        
                        # Draw bounding box
                        if show_boxes:
                            display = self._draw_box(display, box, color)
                        
                        # Draw class label
                        if show_labels:
                            label_pos = (box[0], box[1] - 5)
                            class_name = annotation.get('class_name', f'Class {class_id}')
                            label_text = f"{class_name} ({class_id})"
                            display = self._draw_label(display, label_text, label_pos, color)
                
                # Draw contour points
                if show_points and 'contour_points' in annotation:
                    display = self._draw_points(display, annotation['contour_points'], color)
            
            # Draw current selection (active mask being created)
            if current_mask is not None and show_masks:
                # Convert to uint8 if needed
                if current_mask.dtype == bool:
                    current_mask = current_mask.astype(np.uint8)
                
                # Ensure the mask matches image dimensions
                if current_mask.shape[:2] != (img_h, img_w):
                    current_mask = cv2.resize(current_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                display = self._draw_mask(display, current_mask, (0, 255, 0))
                
            # Draw current box if drawing
            if box_start and box_end and show_boxes:
                current_box = [
                    min(box_start[0], box_end[0]),
                    min(box_start[1], box_end[1]),
                    max(box_start[0], box_end[0]),
                    max(box_start[1], box_end[1])
                ]
                display = self._draw_box(display, current_box, (0, 255, 0))
                
            # Draw input points for point-based annotation
            if input_points and input_point_labels and show_points:
                display = self.draw_input_points(display, input_points, input_point_labels)
                
            return display
        except Exception as e:
            import logging
            logging.error(f"Error in create_composite_view: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            # Return original image if there's an error
            return image.copy()
        
    
    
    
    
    
    
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
        
        # Add class information
        if current_class:
            class_text = f"Class: {current_class} (ID: {current_class_id})"
            overlay = self._add_text_with_background(overlay, class_text, (10, 30))
        
        # Add image counter
        if current_image_path and current_idx is not None and total_images is not None:
            current_img_name = os.path.basename(current_image_path)
            counter_text = f"Image {current_idx + 1}/{total_images} - {current_img_name}"
            overlay = self._add_text_with_background(overlay, counter_text, (10, 60))
        
        # Add annotation counter
        h, _ = image.shape[:2]
        annotations_text = f"Annotations: {num_annotations}"
        overlay = self._add_text_with_background(overlay, annotations_text, (10, h - 30))
        
        # Add status message
        if status:
            _, w = image.shape[:2]
            overlay = self._add_text_with_background(overlay, status, (w//2 - 100, 30))
            
        return overlay

    def _format_contour_for_drawing(self, contour_data) -> np.ndarray:
        """Format contour data in the format expected by cv2.drawContours.
        
        This method handles various input formats:
        - List of [x, y] points
        - List of [[x, y]] points
        - NumPy array with different shapes
        
        Returns:
            np.ndarray with shape (-1, 1, 2) suitable for cv2.drawContours
        """
        try:
            import logging
            logger = logging.getLogger(__name__)
            
            # If already NumPy array
            if isinstance(contour_data, np.ndarray):
                # Check shape and format accordingly
                if len(contour_data.shape) == 3 and contour_data.shape[1] == 1 and contour_data.shape[2] == 2:
                    # Already in correct format: (-1, 1, 2)
                    return contour_data
                elif len(contour_data.shape) == 2 and contour_data.shape[1] == 2:
                    # Format is (-1, 2), reshape to (-1, 1, 2)
                    return contour_data.reshape(-1, 1, 2)
            
            # Convert from list formats
            points = []
            
            # Handle different list formats
            if isinstance(contour_data, list):
                for point in contour_data:
                    if isinstance(point, list):
                        if len(point) == 2 and all(isinstance(p, (int, float)) for p in point):
                            # Format [x, y]
                            points.append([point])
                        elif len(point) == 1 and isinstance(point[0], list) and len(point[0]) == 2:
                            # Format [[x, y]]
                            points.append(point)
                        else:
                            logger.warning(f"Unexpected point format: {point}")
                    else:
                        logger.warning(f"Point is not a list: {point}")
            
            if len(points) < 3:
                logger.warning(f"Not enough valid points for a contour: {len(points)} points")
                raise ValueError("Not enough valid points for contour")
                
            return np.array(points, dtype=np.int32)
        except Exception as e:
            logging.error(f"Error formatting contour: {e}")
            # Return empty contour
            return np.array([[[0, 0]], [[0, 1]], [[1, 1]]], dtype=np.int32)