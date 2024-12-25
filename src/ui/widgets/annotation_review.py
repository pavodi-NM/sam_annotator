from typing import List, Dict, Optional, Callable
import cv2
import numpy as np

class AnnotationReview:
    """Widget for reviewing and managing annotations."""
    
    def __init__(self, window_name: str = "Annotation Review", width: int = 300):
        """
        Initialize the annotation review widget.
        
        Args:
            window_name (str): Name of the review window
            width (int): Width of the review panel
        """
        self.window_name = window_name
        self.width = width
        self.panel_height = 600  # Default height, will adjust based on content
        self.item_height = 60    # Height for each annotation entry
        self.padding = 5         # Padding between elements
        
        # State
        self.annotations: List[Dict] = []
        self.selected_idx: Optional[int] = None
        self.hover_idx: Optional[int] = None
        self.scroll_position = 0
        
        # Callbacks
        self.on_delete: Optional[Callable[[int], None]] = None
        self.on_select: Optional[Callable[[int], None]] = None
        self.on_class_change: Optional[Callable[[int, int], None]] = None
        
        # Create window
        cv2.namedWindow(self.window_name)
        
    def set_annotations(self, annotations: List[Dict]) -> None:
        """Set current annotations list."""
        self.annotations = annotations
        self.selected_idx = None
        self.hover_idx = None
        self._update_panel_size()
        
    def register_callbacks(self,
                         on_delete: Callable[[int], None],
                         on_select: Callable[[int], None],
                         on_class_change: Callable[[int, int], None]) -> None:
        """Register callback functions."""
        self.on_delete = on_delete
        self.on_select = on_select
        self.on_class_change = on_class_change
        
    def _update_panel_size(self) -> None:
        """Update panel size based on number of annotations."""
        total_height = max(600, len(self.annotations) * self.item_height)
        self.panel_height = total_height
        self.panel = np.zeros((total_height, self.width, 3), dtype=np.uint8)
    
    
    

    def _draw_annotation_entry(self, 
                             y_pos: int, 
                             annotation: Dict, 
                             idx: int,
                             is_selected: bool,
                             is_hovered: bool) -> None:
        """Draw a single annotation entry."""
        # Background
        color = (100, 100, 255) if is_selected else \
                (70, 70, 70) if is_hovered else \
                (50, 50, 50)
        cv2.rectangle(self.panel,
                     (0, y_pos),
                     (self.width, y_pos + self.item_height),
                     color, -1)
        
        # Annotation info
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        
        # Class info
        class_name = self.class_names[annotation['class_id']] if hasattr(self, 'class_names') else \
                    annotation.get('class_name', f"Class {annotation['class_id']}")
        class_text = f"Class: {class_name}"
        cv2.putText(self.panel, class_text,
                   (self.padding, y_pos + 20),
                   font, font_scale, text_color, 1)
        
        # Box coordinates
        box = annotation.get('box', [0, 0, 0, 0])
        box_text = f"Box: ({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})"
        cv2.putText(self.panel, box_text,
                   (self.padding, y_pos + 40),
                   font, font_scale, text_color, 1)
                   
        # Contour info (optional)
        if 'contour_points' in annotation:
            contour = annotation['contour_points']
            points_text = f"Points: {len(contour)}"
            cv2.putText(self.panel, points_text,
                       (self.padding, y_pos + 55),
                       font, font_scale, text_color, 1)
        
        # Delete button
        button_color = (0, 0, 200)
        cv2.rectangle(self.panel,
                     (self.width - 60, y_pos + 10),
                     (self.width - 10, y_pos + 30),
                     button_color, -1)
        cv2.putText(self.panel, "Delete",
                   (self.width - 55, y_pos + 25),
                   font, 0.4, (255, 255, 255), 1)
    
        
    def handle_mouse(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse events in the review panel."""
        # Adjust y for scroll position
        y += self.scroll_position
        
        # Calculate hovered annotation
        idx = y // self.item_height
        if 0 <= idx < len(self.annotations):
            if event == cv2.EVENT_MOUSEMOVE:
                self.hover_idx = idx
                self.render()
            
            elif event == cv2.EVENT_LBUTTONDOWN:
                # Check if click is on delete button
                item_y = idx * self.item_height
                if (self.width - 60 <= x <= self.width - 10 and
                    item_y + 10 <= y <= item_y + 30):
                    if self.on_delete:
                        self.on_delete(idx)
                else:
                    self.selected_idx = idx
                    if self.on_select:
                        self.on_select(idx)
                self.render()
    
    def handle_keyboard(self, key: int) -> None:
        """Handle keyboard events for the review panel."""
        if key == ord('w'):  # Scroll up
            self.scroll_position = max(0, self.scroll_position - 30)
            self.render()
        elif key == ord('s'):  # Scroll down
            max_scroll = max(0, self.panel_height - 600)
            self.scroll_position = min(max_scroll, self.scroll_position + 30)
            self.render()
            
    def render(self) -> None:
        """Render the review panel."""
        self.panel.fill(30)  # Dark background
        
        # Draw annotations
        for idx, annotation in enumerate(self.annotations):
            y_pos = idx * self.item_height - self.scroll_position
            
            # Skip if not in view
            if y_pos + self.item_height < 0 or y_pos > 600:
                continue
                
            self._draw_annotation_entry(
                y_pos=y_pos,
                annotation=annotation,
                idx=idx,
                is_selected=idx == self.selected_idx,
                is_hovered=idx == self.hover_idx
            )
        
        # Create display window (fixed height)
        display = self.panel[self.scroll_position:self.scroll_position + 600]
        cv2.imshow(self.window_name, display)
    
    def destroy(self) -> None:
        """Destroy the review window."""
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(self.window_name)
        except:
            pass