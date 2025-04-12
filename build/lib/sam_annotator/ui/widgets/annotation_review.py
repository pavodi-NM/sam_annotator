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
        self.panel_height = 600  # Default height
        self.item_height = 60    # Height for each annotation entry
        self.padding = 10        # Padding between elements
        self.header_height = 30  # Height for header
        self.is_visible = False # start hidden by default
        
        # State
        self.annotations: List[Dict] = []
        self.selected_idx: Optional[int] = None
        self.hover_idx: Optional[int] = None
        self.scroll_position = 0
        
        # Callbacks
        self.on_delete: Optional[Callable[[int], None]] = None
        self.on_select: Optional[Callable[[int], None]] = None
        self.on_class_change: Optional[Callable[[int, int], None]] = None
        self.mouse_callback: Optional[Callable] = None

    def toggle_visibility(self) -> None:
        """Toggle the review panel visibility."""
        self.is_visible = not self.is_visible
        if self.is_visible:
            cv2.namedWindow(self.window_name)
            if self.mouse_callback:
                cv2.setMouseCallback(self.window_name, self.mouse_callback)
            self.render()
        else:
            cv2.destroyWindow(self.window_name)
   
    def set_mouse_callback(self, callback: Callable) -> None:
        """Store mouse callback for when window is created."""
        self.mouse_callback = callback
        
    def set_annotations(self, annotations: List[Dict]) -> None:
        """Set current annotations list."""
        self.annotations = annotations
        self.selected_idx = None
        self.hover_idx = None
        self._update_panel_size()
        if self.is_visible:
            self.render()
        
    def register_callbacks(self,
                         on_delete: Callable[[int], None],
                         on_select: Callable[[int], None],
                         on_class_change: Callable[[int, int], None]) -> None:
        """Register callback functions."""
        self.on_delete = on_delete
        self.on_select = on_select
        self.on_class_change = on_class_change
        
    def _update_panel_size(self) -> None:
        """Update panel size based on content."""
        self.panel_height = max(600, len(self.annotations) * self.item_height + self.header_height)
        self.panel = np.zeros((self.panel_height, self.width, 3), dtype=np.uint8)
        
    def _draw_header(self) -> None:
        """Draw the header section."""
        # Header background
        cv2.rectangle(self.panel,
                     (0, 0),
                     (self.width, self.header_height),
                     (70, 70, 70), -1)

        # Header text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.panel,
                   f"Annotations ({len(self.annotations)})",
                   (self.padding, 20),
                   font, 0.5, (255, 255, 255), 1)

    def _draw_toolbar(self) -> None:
        """Draw toolbar with control buttons."""
        # Toolbar background
        cv2.rectangle(self.panel,
                     (0, 0),
                     (self.width, self.toolbar_height),
                     (60, 60, 60), -1)
        
        # Draw buttons
        x_pos = 5
        for button in self.toolbar_buttons:
            # Button background
            cv2.rectangle(self.panel,
                         (x_pos, 5),
                         (x_pos + button['width'], self.toolbar_height - 5),
                         (80, 80, 80), -1)
            
            # Icon
            cv2.putText(self.panel, button['icon'],
                       (x_pos + 8, self.toolbar_height - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (200, 200, 200), 1)
                       
            x_pos += button['width'] + 5
            
        # Add title
        title = f"Annotations ({len(self.annotations)})"
        cv2.putText(self.panel, title,
                   (x_pos + 10, self.toolbar_height - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (200, 200, 200), 1)
    
    def _draw_annotation_entry(self, y_pos: int, annotation: Dict, idx: int,
                             is_selected: bool, is_hovered: bool) -> None:
        """Draw a single annotation entry."""
        # Entry background
        bg_color = (100, 50, 50) if is_selected else \
                  (60, 60, 60) if is_hovered else \
                  (40, 40, 40)
        
        cv2.rectangle(self.panel,
                     (0, y_pos),
                     (self.width, y_pos + self.item_height),
                     bg_color, -1)

        # Text content
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)

        # Class name
        class_name = annotation.get('class_name', f"Class {annotation['class_id']}")
        cv2.putText(self.panel,
                   class_name,
                   (self.padding, y_pos + 20),
                   font, 0.5, text_color, 1)

        # Box coordinates
        box = annotation.get('box', [0, 0, 0, 0])
        box_text = f"{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}"
        cv2.putText(self.panel,
                   box_text,
                   (self.padding, y_pos + 40),
                   font, 0.4, text_color, 1)

        # Points
        if 'contour_points' in annotation:
            points_text = f"{len(annotation['contour_points'])} points"
            cv2.putText(self.panel,
                       points_text,
                       (self.padding, y_pos + 55),
                       font, 0.4, text_color, 1)

        # Delete button
        button_color = (0, 0, 200) if is_hovered else (0, 0, 150)
        cv2.rectangle(self.panel,
                     (self.width - 60, y_pos + 10),
                     (self.width - 10, y_pos + 30),
                     button_color, -1)
        cv2.putText(self.panel,
                   "Delete",
                   (self.width - 55, y_pos + 25),
                   font, 0.4, (255, 255, 255), 1)
      
    def handle_keyboard(self, key: int) -> bool:
        """Handle keyboard events for the review panel."""
        if key == 27:  # ESC
            self.toggle_visibility()
            return True
        elif key == ord('r'):  # Toggle visibility
            self.toggle_visibility()
            return True
        elif not self.is_visible:
            return False
            
        if key == ord('w'):  # Scroll up
            self.scroll_position = max(0, self.scroll_position - 30)
            self.render()
            return True
        elif key == ord('s'):  # Scroll down
            max_scroll = max(0, self.panel_height - 600)
            self.scroll_position = min(max_scroll, self.scroll_position + 30)
            self.render()
            return True
            
        return False
            
    def render(self) -> None:
        """Render the review panel."""
        if not self.is_visible:
            return

        self._update_panel_size()
        self.panel.fill(30)  # Dark background

        # Draw header
        self._draw_header()

        # Draw annotations
        for idx, annotation in enumerate(self.annotations):
            y_pos = self.header_height + idx * self.item_height - self.scroll_position
            
            # Skip if not in view
            if y_pos + self.item_height < self.header_height or y_pos > 600:
                continue
                
            self._draw_annotation_entry(
                y_pos=y_pos,
                annotation=annotation,
                idx=idx,
                is_selected=idx == self.selected_idx,
                is_hovered=idx == self.hover_idx
            )

        # Create display window (fixed height)
        display = self.panel[:600]
        cv2.imshow(self.window_name, display)
    
    def handle_mouse(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse events."""
        if not self.is_visible:
            return

        # Adjust y for scroll position
        y += self.scroll_position - self.header_height

        # Calculate hovered annotation
        idx = y // self.item_height
        
        if 0 <= idx < len(self.annotations):
            if event == cv2.EVENT_MOUSEMOVE:
                self.hover_idx = idx
                self.render()
            
            elif event == cv2.EVENT_LBUTTONDOWN:
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
    
    def destroy(self) -> None:
        """Destroy the review window."""
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(self.window_name)
        except:
            pass