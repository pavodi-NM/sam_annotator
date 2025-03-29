from typing import List, Tuple, Optional, Callable
import cv2
import numpy as np
from ...config.settings import BUTTON_HEIGHT

class ClassSelector:
    """Widget for class selection interface."""
    
    def __init__(self, window_name: str, width: int = 200):
        """
        Initialize the class selector widget.
        
        Args:
            window_name (str): Name of the window to create
            width (int): Width of the class selector window
        """
        self.window_name = window_name
        self.width = width
        self.button_height = BUTTON_HEIGHT
        self.classes: List[str] = []
        self.current_class_id: int = 0
        self.hover_idx: Optional[int] = None
        
        # Create window
        cv2.namedWindow(self.window_name)
        
    def set_classes(self, classes: List[str]) -> None:
        """Set available classes."""
        self.classes = classes
        self._update_window_size()
        
    def set_current_class(self, class_id: int) -> None:
        """Set current selected class."""
        if 0 <= class_id < len(self.classes):
            self.current_class_id = class_id
            
    def _update_window_size(self) -> None:
        """Update window size based on number of classes."""
        height = len(self.classes) * self.button_height
        self.window_image = np.zeros((height, self.width, 3), dtype=np.uint8)
        
    def _get_class_at_position(self, y: int) -> Optional[int]:
        """Get class index at given y coordinate."""
        idx = y // self.button_height
        if 0 <= idx < len(self.classes):
            return idx
        return None
        
    def handle_mouse(self, event: int, x: int, y: int, flags: int, param: any) -> Optional[int]:
        """
        Handle mouse events.
        
        Returns:
            Selected class index if a selection was made, None otherwise
        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_idx = self._get_class_at_position(y)
            self.render()
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            selected_idx = self._get_class_at_position(y)
            if selected_idx is not None:
                self.current_class_id = selected_idx
                self.render()
                return selected_idx
                
        return None
        
    def render(self) -> None:
        """Render the class selector window."""
        self.window_image.fill(0)  # Clear the image
        
        for i, class_name in enumerate(self.classes):
            y = i * self.button_height
            
            # Determine button color
            if i == self.current_class_id:
                color = (0, 255, 0)  # Selected - Green
            elif i == self.hover_idx:
                color = (100, 100, 100)  # Hover - Gray
            else:
                color = (50, 50, 50)  # Normal - Dark Gray
                
            # Draw button background
            cv2.rectangle(self.window_image, 
                         (0, y),
                         (self.width, y + self.button_height),
                         color, -1)
                         
            # Draw class name
            cv2.putText(self.window_image,
                       f"{i}: {class_name}",
                       (5, y + self.button_height - 8),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6,
                       (255, 255, 255),  # White text
                       1)
                       
            # Draw separator line
            cv2.line(self.window_image,
                    (0, y + self.button_height - 1),
                    (self.width, y + self.button_height - 1),
                    (0, 0, 0), 1)
                    
        cv2.imshow(self.window_name, self.window_image)
        
    def destroy(self) -> None:
        """Destroy the class selector window."""
        try:
            # Check if window exists by trying to get its property
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(self.window_name)
        except:
            pass  # Window doesn't exist or already destroyed