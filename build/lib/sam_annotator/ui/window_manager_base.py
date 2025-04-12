from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np
from ..config.settings import WINDOW_TITLE, CLASS_WINDOW_TITLE, BUTTON_HEIGHT
from ..utils.visualization import VisualizationManager

class WindowManager:
    """Manages all window-related operations for the SAM Annotator."""
    
    def __init__(self):
        """Initialize window manager."""
        self.main_window = WINDOW_TITLE
        self.class_window = CLASS_WINDOW_TITLE
        self.vis_manager = VisualizationManager()
        
        # Window state
        self.current_image: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.class_image: Optional[np.ndarray] = None
        self.image_scale: float = 1.0
        self.window_state = {
            'show_masks': True,
            'show_boxes': True,
            'show_labels': True
        }
        
        # Initialize windows
        cv2.namedWindow(self.main_window)
        cv2.namedWindow(self.class_window)
        
    def setup_windows(self, mouse_callback, class_callback) -> None:
        """Set up window callbacks and create trackbars."""
        cv2.setMouseCallback(self.main_window, mouse_callback)
        cv2.setMouseCallback(self.class_window, class_callback)
        
        # Create opacity trackbar
        cv2.createTrackbar('Mask Opacity', self.main_window, 50, 100,
                          lambda x: self.vis_manager.set_mask_opacity(x / 100))
    
    def update_main_window(self, 
                          image: np.ndarray,
                          annotations: List[Dict],
                          current_class: str,
                          current_class_id: int,
                          current_image_path: str,
                          current_idx: int,
                          total_images: int,
                          box_start: Optional[Tuple[int, int]] = None,
                          box_end: Optional[Tuple[int, int]] = None,
                          status: str = "") -> None:
        """Update main window with current state."""
        # Create composite view
        display = self.vis_manager.create_composite_view(
            image=image,
            annotations=annotations,
            current_mask=self.current_mask,
            box_start=box_start,
            box_end=box_end
        )
        
        # Add status overlay
        display = self.vis_manager.add_status_overlay(
            image=display,
            status=status,
            current_class=current_class,
            current_class_id=current_class_id,
            current_image_path=current_image_path,
            current_idx=current_idx,
            total_images=total_images,
            num_annotations=len(annotations)
        )
        
        # Apply scaling if needed
        if self.image_scale != 1.0:
            display = cv2.resize(display, None, 
                               fx=self.image_scale, 
                               fy=self.image_scale)
        
        cv2.imshow(self.main_window, display)
    
    def update_class_window(self, class_names: List[str], current_class_id: int) -> None:
        """Update class selection window."""
        button_height = BUTTON_HEIGHT
        height = len(class_names) * button_height
        self.class_image = np.zeros((height, 200, 3), dtype=np.uint8)
        
        for i, class_name in enumerate(class_names):
            y = i * button_height
            color = (0, 255, 0) if i == current_class_id else (200, 200, 200)
            cv2.rectangle(self.class_image, (0, y), 
                         (200, y + button_height), color, -1)
            cv2.putText(self.class_image, f"{i}: {class_name}", 
                       (5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 0), 1)
        
        cv2.imshow(self.class_window, self.class_image)
    
    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        """Set current mask."""
        self.current_mask = mask
    
    def toggle_view_option(self, option: str) -> None:
        """Toggle visibility options."""
        if option in self.window_state:
            self.window_state[option] = not self.window_state[option]
    
    def set_image_scale(self, scale: float) -> None:
        """Set image scale factor."""
        self.image_scale = max(0.1, min(5.0, scale))
    
    def destroy_windows(self) -> None:
        """Destroy all windows."""
        cv2.destroyAllWindows()