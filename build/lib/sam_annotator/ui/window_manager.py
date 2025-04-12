from typing import Optional, Tuple, List, Dict, Callable
import cv2
import numpy as np
from ..config.settings import WINDOW_TITLE, CLASS_WINDOW_TITLE
from ..utils.visualization import VisualizationManager
from .widgets.status_overlay import StatusOverlay
from .widgets.class_selector import ClassSelector
from .widgets.annotation_review import AnnotationReview
from .widgets.view_controls import ViewControls 

class WindowManager:
    """Manages all window-related operations for the SAM Annotator."""
    
    def __init__(self, logger=None):
        """Initialize window manager."""
        self.main_window = WINDOW_TITLE
        self.logger = logger
        self.vis_manager = VisualizationManager()
        
        # Initialize window state with default values
        self.window_state = {
            'show_masks': True,
            'show_boxes': True,
            'show_labels': True,
            'show_points': True,
            'mask_opacity': 0.5,
            'zoom_level': 1.0,
            'pan_x': 0,
            'pan_y': 0
        }
        
        # Initialize widgets
        self.status_overlay = StatusOverlay()
        self.class_selector = ClassSelector(CLASS_WINDOW_TITLE)
        self.annotation_review = AnnotationReview()
        self.view_controls = ViewControls(logger=self.logger)  # Pass logger to ViewControls
        
        # Sync view controls with window state
        self.view_controls.view_state = self.window_state.copy()
        
        # Window state 
        self.current_image: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        
        # Initialize windows (create main window - always visible)
        cv2.namedWindow(self.main_window)
        
        if self.logger:
            self.logger.info("WindowManager initialized")
    
    def setup_windows(self, 
                     mouse_callback, 
                     class_callback,
                     review_callbacks: Dict[str, Callable]) -> None:
        """Set up window callbacks."""
        # Set main window mouse callback (always visible)
        cv2.setMouseCallback(self.main_window, mouse_callback)
        
        # Set class selector callback
        cv2.setMouseCallback(self.class_selector.window_name, class_callback)
        
        # Store callbacks for annotation review (will be set when window is created)
        self.annotation_review.set_mouse_callback(
            lambda event, x, y, flags, param: 
                self.annotation_review.handle_mouse(event, x, y, flags, param)
        )
        
        # Register review callbacks
        self.annotation_review.register_callbacks(
            on_delete=review_callbacks['delete'],
            on_select=review_callbacks['select'],
            on_class_change=review_callbacks['class_change']
        )
        
        # Register view control callback
        self.view_controls.register_callback(self._handle_view_state_change)
        
        # Create opacity trackbar in main window
        cv2.createTrackbar('Mask Opacity', self.main_window, 50, 100,
                          lambda x: self.vis_manager.set_mask_opacity(x / 100))
    
    def _handle_view_state_change(self, new_state: Dict) -> None:
        """Handle changes in view controls."""
        try:
            if self.logger:
                self.logger.debug(f"Handling view state change: {new_state}")
            
            # Update window state
            self.window_state.update(new_state)
            
            # Update visualization settings
            self.vis_manager.set_mask_opacity(new_state['mask_opacity'])
            
            # Force main window update if we have current image data
            if hasattr(self, '_current_update_args'):
                if self.logger:
                    self.logger.debug("Forcing main window update")
                temp_args = self._current_update_args.copy()
                self.update_main_window(**temp_args)
            else:
                if self.logger:
                    self.logger.warning("No current update args available")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error handling view state change: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                
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
                          input_points: Optional[List[List[int]]] = None,
                          input_point_labels: Optional[List[int]] = None,
                          status: str = "",
                          annotation_mode: str = "box") -> None:
        """Update main window with current state."""
        try:
            # Store current arguments for state updates
            self._current_update_args = locals()
            del self._current_update_args['self']  # Remove self reference
            
            # Create composite view with current view settings
            display = self.vis_manager.create_composite_view(
                image=image,
                annotations=annotations,
                current_mask=self.current_mask,
                box_start=box_start,
                box_end=box_end,
                input_points=input_points,
                input_point_labels=input_point_labels,
                show_masks=self.window_state['show_masks'],
                show_boxes=self.window_state['show_boxes'],
                show_labels=self.window_state['show_labels'],
                show_points=self.window_state['show_points']
            )
            
            # Add status overlay
            display = self.status_overlay.render(
                image=display,
                status=status,
                current_class=current_class,
                current_class_id=current_class_id,
                current_image_path=current_image_path,
                current_idx=current_idx,
                total_images=total_images,
                num_annotations=len(annotations),
                annotation_mode=annotation_mode
            )
            
            # Apply zoom
            if self.window_state['zoom_level'] != 1.0:
                h, w = display.shape[:2]
                new_size = (int(w * self.window_state['zoom_level']),
                          int(h * self.window_state['zoom_level']))
                display = cv2.resize(display, new_size)
            
            cv2.imshow(self.main_window, display)
            
            if self.logger:
                self.logger.debug("Main window updated")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating main window: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
     
    def update_review_panel(self, annotations: List[Dict]) -> None:
        """Update annotation review panel."""
        self.annotation_review.set_annotations(annotations)
        self.annotation_review.render()    
         
    def handle_review_keyboard(self, key: int) -> None:
        """Handle keyboard events for review panel."""
        self.annotation_review.handle_keyboard(key)
         
    def handle_keyboard_event(self, key: int) -> Optional[str]:
        """Handle keyboard events."""
        # Let annotation review handle its keys
        if self.annotation_review.handle_keyboard(key):
            return "update_view"
            
        # Let view controls handle its keys
        if self.view_controls.handle_keyboard(key):
            return "update_view"
            
        # Handle other keyboard events...
        return None   
    
    def get_selected_annotation_idx(self) -> Optional[int]:
        """Get currently selected annotation index."""
        return self.annotation_review.selected_idx
       
    def update_class_window(self, class_names: List[str], current_class_id: int) -> None:
        """Update class selection window."""
        self.class_selector.set_classes(class_names)
        self.class_selector.set_current_class(current_class_id)
        self.class_selector.render()
    
    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        """Set current mask."""
        self.current_mask = mask
    
    def toggle_view_option(self, option: str) -> None:
        """Toggle visibility options."""
        if option in self.window_state:
            self.window_state[option] = not self.window_state[option]
            
            # Force update if we have current data
            if hasattr(self, '_current_update_args') and self._current_update_args:
                if self.logger:
                    self.logger.debug("Forcing window update after toggling view option")
                try:
                    temp_args = self._current_update_args.copy()
                    self.update_main_window(**temp_args)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error updating window after toggling view option: {e}")
    
    def set_image_scale(self, scale: float) -> None:
        """Set image scale factor."""
        self.image_scale = max(0.1, min(5.0, scale))
    
    def destroy_windows(self) -> None:
        """Destroy all windows."""
        cv2.destroyAllWindows()
        self.class_selector.destroy()