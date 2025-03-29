from typing import Optional, Tuple, List, Callable
import cv2
import numpy as np
from ..config.shortcuts import SHORTCUTS

class EventHandler: 
    """Handles all mouse and keyboard events for the SAM Annotator."""
    
    def __init__(self, window_manager, logger):
        """Initialize event handler."""
        self.window_manager = window_manager
        self.logger  = logger
        self.drawing = False
        self.box_start: Optional[Tuple[int, int]] = None
        self.box_end: Optional[Tuple[int, int]] = None
        self.points: List[List[int]] = []
        self.point_labels: List[int] = []
        
        # View control constants
        self.ZOOM_STEP = 0.1
        self.OPACITY_STEP = 0.1
        
        # Callback storage
        self.on_mask_prediction: Optional[Callable] = None
        self.on_class_selection: Optional[Callable] = None
        
    def register_callbacks(self, 
                         on_mask_prediction: Callable,
                         on_class_selection: Callable) -> None: 
        """Register callback functions."""
        self.on_mask_prediction = on_mask_prediction
        self.on_class_selection = on_class_selection

    def handle_mouse_event(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse events for the main window."""
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.box_start = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.box_end = (x, y)
                if self.on_mask_prediction:
                    self.on_mask_prediction(self.box_start, self.box_end, drawing=True)
                
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.box_end = (x, y)
                self.points = []
                self.point_labels = []
                
                # Add center point of the box
                center_x = (self.box_start[0] + x) // 2
                center_y = (self.box_start[1] + y) // 2
                self.points.append([center_x, center_y])
                self.point_labels.append(1)
                
                if self.on_mask_prediction:
                    self.on_mask_prediction(self.box_start, self.box_end)
                    
        except Exception as e:
            print(f"Error in mouse callback: {str(e)}")

    def handle_class_window_event(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse events for the class selection window."""
        if event == cv2.EVENT_LBUTTONDOWN:
            button_height = 30
            selected_class = y // button_height
            if self.on_class_selection:
                self.on_class_selection(selected_class)
 
    def handle_keyboard_event(self, key: int) -> Optional[str]: 
        """Handle keyboard events and return action string."""
        try:
            # Handle function keys
            if key == -1:  # No key pressed
                return None
            
            if key == ord('e'):
                self.logger.info("Export mode - waiting for format selection (c/y)...")
                second_key = cv2.waitKey(0) & 0xFF
                
                if second_key == ord('c'):
                    self.logger.info("COCO export selected")
                    return "export_coco"
                elif second_key == ord('y'):
                    self.logger.info("YOLO export selected")
                    return "export_yolo"
                
                elif second_key == ord('p'):
                    self.logger.info("Pascal export selected")
                    return "export_pascal"  # Make sure this is being returned
                
            # Let view controls handle their keys first
            if self.window_manager.view_controls.handle_keyboard(key):
                return "update_view"
                
            # Let annotation review handle its keys
            if self.window_manager.annotation_review.handle_keyboard(key):
                return "update_view"
                
            # Convert key to character, handling both upper and lowercase
            try:
                char = chr(key).lower()
                self.logger.debug(f"Key pressed: {char} (ASCII: {key})")
            except ValueError:
                self.logger.debug(f"Special key pressed: {key}")
                return None
            
            # Basic navigation shortcuts
            if char == SHORTCUTS['quit']:
                self.logger.info("Quit command received")
                return 'quit'
            elif char == SHORTCUTS['next_image']:
                self.logger.info("Next image command received")
                return 'next'
            elif char == SHORTCUTS['prev_image']:
                self.logger.info("Previous image command received")
                return 'prev'
            elif char == SHORTCUTS['save']:
                self.logger.info("Save command received")
                return 'save'
            elif char == SHORTCUTS['clear_selection']:
                self.logger.info("Clear selection command received")
                return 'clear_selection'
            elif char == SHORTCUTS['add_annotation']:
                self.logger.info("Add annotation command received")
                return 'add'
            elif char == SHORTCUTS['undo']:
                self.logger.info("Undo command received")
                return 'undo'
            elif char == SHORTCUTS['redo']:
                self.logger.info("Redo command received")
                return 'redo'
            elif char == SHORTCUTS['clear_all']:
                self.logger.info("Clear all command received")
                return 'clear_all'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in keyboard event handler: {str(e)}")
            return None 
   
    def reset_state(self) -> None:
        """Reset the event handler state."""
        self.drawing = False
        self.box_start = None
        self.box_end = None
        self.points = []
        self.point_labels = []