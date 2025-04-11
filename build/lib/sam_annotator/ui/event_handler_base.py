from typing import Optional, Tuple, List, Callable
import cv2
import numpy as np
from ..config.shortcuts import SHORTCUTS

class EventHandler:
    """Handles all mouse and keyboard events for the SAM Annotator."""
    
    def __init__(self, window_manager):
        """Initialize event handler."""
        self.window_manager = window_manager
        self.drawing = False
        self.box_start: Optional[Tuple[int, int]] = None
        self.box_end: Optional[Tuple[int, int]] = None
        self.points: List[List[int]] = []
        self.point_labels: List[int] = []
        
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
            char = chr(key)
            if char == SHORTCUTS['quit']:
                return 'quit'
            elif char == SHORTCUTS['next_image']:
                return 'next'
            elif char == SHORTCUTS['prev_image']:
                return 'prev'
            elif char == SHORTCUTS['save']:
                return 'save'
            elif char == SHORTCUTS['clear_selection']:
                return 'clear_selection'
            elif char == SHORTCUTS['add_annotation']:
                return 'add'
            elif char == SHORTCUTS['undo']:
                return 'undo'
            elif char == SHORTCUTS['clear_all']:
                return 'clear_all'
            return None
        except Exception as e:
            print(f"Error in keyboard event handler: {str(e)}")
            return None

    def reset_state(self) -> None:
        """Reset the event handler state."""
        self.drawing = False
        self.box_start = None
        self.box_end = None
        self.points = []
        self.point_labels = []