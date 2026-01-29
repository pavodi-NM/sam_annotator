from typing import Optional, Tuple, List, Callable, Dict
import cv2
import numpy as np
import time
from ..config.shortcuts import SHORTCUTS
from ..config.settings import BUTTON_HEIGHT

class EventHandler:
    """Handles all mouse and keyboard events for the SAM Annotator."""

    # Debounce time in seconds for critical actions
    DEBOUNCE_TIME = 0.3  # 300ms debounce

    # Actions that should be debounced to prevent accidental repeats
    DEBOUNCED_ACTIONS = {'save', 'add', 'undo', 'redo', 'clear_all', 'next', 'prev'}

    def __init__(self, window_manager, logger):
        """Initialize event handler."""
        self.window_manager = window_manager
        self.logger  = logger
        self.drawing = False
        self.box_start: Optional[Tuple[int, int]] = None
        self.box_end: Optional[Tuple[int, int]] = None
        self.points: List[List[int]] = []
        self.point_labels: List[int] = []

        # Annotation mode (box or point)
        self.mode = 'box'  # Default to box mode

        # View control constants
        self.ZOOM_STEP = 0.1
        self.OPACITY_STEP = 0.1

        # Callback storage
        self.on_mask_prediction: Optional[Callable] = None
        self.on_class_selection: Optional[Callable] = None
        self.on_point_prediction: Optional[Callable] = None

        # Debouncing: track last execution time for each action
        self._last_action_time: Dict[str, float] = {}
        
    def register_callbacks(self,
                         on_mask_prediction: Callable,
                         on_class_selection: Callable,
                         on_point_prediction: Optional[Callable] = None) -> None:
        """Register callback functions."""
        self.on_mask_prediction = on_mask_prediction
        self.on_class_selection = on_class_selection
        self.on_point_prediction = on_point_prediction

    def _should_debounce(self, action: str) -> bool:
        """Check if an action should be debounced (skipped due to recent execution).

        Args:
            action: The action name to check

        Returns:
            True if the action should be skipped (debounced), False if it should execute
        """
        if action not in self.DEBOUNCED_ACTIONS:
            return False

        current_time = time.time()
        last_time = self._last_action_time.get(action, 0)

        if current_time - last_time < self.DEBOUNCE_TIME:
            # Too soon since last execution, debounce this
            return True

        # Update last action time and allow execution
        self._last_action_time[action] = current_time
        return False

    def handle_mouse_event(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse events for the main window."""
        try:
            if self.mode == 'box':
                self._handle_box_mode(event, x, y)
            elif self.mode == 'point':
                self._handle_point_mode(event, x, y)
        except Exception as e:
            self.logger.error(f"Error in mouse callback: {str(e)}")
            
    def _handle_box_mode(self, event: int, x: int, y: int) -> None:
        """Handle mouse events in box annotation mode."""
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
                
    def _handle_point_mode(self, event: int, x: int, y: int) -> None:
        """Handle mouse events in point annotation mode."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left-click: Add foreground point (label=1)
            self.points.append([x, y])
            self.point_labels.append(1)
            
            if self.on_point_prediction:
                self.on_point_prediction(self.points, self.point_labels)
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click: Add background point (label=0)
            self.points.append([x, y])
            self.point_labels.append(0)
            
            if self.on_point_prediction:
                self.on_point_prediction(self.points, self.point_labels)

    def handle_class_window_event(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse events for the class selection window."""
        # Delegate all mouse events to the class selector widget
        selected_class_id = self.window_manager.class_selector.handle_mouse(event, x, y, flags, param)
        if selected_class_id is not None and self.on_class_selection:
            self.on_class_selection(selected_class_id)
 
    def handle_keyboard_event(self, key: int) -> Optional[str]:
        """Handle keyboard events and return action string."""
        try:
            # Handle function keys
            if key == -1 or key == 255:  # No key pressed
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
                
            # Let annotation review handle its keys first (w/s when visible)
            if self.window_manager.annotation_review.handle_keyboard(key):
                return "update_view"

            # Let class selector handle arrow keys for navigation (BEFORE view controls)
            # This ensures arrow keys go to class selection, not other handlers
            selected_class_id = self.window_manager.class_selector.handle_keyboard(key)
            if selected_class_id is not None:
                # Class was selected via keyboard, trigger callback
                if self.on_class_selection:
                    self.on_class_selection(selected_class_id)
                return "update_view"  # Update main view with new class selection

            # Let view controls handle their keys
            if self.window_manager.view_controls.handle_keyboard(key):
                return "update_view"
                
            # Convert key to character, handling both upper and lowercase
            try:
                char = chr(key).lower()
                self.logger.debug(f"Key pressed: {char} (ASCII: {key})")
            except ValueError:
                self.logger.debug(f"Special key pressed: {key}")
                return None
            
            # Toggle annotation mode
            if char == SHORTCUTS['toggle_mode']:
                if self.mode == 'box':
                    self.mode = 'point'
                    self.logger.info("Switched to point annotation mode")
                    return 'switch_mode_point'
                else:
                    self.mode = 'box'
                    self.logger.info("Switched to box annotation mode")
                    return 'switch_mode_box'
            
            # Basic navigation shortcuts
            action = None

            if char == SHORTCUTS['quit']:
                action = 'quit'
            elif char == SHORTCUTS['next_image']:
                action = 'next'
            elif char == SHORTCUTS['prev_image']:
                action = 'prev'
            elif char == SHORTCUTS['jump_to_image']:
                action = 'jump'
            elif char == SHORTCUTS['save']:
                action = 'save'
            elif char == SHORTCUTS['clear_selection']:
                action = 'clear_selection'
            elif char == SHORTCUTS['add_annotation']:
                action = 'add'
            elif char == SHORTCUTS['undo']:
                action = 'undo'
            elif char == SHORTCUTS['redo']:
                action = 'redo'
            elif char == SHORTCUTS['clear_all']:
                action = 'clear_all'
            elif char == SHORTCUTS['toggle_auto_advance']:
                action = 'toggle_auto_advance'

            # Apply debouncing for critical actions
            if action and self._should_debounce(action):
                return None  # Skip this repeated keypress

            if action:
                self.logger.info(f"{action.replace('_', ' ').title()} command received")

            return action
            
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
        
    def toggle_mode(self) -> str:
        """Toggle between box and point annotation modes."""
        if self.mode == 'box':
            self.mode = 'point'
            mode_name = 'point'
        else:
            self.mode = 'box'
            mode_name = 'box'
        
        self.reset_state()
        self.logger.info(f"Switched to {mode_name} annotation mode")
        return mode_name