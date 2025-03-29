from typing import Dict, Optional, Tuple, Callable
import cv2
import numpy as np

class ViewControls:
    """Widget for controlling view settings like zoom, pan, and layer visibility."""
    
    def __init__(self, window_name: str = "View Controls", width: int = 300, logger=None):
        """Initialize the view controls widget."""
        self.window_name = "View Controls"
        self.logger = logger
        self.is_visible = False  # Start hidden by default
        
        # View state
        self.view_state = {
            'show_masks': True,
            'show_boxes': True,
            'show_labels': True,
            'show_points': True
        }
        
        # Callback for state changes
        self.on_state_change = None
        
    def _notify_state_change(self):
        """Notify callback of state changes."""
        if self.on_state_change:
            if self.logger:
                self.logger.debug(f"View state changed: {self.view_state}")
            self.on_state_change(self.view_state.copy())
            
    def toggle_state(self, key: str) -> None:
        """Toggle a boolean state value."""
        if key in self.view_state:
            self.view_state[key] = not self.view_state[key]
            if self.logger:
                self.logger.info(f"Toggled {key} to {self.view_state[key]}")
            self._notify_state_change()
            self.render()
        
    def register_callback(self, callback):
        """Register callback for state changes."""
        self.on_state_change = callback
        
    def _create_control_panel(self):
        """Create the control panel image."""
        self.panel = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.panel.fill(50)  # Dark gray background
        
    def _draw_slider(self, y_pos: int, name: str, value: float, 
                    min_val: float, max_val: float) -> None:
        """Draw a slider control."""
        # Draw slider track
        track_start = self.padding
        track_end = self.width - self.padding
        track_width = track_end - track_start
        
        cv2.line(self.panel,
                 (track_start, y_pos),
                 (track_end, y_pos),
                 (100, 100, 100), 1)
        
        # Draw slider handle
        handle_pos = int(track_start + (value - min_val) * track_width / (max_val - min_val))
        cv2.circle(self.panel,
                  (handle_pos, y_pos),
                  6, (200, 200, 200), -1)
        
        # Draw label and value
        cv2.putText(self.panel,
                   f"{name}: {value:.2f}",
                   (track_start, y_pos - 10),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (200, 200, 200), 1)
                   
    def _draw_toggle(self, y_pos: int, name: str, value: bool) -> None:
        """Draw a toggle control."""
        # Draw toggle background
        color = (0, 255, 0) if value else (100, 100, 100)
        cv2.rectangle(self.panel,
                     (self.padding, y_pos),
                     (self.padding + 30, y_pos + 20),
                     color, -1)
        
        # Draw label
        cv2.putText(self.panel,
                   name,
                   (self.padding + 40, y_pos + 15),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (200, 200, 200), 1)
                   
    def _get_slider_value(self, mouse_x: int, slider_min: float, 
                         slider_max: float) -> float:
        """Convert mouse position to slider value."""
        track_start = self.padding
        track_end = self.width - self.padding
        track_width = track_end - track_start
        
        # Clamp mouse position to track bounds
        mouse_x = max(track_start, min(track_end, mouse_x))
        
        # Convert to value
        ratio = (mouse_x - track_start) / track_width
        return slider_min + ratio * (slider_max - slider_min)
     
    def update_state(self, key: str, value: any) -> None:
        """Update a state value and notify change."""
        if key in self.view_state:
            self.view_state[key] = value
            if self.logger:
                self.logger.debug(f"Updated {key} to {value}")
            self._notify_state_change()
            self.render()
            
    def toggle_visibility(self) -> None:
        """Toggle the control panel visibility."""
        self.is_visible = not self.is_visible
        if self.is_visible:
            cv2.namedWindow(self.window_name)
            self.render()
        else:
            cv2.destroyWindow(self.window_name)
            
        if self.logger:
            self.logger.debug(f"View controls visibility toggled to {self.is_visible}")
            
    def handle_keyboard(self, key: int) -> bool:
        """Handle keyboard shortcuts for view controls."""
        try:
            char = chr(key).lower()
            handled = True
            
            if char == 'v':  # Toggle visibility
                self.toggle_visibility()
            elif char == 'm':  # Toggle masks
                self.view_state['show_masks'] = not self.view_state['show_masks']
            elif char == 'b':  # Toggle boxes
                self.view_state['show_boxes'] = not self.view_state['show_boxes']
            elif char == 'l':  # Toggle labels
                self.view_state['show_labels'] = not self.view_state['show_labels']
            elif char == 't':  # Toggle points (t for targets)
                self.view_state['show_points'] = not self.view_state['show_points']
            else:
                handled = False
                
            if handled:
                self._notify_state_change()
                if self.is_visible:
                    self.render()
                return True
                    
        except ValueError:
            pass
            
        return False
            
    
    
    def handle_mouse(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse events in control panel."""
        if not self.is_visible:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check toggle buttons
            toggles = [
                ('show_masks', 150),
                ('show_boxes', 180),
                ('show_labels', 210),
                ('show_points', 240)
            ]
            
            for key, y_pos in toggles:
                if 10 <= x <= 40 and y_pos <= y <= y_pos + 20:
                    self.view_state[key] = not self.view_state[key]
                    self._notify_state_change()
                    self.render()
                    break

    def render(self) -> None:
        """Render the control panel if visible."""
        if not self.is_visible:
            return
            
        # Create panel
        height = 300
        width = 200
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel.fill(50)  # Dark gray background
        
        # Draw title
        cv2.putText(panel, "View Controls",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (200, 200, 200), 1)
        
        # Draw toggles
        toggles = [
            ('show_masks', 'Masks (M)', 150),
            ('show_boxes', 'Boxes (B)', 180),
            ('show_labels', 'Labels (L)', 210),
            ('show_points', 'Points (T)', 240)
        ]
        
        for key, label, y in toggles:
            # Toggle box
            color = (0, 255, 0) if self.view_state[key] else (100, 100, 100)
            cv2.rectangle(panel, (10, y), (40, y + 20), color, -1)
            
            # Label
            cv2.putText(panel, label,
                       (50, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (200, 200, 200), 1)
        
        cv2.imshow(self.window_name, panel)
   
    def get_state(self) -> Dict:
        """Get current view state."""
        return self.view_state.copy()
    
    def destroy(self) -> None:
        """Destroy the view controls window."""
        if self.is_visible:
            cv2.destroyWindow(self.window_name)