from typing import Dict, Optional, Tuple, Callable
import cv2
import numpy as np

class ViewControls:
    """Widget for controlling view settings like zoom, pan, and layer visibility."""
    
    def __init__(self, window_name: str = "View Controls", width: int = 300, logger=None):
        """Initialize the view controls widget."""
        self.window_name = window_name
        self.width = width
        self.height = 400
        self.padding = 10
        self.logger = logger
        
        # View state
        self.view_state = {
            'zoom_level': 1.0,
            'pan_x': 0,
            'pan_y': 0,
            'mask_opacity': 0.5,
            'show_masks': True,
            'show_boxes': True,
            'show_labels': True,
            'show_points': True
        }
        
        # Callbacks
        self.on_state_change: Optional[Callable[[Dict], None]] = None
        
        # Create window
        cv2.namedWindow(self.window_name)
        self._create_control_panel()
        
        if self.logger:
            self.logger.info("ViewControls initialized with default state")
            self.logger.debug(f"Initial view state: {self.view_state}")
        
    def _notify_state_change(self):
        """Notify window manager of state changes."""
        if self.on_state_change:
            if self.logger:
                self.logger.debug(f"Notifying state change: {self.view_state}")
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
            
    def toggle_visibility(self, key: str) -> None:
        """Toggle a visibility state."""
        if key in self.view_state:
            self.view_state[key] = not self.view_state[key]
            if self.logger:
                self.logger.info(f"Toggled {key} to {self.view_state[key]}")
            self._notify_state_change()
            self.render()
    
    def handle_mouse(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse events in the control panel."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is on a toggle button
            toggle_positions = {
                'show_masks': (self.padding, 150, self.padding + 30, 170),
                'show_boxes': (self.padding, 180, self.padding + 30, 200),
                'show_labels': (self.padding, 210, self.padding + 30, 230),
                'show_points': (self.padding, 240, self.padding + 30, 260)
            }
            
            for key, (x1, y1, x2, y2) in toggle_positions.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.view_state[key] = not self.view_state[key]
                    self._notify_state_change()
                    self.render()
                    return
            
            # Handle slider interactions
            slider_y_positions = {
                'zoom': 50,
                'opacity': 100
            }
            
            for name, y_pos in slider_y_positions.items():
                if abs(y - y_pos) < 10:
                    self.active_slider = name
                    # Update value immediately
                    new_value = self._get_slider_value(x, 
                                                     self.slider_values[name][1],
                                                     self.slider_values[name][2])
                    self.slider_values[name] = (new_value, 
                                              self.slider_values[name][1],
                                              self.slider_values[name][2])
                    
                    if name == 'zoom':
                        self.view_state['zoom_level'] = new_value
                    else:
                        self.view_state['mask_opacity'] = new_value
                    
                    self._notify_state_change()
                    self.render()
                    break
        
        elif event == cv2.EVENT_MOUSEMOVE and self.active_slider:
            # Update slider value
            _, min_val, max_val = self.slider_values[self.active_slider]
            new_value = self._get_slider_value(x, min_val, max_val)
            self.slider_values[self.active_slider] = (new_value, min_val, max_val)
            
            if self.active_slider == 'zoom':
                self.view_state['zoom_level'] = new_value
            else:
                self.view_state['mask_opacity'] = new_value
            
            self._notify_state_change()
            self.render()
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.active_slider = None

    def render(self):
        """Render the control panel."""
        self._create_control_panel()
        
        # Draw sliders
        y = 50
        # Zoom slider
        cv2.putText(self.panel, f"Zoom: {self.view_state['zoom_level']:.2f}",
                   (self.padding, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y = 100
        # Opacity slider
        cv2.putText(self.panel, f"Mask Opacity: {self.view_state['mask_opacity']:.2f}",
                   (self.padding, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw toggles
        toggles = [
            ('show_masks', 'Show Masks', 150),
            ('show_boxes', 'Show Boxes', 180),
            ('show_labels', 'Show Labels', 210),
            ('show_points', 'Show Points', 240)
        ]
        
        for key, label, y in toggles:
            # Draw toggle background
            color = (0, 255, 0) if self.view_state[key] else (100, 100, 100)
            cv2.rectangle(self.panel,
                         (self.padding, y),
                         (self.padding + 30, y + 20),
                         color, -1)
            
            # Draw label
            cv2.putText(self.panel, label,
                       (self.padding + 40, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (200, 200, 200), 1)
        
        cv2.imshow(self.window_name, self.panel)
   
    def get_state(self) -> Dict:
        """Get current view state."""
        return self.view_state.copy()
    
    def destroy(self) -> None:
        """Destroy the view controls window."""
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(self.window_name)
        except:
            pass