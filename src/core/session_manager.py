
    
    
import logging
from typing import Optional, List, Tuple

class SessionManager:
    """Manages navigation between images and related state."""
    
    def __init__(self, 
                 file_manager, 
                 window_manager, 
                 event_handler,   
                 logger: Optional[logging.Logger] = None):
        """Initialize session manager with components needed for state management."""
        self.logger = logger or logging.getLogger(__name__)
        self.file_manager = file_manager
        self.window_manager = window_manager
        self.event_handler = event_handler
        
        # Initialize basic state
        self.current_idx = 0
        self.image_files = self.file_manager.get_image_list()
        self.total_images = len(self.image_files)
    
    def get_current_image_path(self) -> str:
        """Get the current image path."""
        return str(self.file_manager.structure['images'] / self.image_files[self.current_idx])
    
    def _clear_state(self) -> None:
        """Clear current state before loading new image."""
        self.event_handler.reset_state()
        self.window_manager.set_mask(None)
        
    def can_move_prev(self) -> bool:
        """Check if we can move to previous image."""
        return self.current_idx > 0
    
    def can_move_next(self) -> bool:
        """Check if we can move to next image."""
        return self.current_idx < len(self.image_files) - 1
    
    def prev_image(self) -> Optional[str]:
        """Move to previous image and return its path if successful."""
        if self.can_move_prev():
            # Clear state first
            self._clear_state()
            
            # Then move to previous image
            self.current_idx -= 1
            return self.get_current_image_path()
        return None
    
    def next_image(self) -> Optional[str]:
        """Move to next image and return its path if successful."""
        if self.can_move_next():
            # Clear state first
            self._clear_state()
            
            # Then move to next image
            self.current_idx += 1
            return self.get_current_image_path()
        return None