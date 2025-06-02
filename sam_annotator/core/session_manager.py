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
    
    def jump_to_image(self, target_index: int) -> Optional[str]:
        """Jump to a specific image index.
        
        Args:
            target_index: Target image index (1-based, will be converted to 0-based)
            
        Returns:
            Image path if successful, None if invalid index
        """
        # Convert from 1-based to 0-based indexing
        zero_based_index = target_index - 1
        
        # Validate the index
        if 0 <= zero_based_index < len(self.image_files):
            # Clear state first
            self._clear_state()
            
            # Jump to the target image
            self.current_idx = zero_based_index
            self.logger.info(f"Jumped to image {target_index} (index {zero_based_index})")
            return self.get_current_image_path()
        else:
            self.logger.error(f"Invalid image number {target_index}. Valid range is 1-{len(self.image_files)}")
            return None
    
    def prompt_and_jump_to_image(self) -> Optional[str]:
        """Prompt user for image number and jump to it.
        
        Returns:
            Image path if successful jump, None if cancelled or invalid
        """
        try:
            # Display prompt with current range
            prompt = f"Jump to image (1-{len(self.image_files)}): "
            print(prompt, end='', flush=True)
            
            # Get user input
            user_input = input().strip()
            
            # Handle empty input (cancel)
            if not user_input:
                print("Jump cancelled.")
                return None
            
            # Try to parse the number
            try:
                target_number = int(user_input)
                return self.jump_to_image(target_number)
            except ValueError:
                print(f"Error: '{user_input}' is not a valid number. Please enter a number between 1 and {len(self.image_files)}.")
                return None
                
        except (EOFError, KeyboardInterrupt):
            print("\nJump cancelled.")
            return None