import os
import json
import time
import logging
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class SessionManager:
    """Manages annotation session state and navigation."""
    
    def __init__(self, file_manager, annotation_manager, window_manager):
        """Initialize session manager.
        
        Args:
            file_manager: Manager for file operations
            annotation_manager: Manager for annotation operations
            window_manager: Manager for UI updates
        """
        self.file_manager = file_manager
        self.annotation_manager = annotation_manager
        self.window_manager = window_manager
        self.logger = logging.getLogger(__name__)
        
        # Session state
        self.current_idx = 0
        self.current_image_path = None
        self.total_images = 0
        self.image_files = []
        self.session_start_time = None
        
        # Session persistence
        self.session_file = os.path.join(self.file_manager.category_path, 'session.json')

    def set_predictor(self, predictor):
        """Set the predictor after initialization."""
        self.predictor = predictor
        
    def initialize_session(self) -> None:
        """Initialize new annotation session."""
        try:
            # Get image files
            self.image_files = self.file_manager.get_image_files()
            if not self.image_files:
                raise ValueError(f"No images found in {self.file_manager.images_path}")
                
            self.total_images = len(self.image_files)
            
            # Find first unannotated image or last image
            self.current_idx = self.get_last_annotated_index()
            
            # Load first image
            self.load_current_image()
            
            # Record session start time
            self.session_start_time = datetime.now()
            
            # Initialize windows
            self.window_manager.update_class_window(
                self.annotation_manager.class_names,
                self.annotation_manager.current_class_id
            )
            self.window_manager.update_review_panel([])
            
            self.logger.info("Session initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing session: {str(e)}")
            raise
    def load_current_image(self) -> None:
        """Load current image and annotations."""
        try:
            if not self.image_files:
                raise ValueError("No images available")
                
            # Get current image path
            self.current_image_path = os.path.join(
                self.file_manager.images_path,
                self.image_files[self.current_idx]
            )
            
            # Load and process image
            display_image, metadata = self.file_manager.load_image(self.current_image_path)
            
            # Set current image in window manager
            self.window_manager.current_image = display_image.copy()
            
            # Load existing annotations
            annotations = self.file_manager.load_annotations(self.current_image_path)
            self.annotation_manager.annotations = annotations
            
            # Set image in predictor if available
            if hasattr(self, 'predictor'):
                original_image = cv2.imread(self.current_image_path)
                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                self.predictor.set_image(original_image_rgb)
            
            # Update display
            self.window_manager.update_main_window(
                image=display_image,
                annotations=annotations,
                current_class=self.annotation_manager.class_names[self.annotation_manager.current_class_id],
                current_class_id=self.annotation_manager.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=self.total_images
            )
            
            # Update review panel
            self.window_manager.update_review_panel(annotations)
            
            self.logger.info(f"Loaded image {self.current_idx + 1}/{self.total_images}")
            
        except Exception as e:
            self.logger.error(f"Error loading current image: {str(e)}")
            raise
    
    
    def next_image(self) -> bool: 
        """Move to next image.
        
        Returns:
            bool: True if successful
        """
        try:
            if self.current_idx < len(self.image_files) - 1:
                # Auto-save current annotations
                self.file_manager.auto_save(
                    self.annotation_manager.annotations,
                    self.current_image_path
                )
                
                # Clear current state
                self.annotation_manager.annotations = []
                self.window_manager.set_mask(None)
                
                # Move to next image
                self.current_idx += 1
                self.load_current_image()
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error moving to next image: {str(e)}")
            return False
            
    def prev_image(self) -> bool:
        """Move to previous image.
        
        Returns:
            bool: True if successful
        """
        try:
            if self.current_idx > 0:
                # Auto-save current annotations
                self.file_manager.auto_save(
                    self.annotation_manager.annotations,
                    self.current_image_path
                )
                
                # Clear current state
                self.annotation_manager.annotations = []
                self.window_manager.set_mask(None)
                
                # Move to previous image
                self.current_idx -= 1
                self.load_current_image()
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error moving to previous image: {str(e)}")
            return False
            
    def get_last_annotated_index(self) -> int:
        """Find last annotated image index.
        
        Returns:
            int: Index of last annotated image
        """
        try:
            for idx in range(len(self.image_files)):
                img_path = os.path.join(
                    self.file_manager.images_path,
                    self.image_files[idx]
                )
                label_path = self.file_manager.get_label_path(img_path)
                
                # If this image doesn't have annotations, start here
                if not os.path.exists(label_path):
                    return idx
                    
            # If all images are annotated, return last index
            return len(self.image_files) - 1
            
        except Exception as e:
            self.logger.error(f"Error finding last annotated index: {str(e)}")
            return 0
            
    def save_current_state(self) -> None:
        """Save current session state."""
        try:
            state = {
                'current_idx': self.current_idx,
                'session_start_time': self.session_start_time.isoformat(),
                'total_images': self.total_images,
                'last_save_time': datetime.now().isoformat()
            }
            
            with open(self.session_file, 'w') as f:
                json.dump(state, f, indent=4)
                
            self.logger.info("Session state saved")
            
        except Exception as e:
            self.logger.error(f"Error saving session state: {str(e)}")
            
    def restore_session(self) -> bool:
        """Restore previous session.
        
        Returns:
            bool: True if session was restored
        """
        try:
            if not os.path.exists(self.session_file):
                return False
                
            with open(self.session_file, 'r') as f:
                state = json.load(f)
                
            # Restore state
            self.current_idx = state['current_idx']
            self.session_start_time = datetime.fromisoformat(state['session_start_time'])
            
            # Load current image
            self.load_current_image()
            
            self.logger.info("Session restored successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring session: {str(e)}")
            return False
            
    def update_display(self) -> None:
        """Update display with current state."""
        try:
            self.window_manager.update_main_window(
                image=self.window_manager.current_image,
                annotations=self.annotation_manager.annotations,
                current_class=self.annotation_manager.class_names[self.annotation_manager.current_class_id],
                current_class_id=self.annotation_manager.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=self.total_images
            )
            
            self.window_manager.update_review_panel(self.annotation_manager.annotations)
            
        except Exception as e:
            self.logger.error(f"Error updating display: {str(e)}")
            
    def handle_window_resize(self, new_size: Tuple[int, int]) -> None:
        """Handle window resize events.
        
        Args:
            new_size: New window dimensions (width, height)
        """
        try:
            # Update image processor settings
            self.file_manager.image_processor.update_target_size(new_size)
            
            # Reload current image with new dimensions
            if self.current_image_path:
                display_image, _ = self.file_manager.load_image(self.current_image_path)
                
                # Scale annotations to new size
                scale_factor = display_image.shape[1] / self.window_manager.current_image.shape[1]
                self.annotation_manager.update_display_annotations(scale_factor)
                
                # Update display
                self.window_manager.current_image = display_image
                self.update_display()
                
        except Exception as e:
            self.logger.error(f"Error handling window resize: {str(e)}")
            
    def get_session_stats(self) -> Dict:
        """Get current session statistics.
        
        Returns:
            Dict containing session statistics
        """
        try:
            current_time = datetime.now()
            session_duration = current_time - self.session_start_time
            
            # Count annotated images
            annotated_count = 0
            for img_file in self.image_files:
                img_path = os.path.join(self.file_manager.images_path, img_file)
                if os.path.exists(self.file_manager.get_label_path(img_path)):
                    annotated_count += 1
                    
            stats = {
                'total_images': self.total_images,
                'annotated_images': annotated_count,
                'remaining_images': self.total_images - annotated_count,
                'completion_percentage': (annotated_count / self.total_images) * 100,
                'session_duration': str(session_duration),
                'current_image': self.current_idx + 1,
                'session_start': self.session_start_time.isoformat()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting session stats: {str(e)}")
            return {}