import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from ..core.command_manager import (
    AddAnnotationCommand,
    DeleteAnnotationCommand,
    ModifyAnnotationCommand
)

class AnnotationManager:
    """Manages annotation operations and state."""
    
    def __init__(self, validation_manager, window_manager, command_manager):
        """Initialize the annotation manager.
        
        Args:
            validation_manager: Manager for validating annotations
            window_manager: Manager for UI updates
            command_manager: Manager for undo/redo operations
        """
        self.validation_manager = validation_manager
        self.window_manager = window_manager
        self.command_manager = command_manager
        self.logger = logging.getLogger(__name__)
        
        # Internal state
        self.annotations: List[Dict] = []
        self.current_class_id: int = 0
        self.selected_annotation_idx: Optional[int] = None
        self.class_names: List[str] = []
        
    def add_annotation(self, annotation_data: Dict) -> bool:
        """Add new annotation with validation.
        
        Args:
            annotation_data: Dictionary containing annotation information
            
        Returns:
            bool: True if annotation was added successfully
        """
        try:
            # Validate the annotation
            is_valid, message = self.validation_manager.validate_annotation(
                annotation_data, annotation_data['mask'].shape)
                
            if not is_valid:
                self.logger.warning(f"Invalid annotation: {message}")
                self.window_manager.update_main_window(
                    status=f"Invalid annotation: {message}")
                return False
                
            # Check for overlap with existing annotations
            is_valid, overlap_ratio = self.validation_manager.check_overlap(
                self.annotations, annotation_data, annotation_data['mask'].shape)
                
            if not is_valid:
                self.logger.warning(f"Excessive overlap detected: {overlap_ratio:.2f}")
                self.window_manager.update_main_window(
                    status=f"Too much overlap: {overlap_ratio:.2f}")
                return False
                
            # Create and execute command
            command = AddAnnotationCommand(self.annotations, annotation_data, self.window_manager)
            success = self.command_manager.execute(command)
            
            if success:
                self.logger.info(f"Successfully added annotation. Total: {len(self.annotations)}")
                self.window_manager.update_review_panel(self.annotations)
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error adding annotation: {str(e)}")
            return False
            
    def delete_annotation(self, idx: int) -> bool:
        """Delete annotation at specified index.
        
        Args:
            idx: Index of annotation to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            if 0 <= idx < len(self.annotations):
                command = DeleteAnnotationCommand(self.annotations, idx, self.window_manager)
                success = self.command_manager.execute(command)
                
                if success:
                    self.logger.info(f"Successfully deleted annotation {idx}")
                    if self.selected_annotation_idx == idx:
                        self.selected_annotation_idx = None
                    self.window_manager.update_review_panel(self.annotations)
                    
                return success
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting annotation: {str(e)}")
            return False
            
    def modify_annotation(self, idx: int, new_data: Dict) -> bool:
        """Modify existing annotation.
        
        Args:
            idx: Index of annotation to modify
            new_data: New annotation data
            
        Returns:
            bool: True if modification was successful
        """
        try:
            if 0 <= idx < len(self.annotations):
                command = ModifyAnnotationCommand(self.annotations, idx, new_data, self.window_manager)
                success = self.command_manager.execute(command)
                
                if success:
                    self.logger.info(f"Successfully modified annotation {idx}")
                    self.window_manager.update_review_panel(self.annotations)
                    
                return success
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error modifying annotation: {str(e)}")
            return False
            
    def handle_class_change(self, idx: int, new_class_id: int) -> bool:
        """Handle class change for annotation.
        
        Args:
            idx: Index of annotation to modify
            new_class_id: New class ID
            
        Returns:
            bool: True if class change was successful
        """
        try:
            if 0 <= idx < len(self.annotations) and 0 <= new_class_id < len(self.class_names):
                new_state = self.annotations[idx].copy()
                new_state['class_id'] = new_class_id
                new_state['class_name'] = self.class_names[new_class_id]
                
                return self.modify_annotation(idx, new_state)
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error changing annotation class: {str(e)}")
            return False
            
    def select_annotation(self, idx: int) -> None:
        """Select annotation for editing.
        
        Args:
            idx: Index of annotation to select
        """
        try:
            if 0 <= idx < len(self.annotations):
                self.selected_annotation_idx = idx
                self.window_manager.update_main_window(
                    annotations=self.annotations,
                    selected_annotation_idx=idx
                )
                
        except Exception as e:
            self.logger.error(f"Error selecting annotation: {str(e)}")
            
    def clear_annotations(self) -> None:
        """Clear all annotations."""
        try:
            self.annotations = []
            self.selected_annotation_idx = None
            self.window_manager.update_main_window(
                annotations=[],
                status="All annotations cleared"
            )
            self.window_manager.update_review_panel([])
            
        except Exception as e:
            self.logger.error(f"Error clearing annotations: {str(e)}")
            
    def get_annotation(self, idx: int) -> Optional[Dict]:
        """Get annotation by index.
        
        Args:
            idx: Index of annotation to retrieve
            
        Returns:
            Optional[Dict]: Annotation data if found, None otherwise
        """
        try:
            if 0 <= idx < len(self.annotations):
                return self.annotations[idx].copy()
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving annotation: {str(e)}")
            return None
            
    def get_all_annotations(self) -> List[Dict]:
        """Get all current annotations.
        
        Returns:
            List[Dict]: List of all annotations
        """
        return self.annotations.copy()
        
    def validate_annotation(self, annotation: Dict) -> Tuple[bool, str]:
        """Validate single annotation.
        
        Args:
            annotation: Annotation data to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            return self.validation_manager.validate_annotation(
                annotation, annotation['mask'].shape)
                
        except Exception as e:
            self.logger.error(f"Error validating annotation: {str(e)}")
            return False, f"Validation error: {str(e)}"
            
    def update_display_annotations(self, scale_factor: float) -> None:
        """Update annotations for display with scaling.
        
        Args:
            scale_factor: Factor to scale annotations by
        """
        try:
            for annotation in self.annotations:
                # Scale contour points
                contour = annotation['contour_points'].copy()
                contour = contour.astype(np.float32)
                contour *= scale_factor
                annotation['contour_points'] = contour.astype(np.int32)
                
                # Update bounding box
                x, y, w, h = cv2.boundingRect(annotation['contour_points'])
                annotation['box'] = [x, y, x + w, y + h]
                
                # Update mask if present
                if 'mask' in annotation:
                    h, w = annotation['mask'].shape
                    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                    annotation['mask'] = cv2.resize(
                        annotation['mask'].astype(np.uint8),
                        (new_w, new_h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    
            self.window_manager.update_main_window(
                annotations=self.annotations
            )
            
        except Exception as e:
            self.logger.error(f"Error updating display annotations: {str(e)}")