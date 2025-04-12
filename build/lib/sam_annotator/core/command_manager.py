from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
import logging

# Base Command Interface
class Command(ABC):
    """Abstract base class for all commands."""
    
    @abstractmethod
    def execute(self) -> bool:
        """Execute the command. Returns True if successful."""
        pass
    
    @abstractmethod
    def undo(self) -> bool:
        """Undo the command. Returns True if successful."""
        pass
    
    @abstractmethod
    def redo(self) -> bool:
        """Redo the command. By default, just executes again."""
        pass

# Annotation Commands
@dataclass
class AnnotationState:
    """Data class to store annotation state."""
    class_id: int
    class_name: str
    mask: np.ndarray
    contour_points: np.ndarray
    original_contour: np.ndarray
    box: List[int]

class AddAnnotationCommand(Command):
    """Command to add a new annotation."""
    
    def __init__(self, annotations: List[Dict], annotation: Dict, window_manager):
        self.annotations = annotations
        self.annotation = annotation.copy()  # Create deep copy of annotation
        self.window_manager = window_manager
        self.index = None
    
    def execute(self) -> bool:
        try:
            self.annotations.append(self.annotation)
            self.index = len(self.annotations) - 1
            self.window_manager.update_review_panel(self.annotations)
            return True
        except Exception as e:
            logging.error(f"Error executing AddAnnotationCommand: {str(e)}")
            return False
    
    def undo(self) -> bool:
        try:
            if self.index is not None:
                self.annotations.pop(self.index)
                self.window_manager.update_review_panel(self.annotations)
                return True
            return False
        except Exception as e:
            logging.error(f"Error undoing AddAnnotationCommand: {str(e)}")
            return False
    
    def redo(self) -> bool:
        """Redo the add annotation command."""
        return self.execute()

class DeleteAnnotationCommand(Command):
    """Command to delete an annotation."""
    
    def __init__(self, annotations: List[Dict], index: int, window_manager):
        self.annotations = annotations
        self.index = index
        self.window_manager = window_manager
        self.deleted_annotation = None
    
    def execute(self) -> bool:
        try:
            if 0 <= self.index < len(self.annotations):
                self.deleted_annotation = self.annotations.pop(self.index)
                self.window_manager.update_review_panel(self.annotations)
                return True
            return False
        except Exception as e:
            logging.error(f"Error executing DeleteAnnotationCommand: {str(e)}")
            return False
    
    def undo(self) -> bool:
        try:
            if self.deleted_annotation is not None:
                self.annotations.insert(self.index, self.deleted_annotation)
                self.window_manager.update_review_panel(self.annotations)
                return True
            return False
        except Exception as e:
            logging.error(f"Error undoing DeleteAnnotationCommand: {str(e)}")
            return False
    
    def redo(self) -> bool:
        """Redo the delete annotation command."""
        return self.execute()

class ModifyAnnotationCommand(Command):
    """Command to modify an existing annotation."""
    
    def __init__(self, annotations: List[Dict], index: int, new_state: Dict, window_manager):
        self.annotations = annotations
        self.index = index
        self.new_state = new_state.copy()  # Create deep copy of new state
        self.old_state = None
        self.window_manager = window_manager
    
    def execute(self) -> bool:
        try:
            if 0 <= self.index < len(self.annotations):
                self.old_state = self.annotations[self.index].copy()
                self.annotations[self.index] = self.new_state
                self.window_manager.update_review_panel(self.annotations)
                return True
            return False
        except Exception as e:
            logging.error(f"Error executing ModifyAnnotationCommand: {str(e)}")
            return False
    
    def undo(self) -> bool:
        try:
            if self.old_state is not None:
                self.annotations[self.index] = self.old_state
                self.window_manager.update_review_panel(self.annotations)
                return True
            return False
        except Exception as e:
            logging.error(f"Error undoing ModifyAnnotationCommand: {str(e)}")
            return False
    
    def redo(self) -> bool:
        """Redo the modify annotation command."""
        return self.execute()

# Command Manager
class CommandManager:
    """Manages command execution and undo/redo stacks."""
    
    def __init__(self):
        self.undo_stack: List[Command] = []
        self.redo_stack: List[Command] = []
        self.logger = logging.getLogger(__name__)
    
    def execute(self, command: Command) -> bool:
        """Execute a command and add it to the undo stack."""
        try:
            if command.execute():
                self.undo_stack.append(command)
                self.redo_stack.clear()  # Clear redo stack when new command is executed
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error executing command: {str(e)}")
            return False
    
    def undo(self) -> bool:
        """Undo the last command."""
        try:
            if not self.undo_stack:
                return False
            
            command = self.undo_stack.pop()
            if command.undo():
                self.redo_stack.append(command)
                return True
                
            # If undo failed, don't add to redo stack
            return False
        except Exception as e:
            self.logger.error(f"Error undoing command: {str(e)}")
            return False
    
    def redo(self) -> bool:
        """Redo the last undone command."""
        try:
            if not self.redo_stack:
                return False
            
            command = self.redo_stack.pop()
            if command.redo():
                self.undo_stack.append(command)
                return True
                
            # If redo failed, don't add to undo stack
            return False
        except Exception as e:
            self.logger.error(f"Error redoing command: {str(e)}")
            return False
    
    def clear(self) -> None:
        """Clear both undo and redo stacks."""
        self.undo_stack.clear()
        self.redo_stack.clear()