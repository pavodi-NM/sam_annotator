# base_task.py
from abc import ABC, abstractmethod
from typing import Any, Dict
from datetime import datetime

class AnnotationTask(ABC):
    """Abstract base class for annotation tasks."""
    
    def __init__(self):
        self.start_time = None
        self.status = {
            'initialized': False,
            'running': False,
            'completed': False,
            'error': None
        }
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the task with configuration.
        
        Args:
            config: Configuration dictionary for task setup
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process the input data.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up task resources."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate task configuration.
        
        Returns:
            bool: True if configuration is valid
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current task status.
        
        Returns:
            Dict containing task status information
        """
        pass