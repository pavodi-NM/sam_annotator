from abc import ABC, abstractmethod
from typing import Dict, List
import os
import logging

class BaseExporter(ABC):
    """Base class for dataset exporters."""
    
    def __init__(self, dataset_path: str, export_path: str = None):
        self.dataset_path = dataset_path
        self.export_path = export_path or os.path.join(dataset_path, 'exports')
        self.logger = logging.getLogger(__name__)
        
        # Create export directory if it doesn't exist
        os.makedirs(self.export_path, exist_ok=True)
    
    @abstractmethod
    def export(self) -> str:
        """Export the dataset to specific format.
        
        Returns:
            str: Path to exported dataset
        """
        pass
    
    def _get_image_files(self) -> List[str]:
        """Get list of image files in dataset."""
        images_dir = os.path.join(self.dataset_path, 'images')
        return [f for f in os.listdir(images_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def _get_annotation_file(self, image_file: str) -> str:
        """Get corresponding annotation file path."""
        base_name = os.path.splitext(image_file)[0]
        return os.path.join(self.dataset_path, 'labels', f"{base_name}.txt")
        
    def _get_annotation_files(self) -> List[str]:
        """Get list of annotation files in dataset."""
        labels_dir = os.path.join(self.dataset_path, 'labels')
        if not os.path.exists(labels_dir):
            return []
        return [os.path.join(labels_dir, f) for f in os.listdir(labels_dir)
                if f.endswith('.txt')]