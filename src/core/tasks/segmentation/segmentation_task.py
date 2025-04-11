# src/core/tasks/segmentation/segmentation_task.py
from typing import Dict, Any, List, Tuple
from ....core.base_task import BaseAnnotationTask
from ....utils.image_utils import convert_mask_to_polygon
from .sam_predictor import SAMPredictor

class SegmentationTask(BaseAnnotationTask):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.predictor = SAMPredictor(
            model_type=config.get('model_type', 'sam2'),
            weights_path=config.get('weights_path')
        )
        self.points = []
        self.current_mask = None
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize SAM predictor and required resources"""
        self.predictor.initialize()
        
    def process_interaction(self, x: float, y: float, interaction_type: str) -> Dict[str, Any]:
        """
        Process click interactions for SAM segmentation
        
        Returns:
            Dict containing:
            - annotation_complete: bool
            - annotation: Dict (if complete)
            - visualization: numpy.ndarray (current state)
        """
        result = {
            'annotation_complete': False,
            'visualization': None
        }
        
        if interaction_type == 'click':
            self.points.append((x, y))
            # Get prediction from SAM
            self.current_mask = self.predictor.predict(self.points)
            result['visualization'] = self.current_mask
            
        elif interaction_type == 'confirm':
            if self.current_mask is not None:
                # Convert mask to YOLO polygon format
                polygon = convert_mask_to_polygon(self.current_mask)
                result['annotation_complete'] = True
                result['annotation'] = {
                    'type': 'segmentation',
                    'points': polygon,
                    'mask': self.current_mask
                }
                
        return result
    
    def validate_annotation(self, annotation: Dict[str, Any]) -> bool:
        """Validate segmentation annotation"""
        if annotation['type'] != 'segmentation':
            return False
            
        polygon = annotation.get('points', [])
        # Minimum points for a valid polygon
        if len(polygon) < 6:  # At least 3 points (x,y pairs)
            return False
            
        return True
    
    def get_visualization(self, image: Any, annotation: Dict[str, Any]) -> Any:
        """Generate visualization of the segmentation"""
        if annotation.get('mask') is not None:
            return self.predictor.overlay_mask(image, annotation['mask'])
        return image