from typing import Optional, Tuple, List, Callable
import torch
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

class SAMPredictor:
    """Wrapper for SAM prediction functionality."""
    def __init__(self, model_type: str = "vit_h"):
        self.model_type = model_type
        self.predictor = None
        
    def initialize(self, checkpoint_path: str):
        """Initialize the SAM model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def set_image(self, image: np.ndarray) -> None:
        """Set the image for prediction."""
        if self.predictor is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
        self.predictor.set_image(image)
        
    def predict(self, 
                point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                box: Optional[np.ndarray] = None,
                multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks using SAM."""
        if self.predictor is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )