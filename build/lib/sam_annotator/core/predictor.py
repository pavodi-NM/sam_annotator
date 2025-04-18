from typing import Optional, Tuple, List, Dict
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import SAM as SAM2 
import numpy as np
import logging
import hashlib
from functools import lru_cache
import gc
import psutil

from .base_predictor import BaseSAMPredictor
from .memory_manager import GPUMemoryManager
from ..utils.visualization import VisualizationManager

            
class SAM1Predictor(BaseSAMPredictor):
    """Original SAM implementation with advanced memory management."""
    
    def __init__(self, model_type: str = "vit_h"):
        self.model_type = model_type
        self.predictor = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory manager
        self.memory_manager = GPUMemoryManager()
        
        # Cache settings
        self.current_image_embedding = None
        self.current_image_hash = None
        self.prediction_cache = {}
        self.max_cache_size = 50
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            self.logger.warning("Running on CPU. Performance may be limited.")
            
    def initialize(self, checkpoint_path: str) -> None:
        """Initialize the SAM model with memory optimizations."""
        try:
            # Check available GPU memory before loading
            if self.device.type == 'cuda':
                memory_info = self.memory_manager.get_gpu_memory_info()
                available_memory = memory_info['total'] - memory_info['used']
                
                # Estimate if we have enough memory (SAM typically needs ~10GB)
                if available_memory < 10 * (1024 ** 3):  # 10GB in bytes
                    self.logger.warning("Limited GPU memory available. Performance may be affected.")
            
            # Load model with memory optimizations
            sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            
            # Move model to device and optimize
            sam.to(device=self.device)
            
            if self.device.type == 'cuda':
                # Enable memory optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            
            # Initialize predictor
            self.predictor = SamPredictor(sam)
            self.logger.info(f"Initialized SAM model on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error initializing SAM model: {str(e)}")
            raise

    def set_image(self, image: np.ndarray) -> None:
        """Set image with embedding caching."""
        if self.predictor is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
            
        try:
            # Calculate image hash
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            
            # Only compute new embedding if image changed
            if image_hash != self.current_image_hash:
                with torch.no_grad():
                    self.predictor.set_image(image)
                    self.current_image_hash = image_hash
                    
                # Clear prediction cache for new image
                self.prediction_cache.clear()
                
                # Check memory status after setting image
                status_ok, message = self.memory_manager.check_memory_status()
                if not status_ok:
                    self.logger.warning(message)
                    self.memory_manager.optimize_memory()
                    
        except Exception as e:
            self.logger.error(f"Error setting image: {str(e)}")
            raise

    def predict(self,
               point_coords: Optional[np.ndarray] = None,
               point_labels: Optional[np.ndarray] = None,
               box: Optional[np.ndarray] = None,
               multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks with memory management."""
        if self.predictor is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
            
        try:
            # Check memory status before prediction
            status_ok, message = self.memory_manager.check_memory_status()
            if not status_ok:
                self.logger.error(message)
                raise RuntimeError(message)
                
            # Generate cache key and check cache
            cache_key = self._generate_cache_key(point_coords, point_labels, box)
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            
            # Run prediction with optimizations
            with torch.no_grad():
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=multimask_output
                )
                
                # Cache results if memory allows
                memory_info = self.memory_manager.get_gpu_memory_info()
                if memory_info['utilization'] < self.memory_manager.warning_threshold:
                    self.prediction_cache[cache_key] = (masks, scores, logits)
                    
                    # Manage cache size
                    if len(self.prediction_cache) > self.max_cache_size:
                        self.clear_cache(keep_current=True)
                
                return masks, scores, logits
                
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            # Try to recover memory
            self.memory_manager.optimize_memory(force=True)
            raise
            
    def clear_cache(self, keep_current: bool = False) -> None:
        """Clear prediction cache with option to keep current image."""
        if keep_current and self.current_image_hash:
            current_predictions = {k: v for k, v in self.prediction_cache.items()
                                if k.startswith(str(self.current_image_hash))}
            self.prediction_cache.clear()
            self.prediction_cache.update(current_predictions)
        else:
            self.prediction_cache.clear()
            
        # Force memory optimization
        self.memory_manager.optimize_memory(force=True)
        
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage ratio."""
        return self.memory_manager.get_gpu_memory_info()['utilization']
        
    def _generate_cache_key(self,
                          point_coords: Optional[np.ndarray],
                          point_labels: Optional[np.ndarray],
                          box: Optional[np.ndarray]) -> str:
        """Generate cache key for prediction inputs."""
        key_parts = [str(self.current_image_hash)]
        
        if point_coords is not None:
            key_parts.append(hashlib.md5(point_coords.tobytes()).hexdigest())
        if point_labels is not None:
            key_parts.append(hashlib.md5(point_labels.tobytes()).hexdigest())
        if box is not None:
            key_parts.append(hashlib.md5(box.tobytes()).hexdigest())
            
        return "_".join(key_parts)
    
    

class SAM2Predictor(BaseSAMPredictor):
    """Ultralytics SAM2 implementation with memory management."""
    
    def __init__(self, model_type: str = "base"):
        super().__init__()
        self.model_type = model_type
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory manager
        self.memory_manager = GPUMemoryManager()
        
        # Cache settings
        self.current_image = None
        self.current_image_hash = None
        self.prediction_cache = {}
        self.max_cache_size = 50
        self.current_results = None
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            self.logger.warning("Running on CPU. Performance may be limited.")
            
    def initialize(self, checkpoint_path: str) -> None:
        """Initialize SAM2 model with memory optimizations."""
        try:
            # Check available GPU memory before loading
            if self.device.type == 'cuda':
                memory_info = self.memory_manager.get_gpu_memory_info()
                available_memory = memory_info['total'] - memory_info['used']
                
                if available_memory < 4 * (1024 ** 3):  # 4GB threshold for SAM2
                    self.logger.warning("Limited GPU memory available. Performance may be affected.")
            
            self.model = SAM2(checkpoint_path)
            if self.device.type == 'cuda':
                # Enable memory optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            
            self.logger.info(f"Initialized SAM2 model on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error initializing SAM2: {str(e)}")
            raise
            
    def predict(self, 
                point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                box: Optional[np.ndarray] = None,
                multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks with memory management and caching."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        try:
            # Check memory status before prediction
            status_ok, message = self.memory_manager.check_memory_status()
            if not status_ok:
                self.logger.error(message)
                raise RuntimeError(message)
            
            with torch.no_grad():
                if box is not None:
                    # Format box for SAM2 (simpler approach matching working implementation)
                    results = self.model(
                        source=self.current_image,
                        bboxes=[box.tolist()]  # Just pass the box directly
                    )
                elif point_coords is not None and point_labels is not None:
                    # Format points for SAM2
                    points = []
                    for coord, label in zip(point_coords, point_labels):
                        points.append([float(coord[0]), float(coord[1]), int(label)])
                    
                    results = self.model(
                        source=self.current_image,
                        points=[points]  # Wrap points in list
                    )
                else:
                    raise ValueError("Either points with labels or box must be provided")
                    
                # Store results for potential reuse
                self.current_results = results
                
                # Convert results to match SAM1 format
                if len(results) > 0 and results[0].masks is not None:
                    # Get masks and ensure they're in the correct format
                    masks = results[0].masks.data.cpu().numpy()
                    
                    # Handle single mask case
                    if len(masks.shape) == 2:
                        masks = np.expand_dims(masks, 0)
                    
                    # Use confidence scores if available, otherwise use ones
                    scores = results[0].conf.cpu().numpy() if hasattr(results[0], 'conf') else \
                            np.ones(len(masks))
                            
                    # Create placeholder logits
                    logits = np.ones((len(masks), 1))
                else:
                    masks = np.zeros((1, self.current_image.shape[0], self.current_image.shape[1]), dtype=bool)
                    scores = np.array([0.0])
                    logits = np.array([[0.0]])
                
                return masks, scores, logits
                
        except Exception as e:
            self.logger.error(f"Error in SAM2 prediction: {str(e)}")
            self.memory_manager.optimize_memory(force=True)
            raise
            
    def set_image(self, image: np.ndarray) -> None:
        """Store image with caching."""
        try:
            # Calculate image hash
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            
            # Only update if image changed
            if image_hash != self.current_image_hash:
                self.current_image = image
                self.current_image_hash = image_hash
                self.prediction_cache.clear()
                
        except Exception as e:
            self.logger.error(f"Error setting image in SAM2: {str(e)}")
            raise
            
    def clear_cache(self, keep_current: bool = False) -> None:
        """Clear prediction cache with option to keep current image."""
        if keep_current and self.current_image_hash:
            current_predictions = {k: v for k, v in self.prediction_cache.items()
                                if k.startswith(str(self.current_image_hash))}
            self.prediction_cache.clear()
            self.prediction_cache.update(current_predictions)
        else:
            self.prediction_cache.clear()
            
        # Force memory optimization
        self.memory_manager.optimize_memory(force=True)
        
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage ratio."""
        return self.memory_manager.get_gpu_memory_info()['utilization']
        
    def _generate_cache_key(self,
                          point_coords: Optional[np.ndarray],
                          point_labels: Optional[np.ndarray],
                          box: Optional[np.ndarray]) -> str:
        """Generate cache key for prediction inputs."""
        key_parts = [str(self.current_image_hash)]
        
        if point_coords is not None:
            key_parts.append(hashlib.md5(point_coords.tobytes()).hexdigest())
        if point_labels is not None:
            key_parts.append(hashlib.md5(point_labels.tobytes()).hexdigest())
        if box is not None:
            key_parts.append(hashlib.md5(box.tobytes()).hexdigest())
            
        return "_".join(key_parts)
        """Get additional information from the latest prediction."""
        if self.current_results is None:
            return {}
            
        try:
            result = self.current_results[0]
            return {
                'confidence': result.conf.cpu().numpy() if hasattr(result, 'conf') else None,
                'boxes': result.boxes.data.cpu().numpy() if hasattr(result, 'boxes') else None,
            }
        except Exception as e:
            self.logger.error(f"Error getting additional info: {str(e)}")
            return {}