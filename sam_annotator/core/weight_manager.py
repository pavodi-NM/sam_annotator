import os
import logging
import requests
from tqdm import tqdm
from typing import Dict
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

class SAMWeightManager:
    """Manages SAM model weights for both SAM1 and SAM2.""" 
    CHECKPOINTS = {
        "sam1": {
            "vit_h": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "path": "weights/sam_vit_h_4b8939.pth"
            },
            "vit_l": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "path": "weights/sam_vit_l_0b3195.pth"
            },
            "vit_b": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "path": "weights/sam_vit_b_01ec64.pth"
            }
        },
        "sam2": {
            "tiny": {
                "name": "sam2_t.pt",
                "path": "weights/sam2_t.pt"
            },
            "small": {
                "name": "sam2_s.pt",
                "path": "weights/sam2_s.pt"
            },
            "base": {
                "name": "sam2_b.pt",
                "path": "weights/sam2_b.pt"
            },
            "large": {
                "name": "sam2_l.pt",
                "path": "weights/sam2_l.pt"
            },
            "tiny_v2": {  # SAM 2.1 models
                "name": "sam2.1_t.pt",
                "path": "weights/sam2.1_t.pt"
            },
            "small_v2": {
                "name": "sam2.1_s.pt",
                "path": "weights/sam2.1_s.pt"
            },
            "base_v2": {
                "name": "sam2.1_b.pt",
                "path": "weights/sam2.1_b.pt"
            },
            "large_v2": {
                "name": "sam2.1_l.pt",
                "path": "weights/sam2.1_l.pt"
            }
        }
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.weights_dir = "weights"
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Set ultralytics download directory
        os.environ['ULTRALYTICS_DIR'] = os.path.abspath(self.weights_dir)

    def get_checkpoint_path(self, user_checkpoint_path: str = None, 
                          version: str = "sam1",
                          model_type: str = "vit_h") -> str:
        """
        Get appropriate checkpoint path for specified SAM version.
        
        Args:
            user_checkpoint_path: Optional path specified by user
            version: 'sam1' or 'sam2'
            model_type: For SAM1: 'vit_h', 'vit_l', 'vit_b'
                       For SAM2: 'tiny', 'small', 'base', 'large',
                                'tiny_v2', 'small_v2', 'base_v2', 'large_v2'
        Returns:
            str: Path to the checkpoint file
        """
        
        # If user specified a path, verify it exists
        if user_checkpoint_path and os.path.exists(user_checkpoint_path):
            return user_checkpoint_path
            
        try:
            if version.lower() == "sam1":
                if model_type not in self.CHECKPOINTS["sam1"]:
                    raise ValueError(f"Invalid model type for SAM1: {model_type}")
                checkpoint_info = self.CHECKPOINTS["sam1"][model_type]
                checkpoint_path = checkpoint_info["path"]
                
                if not os.path.exists(checkpoint_path):
                    self.logger.info(f"Downloading SAM1 {model_type} weights...")
                    self._download_checkpoint(checkpoint_info["url"], checkpoint_path)
                    
            elif version.lower() == "sam2":
                if model_type not in self.CHECKPOINTS["sam2"]:
                    raise ValueError(
                        f"Invalid model type for SAM2. Choose from: {', '.join(self.CHECKPOINTS['sam2'].keys())}"
                    )
                    
                checkpoint_info = self.CHECKPOINTS["sam2"][model_type]
                checkpoint_path = checkpoint_info["path"]
                
                # Create symlink if weight exists in default location but not in weights directory
                default_path = os.path.join(os.getcwd(), checkpoint_info["name"])
                if os.path.exists(default_path) and not os.path.exists(checkpoint_path):
                    os.symlink(default_path, checkpoint_path)
                
                # Ultralytics will handle the download automatically when the model is initialized
                
            else:
                raise ValueError(f"Unsupported SAM version: {version}")
                
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Error getting checkpoint path: {str(e)}")
            raise

    def get_available_models(self, version: str = None) -> Dict[str, list]:
        """
        Get available model types for specified version(s).
        
        Args:
            version: Optional 'sam1' or 'sam2'. If None, returns all models.
        Returns:
            Dict containing available model types for each version
        """
        if version:
            if version not in self.CHECKPOINTS:
                raise ValueError(f"Invalid version: {version}")
            return {version: list(self.CHECKPOINTS[version].keys())}
        return {v: list(models.keys()) for v, models in self.CHECKPOINTS.items()}

    def _download_checkpoint(self, url: str, target_path: str) -> None:
        """Download checkpoint from specified URL."""
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(target_path, 'wb') as f, tqdm(
                desc="Downloading weights",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
                    
            self.logger.info(f"Successfully downloaded weights to {target_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to download weights: {str(e)}")
            if os.path.exists(target_path):
                os.remove(target_path)
            raise

    def verify_checkpoint(self, checkpoint_path: str) -> bool:
        """Verify checkpoint file exists and has minimum expected size."""
        if not os.path.exists(checkpoint_path):
            return False
            
        # Check minimum size (SAM models are typically >300MB)
        min_size = 300 * 1024 * 1024  # 300MB
        if os.path.getsize(checkpoint_path) < min_size:
            self.logger.warning(f"Checkpoint file seems too small: {checkpoint_path}")
            return False
            
        return True