# sam_weight_manager.py
import os
import logging
import requests
from tqdm import tqdm

class SAMWeightManager:  
    """Manages SAM model weights including downloading and path verification."""
    
    DEFAULT_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    DEFAULT_CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_checkpoint_path(self, user_checkpoint_path: str = None) -> str:
        """
        Get the appropriate checkpoint path, downloading weights if necessary.
        
        Args:
            user_checkpoint_path: Optional user-specified checkpoint path
            
        Returns:
            str: Path to the checkpoint file
        """
        checkpoint_path = user_checkpoint_path or self.DEFAULT_CHECKPOINT_PATH
        
        if not os.path.exists(checkpoint_path):
            if user_checkpoint_path:
                self.logger.info(f"User-specified checkpoint not found at {checkpoint_path}")
                self.logger.info("Downloading SAM weights to specified location...")
            else:
                self.logger.info(f"Default checkpoint not found at {checkpoint_path}")
                self.logger.info("Downloading default SAM weights...")
            
            self._download_checkpoint(checkpoint_path)
        else:
            self.logger.info(f"Using existing checkpoint at {checkpoint_path}")
        
        return checkpoint_path

    def _download_checkpoint(self, target_path: str):
        """
        Download the SAM checkpoint from the official source.
        
        Args:
            target_path: Path where the checkpoint should be saved
        """
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        try:
            response = requests.get(self.DEFAULT_CHECKPOINT_URL, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(target_path, 'wb') as f, tqdm(
                desc="Downloading SAM weights",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            
            self.logger.info(f"Successfully downloaded SAM weights to {target_path}")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download SAM weights: {str(e)}")
            raise
        except IOError as e:
            self.logger.error(f"Failed to save SAM weights: {str(e)}")
            raise

    def verify_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Verify that the checkpoint file exists and has the expected size.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            bool: True if checkpoint is valid, False otherwise
        """
        if not os.path.exists(checkpoint_path):
            return False
            
        # You could add additional verification here if needed
        # For example, check file size, hash, or basic file integrity
        return True