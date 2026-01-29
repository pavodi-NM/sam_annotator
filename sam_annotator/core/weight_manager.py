import os
import logging
import requests
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

from sam_annotator.core.model_registry import (
    MODEL_REGISTRY,
    get_version_requirements,
    supports_auto_download,
    is_valid_model_type,
)


class SAMWeightManager:
    """Manages SAM model weights for SAM1, SAM2, and SAM3."""
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
        },
        "sam3": {
            # SAM3 has a single unified model variant
            # Requires manual download from HuggingFace (no auto-download)
            "sam3": {
                "name": "sam3.pt",
                "path": "weights/sam3.pt",
                "manual_download": True,
                "download_url": "https://huggingface.co/facebook/sam3",
                "size_gb": 3.4,
                "min_ultralytics_version": "8.3.237"
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

            elif version.lower() == "sam3":
                if model_type not in self.CHECKPOINTS["sam3"]:
                    raise ValueError(
                        f"Invalid model type for SAM3. Choose from: {', '.join(self.CHECKPOINTS['sam3'].keys())}"
                    )

                checkpoint_info = self.CHECKPOINTS["sam3"][model_type]
                checkpoint_path = checkpoint_info["path"]

                # SAM3 requires manual download - check if file exists
                if not os.path.exists(checkpoint_path):
                    # Also check common alternative locations
                    alt_paths = [
                        os.path.join(os.getcwd(), checkpoint_info["name"]),
                        os.path.join(os.path.expanduser("~"), ".cache", "sam3", checkpoint_info["name"]),
                        os.path.join(os.path.expanduser("~"), "sam3.pt"),
                    ]

                    found_path = None
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            found_path = alt_path
                            break

                    if found_path:
                        # Create symlink to found checkpoint
                        self.logger.info(f"Found SAM3 checkpoint at: {found_path}")
                        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                        os.symlink(found_path, checkpoint_path)
                    else:
                        # Provide helpful error message with download instructions
                        raise FileNotFoundError(
                            f"SAM3 checkpoint not found at: {checkpoint_path}\n\n"
                            "SAM3 requires MANUAL download (no auto-download available):\n"
                            f"1. Visit: {checkpoint_info['download_url']}\n"
                            "2. Request access (requires HuggingFace account)\n"
                            f"3. Download {checkpoint_info['name']} (~{checkpoint_info['size_gb']} GB)\n"
                            f"4. Place in: {self.weights_dir}/{checkpoint_info['name']}\n\n"
                            "Or specify custom path with: --checkpoint /path/to/sam3.pt\n\n"
                            f"Note: Requires ultralytics>={checkpoint_info['min_ultralytics_version']}"
                        )

            else:
                raise ValueError(f"Unsupported SAM version: {version}. Choose from: sam1, sam2, sam3")
                
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

    def check_weights_exist(
        self,
        version: str,
        model_type: str,
        user_checkpoint_path: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if weights exist without throwing exceptions.

        This is used for pre-flight checks before starting the application.

        Args:
            version: SAM version ('sam1', 'sam2', 'sam3')
            model_type: Model type for the version
            user_checkpoint_path: Optional custom path from user

        Returns:
            Tuple of (exists, resolved_path, error_message)
            - exists: True if weights are available or will auto-download
            - resolved_path: Path to weights if found, None otherwise
            - error_message: Error description if not found, None otherwise
        """
        # If user specified a path, just check if it exists
        if user_checkpoint_path:
            if os.path.exists(user_checkpoint_path):
                return True, user_checkpoint_path, None
            else:
                return False, None, f"Specified checkpoint not found: {user_checkpoint_path}"

        # Validate version and model_type
        if version not in self.CHECKPOINTS:
            return False, None, f"Unknown SAM version: {version}"

        if model_type not in self.CHECKPOINTS[version]:
            valid_types = list(self.CHECKPOINTS[version].keys())
            return False, None, f"Invalid model type '{model_type}' for {version}. Valid: {valid_types}"

        checkpoint_info = self.CHECKPOINTS[version][model_type]
        checkpoint_path = checkpoint_info.get("path", "")

        # For auto-download versions, weights will be downloaded automatically
        if supports_auto_download(version):
            # Check if already exists
            if os.path.exists(checkpoint_path):
                return True, checkpoint_path, None
            # Will be downloaded, so return True
            return True, None, None

        # For manual download versions (like SAM3), check if weights exist
        search_paths = self._get_weight_search_paths(version, model_type)

        for path in search_paths:
            if os.path.exists(path):
                return True, path, None

        # Not found
        return False, None, f"{version.upper()} weights not found. Manual download required."

    def _get_weight_search_paths(self, version: str, model_type: str) -> List[str]:
        """Get list of paths to search for weights.

        Args:
            version: SAM version
            model_type: Model type

        Returns:
            List of paths to check for weights
        """
        if version not in self.CHECKPOINTS or model_type not in self.CHECKPOINTS[version]:
            return []

        checkpoint_info = self.CHECKPOINTS[version][model_type]
        weight_name = checkpoint_info.get("name", f"{model_type}.pt")
        default_path = checkpoint_info.get("path", f"weights/{weight_name}")

        paths = [
            default_path,
            os.path.join(os.getcwd(), weight_name),
            os.path.join(os.getcwd(), "weights", weight_name),
            os.path.join(self.weights_dir, weight_name),
        ]

        # Add user home cache paths for SAM3
        if version == "sam3":
            paths.extend([
                os.path.join(os.path.expanduser("~"), ".cache", "sam3", weight_name),
                os.path.join(os.path.expanduser("~"), weight_name),
            ])

        return paths

    def get_weight_info(self, version: str, model_type: str) -> Optional[Dict]:
        """Get weight information for a specific version and model type.

        Args:
            version: SAM version
            model_type: Model type

        Returns:
            Dictionary with weight info, or None if not found
        """
        if version not in self.CHECKPOINTS:
            return None
        if model_type not in self.CHECKPOINTS[version]:
            return None
        return self.CHECKPOINTS[version][model_type].copy()