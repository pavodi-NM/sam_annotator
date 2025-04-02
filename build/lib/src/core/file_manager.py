import os
import logging
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import cv2
import numpy as np
import pandas as pd 
import yaml
from threading import Lock
import time


# Project-specific imports
from ..data.dataset_manager import DatasetManager  # From same directory as FileManager
from ..utils.image_utils import ImageProcessor  # From utils directory
from ..utils.visualization import VisualizationManager  # From utils directory

# Optional imports for complex type hints
from typing_extensions import TypedDict  

class ImageMetadata(TypedDict):
    dimensions: Tuple[int, int, int]
    size: int
    modified: float
    
    
class FileManager:
    """High-level coordinator for file operations that delegates to DatasetManager."""
    
    def __init__(self, category_path: str, logger: Optional[logging.Logger] = None):
        """Initialize the file manager with base directory path."""
        self.logger = logger or logging.getLogger(__name__)
        self.base_path = Path(category_path)
        #self.category_path = category_path
        
        # Initialize DatasetManager for handling annotations and caching
        self.dataset_manager = DatasetManager(str(category_path))
        
        # Setup directory structure
        self.structure = {
            'images': self.base_path / 'images',
            'labels': self.base_path / 'labels',
            'masks': self.base_path / 'masks',
            'metadata': self.base_path / 'metadata',
            'exports': self.base_path / 'exports',
            'backups': self.base_path / 'backups'
        }
        
        # Initialize directory structure
        self._initialize_directories()
        
    
    def _initialize_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        try:
            for dir_path in self.structure.values():
                dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info("Directory structure initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize directory structure: {str(e)}")
            raise
        
    
    def load_classes(self, classes_csv: str) -> List[str]:
        """Load class names from CSV file."""
        try:
            df = pd.read_csv(classes_csv)
            class_names = df['class_name'].tolist()[:15]  # Limit to 15 classes
            self.logger.info(f"Loaded {len(class_names)} classes from {classes_csv}")
            return class_names
        except Exception as e:
            self.logger.error(f"Error loading classes from {classes_csv}: {str(e)}")
            raise
        
        
    def get_image_list(self) -> List[str]:
        """Get list of valid images by delegating to DatasetManager."""
        try:
            # Use DatasetManager's image caching capabilities
            image_files = [f for f in os.listdir(self.structure['images'])
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files.sort()
            return image_files
        except Exception as e:
            self.logger.error(f"Error getting image list: {str(e)}")
            return []

    def _scale_annotations(self, 
                         annotations: List[Dict], 
                         original_dimensions: Tuple[int, int],
                         display_dimensions: Tuple[int, int]) -> List[Dict]:
        
        """Scale annotations from original to display dimensions."""
        self.logger.info(f"Scaling annotations from {original_dimensions} to {display_dimensions}")
        orig_height, orig_width = original_dimensions
        display_height, display_width = display_dimensions
        
        scale_x = display_width / orig_width
        scale_y = display_height / orig_height
        
        self.logger.info(f"Using scale factors: scale_x={scale_x}, scale_y={scale_y}")
        
        scaled_annotations = []
        for idx, ann in enumerate(annotations):
            try:
                # Check if contour_points exists and has valid data
                if 'contour_points' not in ann or ann['contour_points'] is None:
                    self.logger.warning(f"Missing contour_points in annotation {idx}")
                    continue
                
                # Clone and scale contour points to display dimensions
                contour = ann['contour_points'].copy()
                contour = contour.astype(np.float32)
                contour[:, :, 0] *= scale_x
                contour[:, :, 1] *= scale_y
                contour = contour.astype(np.int32)
                
                # Calculate display box directly from the scaled contour
                x, y, w, h = cv2.boundingRect(contour)
                display_box = [x, y, x + w, y + h]
                
                # Create mask at display size
                mask = np.zeros((display_height, display_width), dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 1)
                
                # Verify mask is valid and contains data
                mask_sum = np.sum(mask)
                if mask_sum == 0:
                    self.logger.warning(f"Generated empty mask for annotation {idx}")
                    
                    # Try to create a simpler mask from the bounding box as fallback
                    if 'box' in ann:
                        self.logger.info(f"Attempting to create mask from bounding box for annotation {idx}")
                        box = ann['box']
                        # Scale box to display dimensions
                        scaled_box = [
                            int(box[0] * scale_x),
                            int(box[1] * scale_y),
                            int(box[2] * scale_x), 
                            int(box[3] * scale_y)
                        ]
                        # Create a mask from the box
                        cv2.rectangle(mask, 
                                    (scaled_box[0], scaled_box[1]), 
                                    (scaled_box[2], scaled_box[3]), 
                                    1, -1)  # -1 means filled rectangle
                
                # Convert to boolean mask
                mask_bool = mask.astype(bool)
                
                # Create flattened contour list for visualization
                contour_list = contour.tolist()
                flattened_contour = []
                for point in contour_list:
                    if len(point) == 1 and isinstance(point[0], list) and len(point[0]) == 2:
                        flattened_contour.append(point[0])
                    else:
                        flattened_contour.append(point)
                
                # Calculate area
                area = cv2.contourArea(contour)
                
                # Ensure we have class_name
                class_name = ann.get('class_name', f'Class {ann["class_id"]}')
                
                # Create the annotation with all required fields
                scaled_annotation = {
                    'id': ann.get('id', len(scaled_annotations)),
                    'class_id': ann['class_id'],
                    'class_name': class_name,
                    'mask': mask_bool,
                    'contour_points': contour,  # Original cv2 contour format
                    'contour': flattened_contour,  # Flattened points for new format
                    'box': ann.get('box', [0, 0, 0, 0]),  # Original box
                    'display_box': display_box,  # Box in display coordinates
                    'area': area,
                    'original_contour': ann['contour_points'],  # Keep original contour
                    'metadata': ann.get('metadata', {
                        'annotation_mode': 'imported',
                        'timestamp': time.time()
                    })
                }
                
                # Log info about the scaled annotation
                self.logger.info(f"Created scaled annotation {idx+1}: " +
                                f"class_id={scaled_annotation['class_id']}, " +
                                f"mask_size={mask.shape}, " +
                                f"has_data={np.sum(mask) > 0}, " +
                                f"display_box={display_box}")
                
                scaled_annotations.append(scaled_annotation)
                
            except Exception as e:
                self.logger.error(f"Error scaling annotation {idx}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
        
        self.logger.info(f"Scaled {len(scaled_annotations)} annotations successfully")
        return scaled_annotations

    def load_annotations(self, 
                        image_path: str,
                        original_dimensions: Tuple[int, int],
                        display_dimensions: Tuple[int, int]) -> List[Dict]:
        """Delegate annotation loading to DatasetManager."""
        try:
            # Let DatasetManager handle the actual loading with its caching
            annotations = self.dataset_manager.load_annotations(image_path)
            
            if not annotations:
                return []
                
            # Scale annotations if needed
            if original_dimensions != display_dimensions:
                annotations = self._scale_annotations(
                    annotations,
                    original_dimensions,
                    display_dimensions
                )
                
            return annotations
            
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
            return []

    def save_annotations(self,
                        annotations: List[Dict],
                        image_name: str,
                        original_dimensions: Tuple[int, int],
                        display_dimensions: Tuple[int, int],
                        class_names: List[str],
                        save_visualization: bool = True) -> bool:
        """Save annotations with proper scaling, visualization, and metadata."""
        try:
            # Get base paths
            base_name = os.path.splitext(image_name)[0]
            original_ext = os.path.splitext(image_name)[1]
            
            # Setup directories
            masks_dir = self.structure['masks']
            metadata_dir = self.structure['metadata']
            original_height, original_width = original_dimensions
            
            # 1. Save normalized contour coordinates using DatasetManager
            label_path = self.structure['labels'] / f"{base_name}.txt"
            
            # Calculate scale factors
            scale_x = original_width / display_dimensions[1]
            scale_y = original_height / display_dimensions[0]
            
            with open(label_path, 'w') as f:
                for annotation in annotations:
                    # Scale contour points to original image space
                    display_contour = annotation['contour_points']
                    original_contour = display_contour.copy()
                    original_contour = original_contour.astype(np.float32)
                    original_contour[:, :, 0] *= scale_x
                    original_contour[:, :, 1] *= scale_y
                    original_contour = original_contour.astype(np.int32)
                    
                    # Write normalized coordinates
                    line = f"{annotation['class_id']}"
                    for point in original_contour:
                        x, y = point[0]
                        x_norm = x / original_width
                        y_norm = y / original_height
                        line += f" {x_norm:.6f} {y_norm:.6f}"
                    f.write(line + '\n')
            
            if save_visualization:
                # 2. Create visualization
                original_image = cv2.imread(str(self.structure['images'] / image_name))
                combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)
                overlay = original_image.copy()
                
                for annotation in annotations:
                    # Scale and create visualization
                    display_contour = annotation['contour_points']
                    original_contour = display_contour.copy()
                    original_contour = original_contour.astype(np.float32)
                    original_contour[:, :, 0] *= scale_x
                    original_contour[:, :, 1] *= scale_y
                    original_contour = original_contour.astype(np.int32)
                    
                    mask = np.zeros((original_height, original_width), dtype=np.uint8)
                    cv2.fillPoly(mask, [original_contour], 1)
                    combined_mask = np.logical_or(combined_mask, mask)
                    
                    # Create overlay
                    mask_area = mask > 0
                    green_overlay = overlay.copy()
                    green_overlay[mask_area] = (0, 255, 0)
                    overlay = cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0)
                
                # Create side-by-side visualization
                combined_mask_rgb = cv2.cvtColor(combined_mask.astype(np.uint8) * 255, 
                                            cv2.COLOR_GRAY2BGR)
                side_by_side = np.hstack((combined_mask_rgb, overlay))
                
                # Add separator line
                separator_x = original_width
                cv2.line(side_by_side, 
                        (separator_x, 0), 
                        (separator_x, original_height),
                        (0, 0, 255),
                        2)
                
                # Save visualization
                mask_path = masks_dir / f"{base_name}_mask{original_ext}"
                cv2.imwrite(str(mask_path), side_by_side)
            
            # 3. Save metadata
            metadata = {
                'num_annotations': len(annotations),
                'class_distribution': {str(i): 0 for i in range(len(class_names))},
                'image_dimensions': {
                    'original': (original_width, original_height),
                    'display': display_dimensions
                },
                'scale_factors': {
                    'x': scale_x,
                    'y': scale_y
                }
            }
            
            # Count instances of each class
            for annotation in annotations:
                class_id = str(annotation['class_id'])
                metadata['class_distribution'][class_id] = \
                    metadata['class_distribution'].get(class_id, 0) + 1
            
            metadata_path = metadata_dir / f"{base_name}.txt"
            with open(metadata_path, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving annotations: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        
    def handle_export(self, format: str, class_names: List[str]) -> Optional[str]:
        """Handle dataset export in various formats."""
        try:
            self.logger.info(f"Starting export to {format} format")  # Debug log
            base_path = str(self.base_path)
            self.logger.info(f"Base path: {base_path}")  # Debug log

            exporter = None
            if format.lower() == 'coco':
                from ..data.exporters.coco_exporter import CocoExporter
                self.logger.info("Initializing COCO exporter")  # Debug log
                exporter = CocoExporter(base_path)
            elif format.lower() == 'yolo':
                from ..data.exporters.yolo_exporter import YoloExporter
                self.logger.info("Initializing YOLO exporter")  # Debug log
                exporter = YoloExporter(base_path)
            elif format.lower() == 'pascal':
                from ..data.exporters.pascal_exporter import PascalVOCExporter
                self.logger.info("Initializing Pascal VOC exporter")  # Debug log
                exporter = PascalVOCExporter(base_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            if exporter is None:
                raise ValueError(f"Failed to initialize exporter for format: {format}")

            # Perform export
            self.logger.info("Starting export operation")  # Debug log
            export_path = exporter.export()
            self.logger.info(f"Export completed. Path: {export_path}")  # Debug log
            return export_path

        except ImportError as e:
            self.logger.error(f"Import error during export: {str(e)}")
            self.logger.error("Make sure all required exporter modules are available")
            return None
        except Exception as e:
            self.logger.error(f"Error exporting dataset: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")  # Add full traceback
            return None

    def get_last_annotated_index(self) -> int:
        """Use DatasetManager's dataset info to find last annotated image."""
        try:
            dataset_info = self.dataset_manager.load_dataset_info()
            image_files = self.get_image_list()
            
            for idx, image_file in enumerate(image_files):
                image_path = str(self.structure['images'] / image_file)
                if not self.dataset_manager.load_annotations(image_path):
                    return max(0, idx)
                    
            return max(0, len(image_files) - 1)
            
        except Exception as e:
            self.logger.error(f"Error finding last annotated image: {str(e)}")
            return 0

    def clear_cache(self, current_image_path: Optional[str] = None) -> None:
        """Delegate cache clearing to DatasetManager."""
        self.dataset_manager.clear_cache(current_image_path)

    def create_backup(self) -> bool:
        """Delegate backup creation to DatasetManager."""
        try:
            backup_path = self.dataset_manager.create_backup()
            return bool(backup_path)
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            return False
        
        
