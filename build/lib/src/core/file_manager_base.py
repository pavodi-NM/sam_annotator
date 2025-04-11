import os
import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class FileManager:
    """Manages file operations for images and annotations."""
    
    def __init__(self, category_path: str, image_processor):
        """Initialize with paths and processor.
        
        Args:
            category_path: Base path for category
            image_processor: Instance of ImageProcessor for image operations
        """
        self.category_path = category_path
        self.image_processor = image_processor
        self.logger = logging.getLogger(__name__)
        
        # Setup paths
        self.setup_paths()
        
        # Auto-save settings
        self.last_auto_save = time.time()
        self.auto_save_interval = 300  # 5 minutes
        
    def setup_paths(self) -> None:
        """Setup directory structure."""
        try:
            # Setup main directories
            self.images_path = os.path.join(self.category_path, 'images')
            self.annotations_path = os.path.join(self.category_path, 'labels')
            self.masks_path = os.path.join(self.category_path, 'masks')
            self.metadata_path = os.path.join(self.category_path, 'metadata')
            
            # Create directories if they don't exist
            for path in [self.annotations_path, self.masks_path, self.metadata_path]:
                os.makedirs(path, exist_ok=True)
                
            self.logger.info("Directory structure setup complete")
            
        except Exception as e:
            self.logger.error(f"Error setting up paths: {str(e)}")
            raise
            
    def load_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Load and process image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple containing processed image and metadata
        """
        try:
            # Load original image
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            # Process image for display with caching
            display_image, metadata = self.image_processor.process_image(original_image)
            
            self.logger.info(f"Loaded image: {image_path}")
            self.logger.info(f"Original size: {metadata['original_size']}")
            self.logger.info(f"Display size: {metadata['display_size']}")
            
            return display_image, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise
            
    def save_annotations(self, annotations: List[Dict], image_path: str) -> bool:
        """Save annotations in multiple formats.
        
        Args:
            annotations: List of annotation dictionaries
            image_path: Path to corresponding image
            
        Returns:
            bool: True if save was successful
        """
        try:
            if not annotations:
                self.logger.warning("No annotations to save!")
                return False
                
            # Get base name for files
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save in multiple formats
            self._save_yolo_format(annotations, base_name)
            self._save_visualization(annotations, image_path, base_name)
            self._save_metadata(annotations, base_name)
            
            self.logger.info(f"Successfully saved {len(annotations)} annotations")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving annotations: {str(e)}")
            return False
            
    def _save_yolo_format(self, annotations: List[Dict], base_name: str) -> None:
        """Save annotations in YOLO format.
        
        Args:
            annotations: List of annotation dictionaries
            base_name: Base name for files
        """
        try:
            label_path = os.path.join(self.annotations_path, f"{base_name}.txt")
            
            with open(label_path, 'w') as f:
                for annotation in annotations:
                    # Use original contour for saving
                    original_contour = annotation['original_contour']
                    
                    # Write class_id and coordinates
                    line = f"{annotation['class_id']}"
                    for point in original_contour:
                        x, y = point[0]
                        x_norm = x / annotation['original_size'][1]  # width
                        y_norm = y / annotation['original_size'][0]  # height
                        line += f" {x_norm:.6f} {y_norm:.6f}"
                    f.write(line + '\n')
                    
            self.logger.info(f"Saved YOLO format annotations to {label_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving YOLO format: {str(e)}")
            raise
            
    def _save_visualization(self, annotations: List[Dict], image_path: str, base_name: str) -> None:
        """Save visualization of annotations.
        
        Args:
            annotations: List of annotation dictionaries
            image_path: Path to original image
            base_name: Base name for files
        """
        try:
            # Load original image
            original_image = cv2.imread(image_path)
            original_height, original_width = original_image.shape[:2]
            
            # Create visualization
            combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            overlay = original_image.copy()
            
            for annotation in annotations:
                # Create mask using original contour
                mask = np.zeros((original_height, original_width), dtype=np.uint8)
                cv2.fillPoly(mask, [annotation['original_contour']], 1)
                
                # Update combined mask
                combined_mask = np.logical_or(combined_mask, mask)
                
                # Create colored overlay
                mask_area = mask > 0
                colored_overlay = overlay.copy()
                colored_overlay[mask_area] = (0, 255, 0)  # Green overlay
                overlay = cv2.addWeighted(overlay, 0.7, colored_overlay, 0.3, 0)
            
            # Convert binary mask to RGB
            combined_mask_rgb = cv2.cvtColor(combined_mask.astype(np.uint8) * 255, 
                                           cv2.COLOR_GRAY2BGR)
            
            # Create side-by-side visualization
            visualization = np.hstack((combined_mask_rgb, overlay))
            
            # Add separator line
            cv2.line(visualization, 
                    (original_width, 0),
                    (original_width, original_height),
                    (0, 0, 255),  # Red line
                    2)
            
            # Save visualization
            vis_path = os.path.join(self.masks_path, 
                                  f"{base_name}_mask{os.path.splitext(image_path)[1]}")
            cv2.imwrite(vis_path, visualization)
            
            self.logger.info(f"Saved visualization to {vis_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving visualization: {str(e)}")
            raise
            
    def _save_metadata(self, annotations: List[Dict], base_name: str) -> None:
        """Save annotation metadata.
        
        Args:
            annotations: List of annotation dictionaries
            base_name: Base name for files
        """
        try:
            metadata = {
                'num_annotations': len(annotations),
                'class_distribution': {},
                'timestamp': datetime.now().isoformat(),
                'image_dimensions': annotations[0]['original_size'] if annotations else None
            }
            
            # Count instances of each class
            for annotation in annotations:
                class_id = str(annotation['class_id'])
                metadata['class_distribution'][class_id] = \
                    metadata['class_distribution'].get(class_id, 0) + 1
            
            # Save metadata
            metadata_path = os.path.join(self.metadata_path, f"{base_name}.txt")
            with open(metadata_path, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                    
            self.logger.info(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")
            raise
            
    def load_annotations(self, image_path: str) -> List[Dict]:
        """Load existing annotations for image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        try:
            # Get label path
            label_path = self.get_label_path(image_path)
            if not os.path.exists(label_path):
                return annotations
                
            # Get image dimensions
            original_image = cv2.imread(image_path)
            orig_height, orig_width = original_image.shape[:2]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    
                    # Convert normalized coordinates to pixel space
                    points = []
                    for i in range(1, len(parts), 2):
                        x = float(parts[i]) * orig_width
                        y = float(parts[i + 1]) * orig_height
                        points.append([[int(x), int(y)]])
                        
                    original_contour = np.array(points, dtype=np.int32)
                    
                    # Scale for display
                    display_contour = self.image_processor.scale_to_display(
                        original_contour, 'contour')
                        
                    annotations.append({
                        'class_id': class_id,
                        'contour_points': display_contour,
                        'original_contour': original_contour,
                        'original_size': (orig_height, orig_width)
                    })
                    
            self.logger.info(f"Loaded {len(annotations)} annotations from {label_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
            
        return annotations
        
    def get_image_files(self) -> List[str]:
        """Get list of image files in directory.
        
        Returns:
            List of image file names
        """
        try:
            image_files = [f for f in os.listdir(self.images_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files.sort()
            return image_files
            
        except Exception as e:
            self.logger.error(f"Error getting image files: {str(e)}")
            return []
            
    def export_dataset(self, format: str) -> str:
        """Export dataset in specified format.
        
        Args:
            format: Export format ('coco' or 'yolo')
            
        Returns:
            Path to exported dataset
        """
        try:
            if format.lower() == 'coco':
                from ..data.exporters.coco_exporter import CocoExporter
                exporter = CocoExporter(self.category_path)
            elif format.lower() == 'yolo':
                from ..data.exporters.yolo_exporter import YoloExporter
                exporter = YoloExporter(self.category_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            export_path = exporter.export()
            self.logger.info(f"Exported dataset to {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Error exporting dataset: {str(e)}")
            raise
            
    def get_label_path(self, image_path: str) -> str:
        """Get corresponding label file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Path to label file
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.annotations_path, f"{base_name}.txt")
        
    def auto_save(self, annotations: List[Dict], image_path: str) -> None:
        """Auto-save annotations periodically.
        
        Args:
            annotations: List of annotation dictionaries
            image_path: Path to image file
        """
        try:
            current_time = time.time()
            if current_time - self.last_auto_save >= self.auto_save_interval:
                if self.save_annotations(annotations, image_path):
                    self.last_auto_save = current_time
                    self.logger.info("Auto-saved annotations")
                    
        except Exception as e:
            self.logger.error(f"Error during auto-save: {str(e)}")