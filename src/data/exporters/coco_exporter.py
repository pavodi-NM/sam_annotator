import cv2
import json
import numpy as np
import os
import shutil
import logging
from datetime import datetime
from typing import List, Dict, Any
from .base_exporter import BaseExporter

class CocoExporter(BaseExporter):
    """Exports dataset to COCO format."""
    
    def __init__(self, dataset_path: str):
        """Initialize the COCO exporter."""
        super().__init__(dataset_path)
        self.logger = logging.getLogger(__name__)
    
    def _get_annotated_images(self) -> List[str]:
        """Get list of images that have corresponding annotation files."""
        image_files = self._get_image_files()
        annotated_images = []
        
        for image_file in image_files:
            annotation_file = self._get_annotation_file(image_file)
            if os.path.exists(annotation_file):
                # Check if annotation file is not empty
                if os.path.getsize(annotation_file) > 0:
                    annotated_images.append(image_file)
        
        return annotated_images
    
    def _parse_yolo_line(self, line: str, image_width: int, image_height: int) -> Dict[str, Any]:
        """Parse a line from YOLO format and convert to absolute coordinates."""
        parts = line.strip().split()
        class_id = int(parts[0])
        points = []
        
        # Parse normalized coordinates and convert to absolute pixels
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                x = float(parts[i]) * image_width
                y = float(parts[i + 1]) * image_height
                points.append([x, y])
        
        return {
            'class_id': class_id,
            'points': points
        }
    
    def export(self) -> str:
        """Export dataset to COCO format."""
        try:
            # Create export directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = os.path.join(self.dataset_path, 'exports', f'coco_export_{timestamp}')
            os.makedirs(export_dir, exist_ok=True)
            
            # Create images directory in export
            images_export_dir = os.path.join(export_dir, 'images')
            os.makedirs(images_export_dir, exist_ok=True)
            
            coco_data = {
                'info': {
                    'description': 'SAM Annotator Export',
                    'date_created': datetime.now().isoformat()
                },
                'images': [],
                'annotations': [],
                'categories': []
            }
            
            # Get list of annotated images
            annotated_images = self._get_annotated_images()
            self.logger.info(f"Found {len(annotated_images)} annotated images")
            
            if not annotated_images:
                self.logger.warning("No annotated images found!")
                return export_dir
            
            # Load class names from annotations
            class_ids = set()
            for image_file in annotated_images:
                annotation_file = self._get_annotation_file(image_file)
                with open(annotation_file, 'r') as f:
                    for line in f:
                        try:
                            class_id = int(line.strip().split()[0])
                            class_ids.add(class_id)
                        except (ValueError, IndexError):
                            continue
            
            # Create categories
            for class_id in sorted(class_ids):
                coco_data['categories'].append({
                    'id': class_id,
                    'name': f'class_{class_id}',
                    'supercategory': 'object'
                })
            
            # Process each annotated image
            ann_id = 0
            for img_id, image_file in enumerate(annotated_images):
                try:
                    image_path = os.path.join(self.dataset_path, 'images', image_file)
                    
                    # Copy image to export directory
                    shutil.copy2(image_path, images_export_dir)
                    
                    # Get image info
                    img = cv2.imread(image_path)
                    if img is None:
                        self.logger.warning(f"Could not read image: {image_path}")
                        continue
                    
                    height, width = img.shape[:2]
                    
                    coco_data['images'].append({
                        'id': img_id,
                        'file_name': image_file,
                        'height': height,
                        'width': width
                    })
                    
                    # Process annotations
                    annotation_file = self._get_annotation_file(image_file)
                    with open(annotation_file, 'r') as f:
                        for line in f:
                            try:
                                # Parse YOLO format line
                                ann_data = self._parse_yolo_line(line, width, height)
                                points = ann_data['points']
                                
                                if len(points) > 2:  # Ensure we have at least a triangle
                                    # Convert points to numpy array for contour operations
                                    contour = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
                                    
                                    # Calculate bounding box
                                    x, y, w, h = cv2.boundingRect(contour)
                                    
                                    # Create segmentation
                                    segmentation = []
                                    for point in points:
                                        segmentation.extend([float(point[0]), float(point[1])])
                                    
                                    # Format for COCO
                                    coco_data['annotations'].append({
                                        'id': ann_id,
                                        'image_id': img_id,
                                        'category_id': ann_data['class_id'],
                                        'segmentation': [segmentation],
                                        'area': float(cv2.contourArea(contour)),
                                        'bbox': [float(x), float(y), float(w), float(h)],
                                        'iscrowd': 0
                                    })
                                    ann_id += 1
                                    
                            except Exception as e:
                                self.logger.warning(f"Error processing annotation in {annotation_file}: {str(e)}")
                                continue
                    
                except Exception as e:
                    self.logger.warning(f"Error processing image {image_file}: {str(e)}")
                    continue
            
            # Save COCO JSON
            json_path = os.path.join(export_dir, 'annotations.json')
            with open(json_path, 'w') as f:
                json.dump(coco_data, f, indent=4)
            
            self.logger.info(f"Exported {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
            return export_dir
            
        except Exception as e:
            self.logger.error(f"Error during COCO export: {str(e)}")
            raise