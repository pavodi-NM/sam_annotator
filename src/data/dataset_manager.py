from typing import List, Dict, Optional, Tuple
import os
import json
import logging
import shutil
import time
from pathlib import Path
import cv2
import numpy as np

class DatasetManager:
    """Manages dataset operations, including import/export and auto-save."""
    
    def __init__(self, dataset_path: str):
        """Initialize dataset manager with the root dataset path."""
        
    
        self.dataset_path = dataset_path
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Ensure required directories exist
        self.images_path = os.path.join(dataset_path, 'images')
        self.labels_path = os.path.join(dataset_path, 'labels')
        self.masks_path = os.path.join(dataset_path, 'masks')
        self.metadata_path = os.path.join(dataset_path, 'metadata')
        
        for path in [self.labels_path, self.masks_path, self.metadata_path]:
            os.makedirs(path, exist_ok=True)
        
        # Auto-save settings
        self.auto_save_enabled = True
        self.auto_save_interval = 300  # 5 minutes
        self.last_auto_save = time.time()
        
        # Backup settings
        self.max_backups = 5
        self.backup_interval = 3600  # 1 hour
        
        # Cache for faster access
        self.image_cache = {}
        self.annotation_cache = {}
        
    def setup_directory_structure(self) -> None:
        """Create necessary directory structure."""
        dirs = ['images', 'labels', 'masks', 'metadata', 'exports', 'backups']
        for dir_name in dirs:
            os.makedirs(os.path.join(self.dataset_path, dir_name), exist_ok=True)
    
    def load_dataset_info(self) -> Dict:
        """Load dataset information and statistics."""
        info_path = os.path.join(self.dataset_path, 'dataset_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        return self._generate_dataset_info()
    
    def _generate_dataset_info(self) -> Dict:
        """Generate dataset information and statistics."""
        info = {
            'total_images': 0,
            'total_annotations': 0,
            'classes': {},
            'last_modified': time.time(),
            'creation_date': time.time()
        }
        
        try:
            # Count images
            image_dir = os.path.join(self.dataset_path, 'images')
            info['total_images'] = len([f for f in os.listdir(image_dir)
                                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            # Count annotations and class distribution
            labels_dir = os.path.join(self.dataset_path, 'labels')
            if os.path.exists(labels_dir):
                for label_file in os.listdir(labels_dir):
                    if not label_file.endswith('.txt'):
                        continue
                        
                    label_path = os.path.join(labels_dir, label_file)
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:  # Ensure valid line
                                info['total_annotations'] += 1
                                class_id = parts[0]
                                if class_id not in info['classes']:
                                    info['classes'][class_id] = {
                                        'name': f'class_{class_id}',
                                        'count': 0
                                    }
                                info['classes'][class_id]['count'] += 1
            
            self.logger.info(f"Dataset info generated: {info['total_images']} images, "
                        f"{info['total_annotations']} annotations")
            
            return info  # Make sure to return the info dictionary
        
        except Exception as e:
            self.logger.error(f"Error generating dataset info: {str(e)}")
            return info  # Return info even if there's an error
            
    def export_dataset(self, format: str = 'coco', export_path: Optional[str] = None) -> str:
        """Export dataset to specified format."""
        try:
            if export_path is None:
                export_path = os.path.join(self.dataset_path, 'exports',
                                         f'{format}_{int(time.time())}')
            
            os.makedirs(export_path, exist_ok=True)
            
            if format == 'coco':
                from .exporters.coco_exporter import CocoExporter
                exporter = CocoExporter(self.dataset_path, export_path)
                return exporter.export()
            elif format == 'yolo':
                from .exporters.yolo_exporter import YoloExporter
                exporter = YoloExporter(self.dataset_path, export_path)
                return exporter.export()
            else:
                raise ValueError(f"Unsupported export format: {format}")
        except Exception as e:
            self.logger.error(f"Error exporting dataset: {str(e)}")
            raise  # Re-raise the exception after logging
    
    def _export_coco(self, export_path: str) -> str:
        """Export dataset in COCO format."""
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Load dataset info for categories
        dataset_info = self.load_dataset_info()
        for class_id, class_info in dataset_info['classes'].items():
            coco_data['categories'].append({
                'id': int(class_id),
                'name': class_info['name'],
                'supercategory': 'object'
            })
        
        # Process images and annotations
        ann_id = 0
        labels_dir = os.path.join(self.dataset_path, 'labels')
        images_dir = os.path.join(self.dataset_path, 'images')
        
        for img_id, image_file in enumerate(os.listdir(images_dir)):
            if not image_file.endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            # Copy image
            shutil.copy2(
                os.path.join(images_dir, image_file),
                os.path.join(export_path, image_file)
            )
            
            # Get image info
            img_path = os.path.join(images_dir, image_file)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            coco_data['images'].append({
                'id': img_id,
                'file_name': image_file,
                'height': h,
                'width': w
            })
            
            # Process annotations
            label_file = os.path.splitext(image_file)[0] + '.json'
            if os.path.exists(os.path.join(labels_dir, label_file)):
                with open(os.path.join(labels_dir, label_file), 'r') as f:
                    annotations = json.load(f)
                    
                    for ann in annotations:
                        if 'contour_points' not in ann:
                            continue
                            
                        # Convert contour points to COCO segmentation format
                        segmentation = [np.array(ann['contour_points']).flatten().tolist()]
                        
                        # Calculate bbox from contour points
                        contour = np.array(ann['contour_points'])
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        coco_data['annotations'].append({
                            'id': ann_id,
                            'image_id': img_id,
                            'category_id': ann['class_id'],
                            'segmentation': segmentation,
                            'area': cv2.contourArea(contour),
                            'bbox': [x, y, w, h],
                            'iscrowd': 0
                        })
                        ann_id += 1
        
        # Save COCO JSON
        with open(os.path.join(export_path, 'annotations.json'), 'w') as f:
            json.dump(coco_data, f, indent=4)
            
        return export_path
    
    def _export_yolo(self, export_path: str) -> str:
        """Export dataset in YOLO format."""
        # Create necessary directories
        os.makedirs(os.path.join(export_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(export_path, 'labels'), exist_ok=True)
        
        # Process images and annotations
        labels_dir = os.path.join(self.dataset_path, 'labels')
        images_dir = os.path.join(self.dataset_path, 'images')
        
        # Create classes.txt
        dataset_info = self.load_dataset_info()
        with open(os.path.join(export_path, 'classes.txt'), 'w') as f:
            for class_id, class_info in sorted(dataset_info['classes'].items()):
                f.write(f"{class_info['name']}\n")
        
        for image_file in os.listdir(images_dir):
            if not image_file.endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            # Copy image
            shutil.copy2(
                os.path.join(images_dir, image_file),
                os.path.join(export_path, 'images', image_file)
            )
            
            # Get image dimensions
            img_path = os.path.join(images_dir, image_file)
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]
            
            # Process annotations
            label_file = os.path.splitext(image_file)[0] + '.json'
            yolo_annotations = []
            
            if os.path.exists(os.path.join(labels_dir, label_file)):
                with open(os.path.join(labels_dir, label_file), 'r') as f:
                    annotations = json.load(f)
                    
                    for ann in annotations:
                        if 'contour_points' not in ann:
                            continue
                            
                        # Calculate bbox from contour points
                        contour = np.array(ann['contour_points'])
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Convert to YOLO format (normalized coordinates)
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        width = w / img_width
                        height = h / img_height
                        
                        # YOLO format: <class> <x_center> <y_center> <width> <height>
                        yolo_annotations.append(
                            f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )
            
            # Save YOLO annotations
            yolo_label_path = os.path.join(
                export_path, 'labels',
                os.path.splitext(image_file)[0] + '.txt'
            )
            with open(yolo_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        
        return export_path

    def create_backup(self) -> str:
        """Create a backup of the current dataset state."""
        backup_dir = os.path.join(self.dataset_path, 'backups')
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(backup_dir, f'backup_{timestamp}')
        
        # Create backup directory
        os.makedirs(backup_path)
        
        # Copy current annotations and dataset info
        shutil.copy2(
            os.path.join(self.dataset_path, 'dataset_info.json'),
            os.path.join(backup_path, 'dataset_info.json')
        )
        
        # Copy labels directory
        shutil.copytree(
            os.path.join(self.dataset_path, 'labels'),
            os.path.join(backup_path, 'labels')
        )
        
        # Maintain maximum number of backups
        backups = sorted([d for d in os.listdir(backup_dir)
                         if os.path.isdir(os.path.join(backup_dir, d))])
        
        while len(backups) > self.max_backups:
            oldest_backup = os.path.join(backup_dir, backups[0])
            shutil.rmtree(oldest_backup)
            backups.pop(0)
            
        return backup_path
    
    def auto_save(self, annotations: List[Dict], image_path: str) -> None:
        """Perform auto-save if needed."""
        current_time = time.time()
        
        if not self.auto_save_enabled:
            return
            
        if current_time - self.last_auto_save >= self.auto_save_interval:
            self.save_annotations(annotations, image_path)
            self.last_auto_save = current_time
            
    def save_annotations(self, annotations: List[Dict], image_path: str) -> None:
        """Save annotations for current image."""
        # Get label path
        image_name = os.path.basename(image_path)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(self.dataset_path, 'labels', label_name)
        
        # Ensure the labels directory exists
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        # Create annotation strings
        annotation_lines = []
        for annotation in annotations:
            # Convert contour points to string format
            points_str = ' '.join([f"{pt[0][0]} {pt[0][1]}" for pt in annotation['contour_points']])
            # Format: class_id num_points x1 y1 x2 y2 ...
            line = f"{annotation['class_id']} {len(annotation['contour_points'])} {points_str}"
            annotation_lines.append(line)
        
        # Write to file
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotation_lines))
            
        # Update dataset info
        self._update_dataset_info()
        
    def _update_dataset_info(self) -> None:
        """Update dataset information after changes."""
        try:
            # Generate fresh info
            info = self._generate_dataset_info()
            
            # Save to file
            info_path = os.path.join(self.dataset_path, 'dataset_info.json')
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=4)
                
            self.logger.info(f"Updated dataset info: {info['total_annotations']} annotations")
            
        except Exception as e:
            self.logger.error(f"Error updating dataset info: {str(e)}")
            # Don't return anything since it's a void method
            
    def load_annotations(self, image_path: str) -> List[Dict]:
        """Load annotations for given image."""
        annotations = []
        try:
            # Get paths and check if label exists
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.dataset_path, 'labels', f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                return annotations
            
            # Get image dimensions
            img = cv2.imread(image_path)
            orig_height, orig_width = img.shape[:2]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    num_points = int(parts[1])
                    
                    # Parse points
                    points = []
                    for i in range(2, len(parts), 2):
                        x = float(parts[i])
                        y = float(parts[i + 1])
                        points.append([[x, y]])
                    
                    # Convert to numpy array
                    contour = np.array(points, dtype=np.int32)
                    
                    # Calculate bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    box = [x, y, x + w, y + h]
                    
                    # Create mask
                    mask = np.zeros((orig_height, orig_width), dtype=bool)
                    cv2.fillPoly(mask, [contour], 1)
                    
                    annotations.append({
                        'class_id': class_id,
                        'contour_points': contour,
                        'box': box,
                        'mask': mask
                    })
                    
            return annotations
            
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
            return []
    
    def get_dataset_statistics(self) -> Dict:
        """Get detailed dataset statistics."""
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': {},
            'size_distribution': [],
            'annotations_per_image': [],
            'last_modified': None
        }
        
        images_dir = os.path.join(self.dataset_path, 'images')
        labels_dir = os.path.join(self.dataset_path, 'labels')
        
        for image_file in os.listdir(images_dir):
            if not image_file.endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            stats['total_images'] += 1
            
            # Get image size
            img_path = os.path.join(images_dir, image_file)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            stats['size_distribution'].append({'width': w, 'height': h})
            
            # Process annotations
            label_file = os.path.splitext(image_file)[0] + '.json'
            if os.path.exists(os.path.join(labels_dir, label_file)):
                with open(os.path.join(labels_dir, label_file), 'r') as f:
                    annotations = json.load(f)
                    stats['total_annotations'] += len(annotations)
                    stats['annotations_per_image'].append(len(annotations))
                    
                    for ann in annotations:
                        class_id = str(ann.get('class_id', -1))
                        if class_id not in stats['class_distribution']:
                            stats['class_distribution'][class_id] = 0
                        stats['class_distribution'][class_id] += 1
            else:
                stats['annotations_per_image'].append(0)
        
        return stats