from typing import List, Dict, Optional, Tuple
import os
import json
import logging
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from weakref import WeakValueDictionary
import psutil
from threading import Thread, Lock
import queue




class LazyImageLoader:
    """Handles lazy loading of images with metadata caching."""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self._image = None
        self._metadata = None
        self._lock = Lock()
    
    @property
    def image(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._image is None:
                self._image = cv2.imread(self.image_path)
                if self._image is not None:
                    self._metadata = {
                        'height': self._image.shape[0],
                        'width': self._image.shape[1],
                        'channels': self._image.shape[2] if len(self._image.shape) > 2 else 1,
                        'size': os.path.getsize(self.image_path)
                    }
            return self._image
    
    @property
    def metadata(self) -> Optional[Dict]:
        if self._metadata is None and os.path.exists(self.image_path):
            # Get basic metadata without loading image
            self._metadata = {
                'size': os.path.getsize(self.image_path),
                'modified': os.path.getmtime(self.image_path)
            }
        return self._metadata
    
    def clear(self):
        with self._lock:
            self._image = None 





class DatasetManager:
    """Enhanced dataset manager with lazy loading and caching."""
    
    def __init__(self, dataset_path: str, cache_size: int = 10):
        """Initialize dataset manager with the root dataset path."""
        self.dataset_path = dataset_path
        self.logger = logging.getLogger(__name__)
        
        # Setup directory structure
        self.images_path = os.path.join(dataset_path, 'images')
        self.labels_path = os.path.join(dataset_path, 'labels')
        self.masks_path = os.path.join(dataset_path, 'masks')
        self.metadata_path = os.path.join(dataset_path, 'metadata')
        self.exports_path = os.path.join(dataset_path, 'exports')
        self.backups_path = os.path.join(dataset_path, 'backups')
        
        for path in [self.labels_path, self.masks_path, self.metadata_path, 
                    self.exports_path, self.backups_path]:
            os.makedirs(path, exist_ok=True)
        
        # Initialize caches with weak references
        self.image_cache = WeakValueDictionary()
        self.annotation_cache = {}
        self.metadata_cache = {}
        self.max_cache_size = cache_size
        
        # Auto-save settings
        self.auto_save_enabled = True
        self.auto_save_interval = 300  # 5 minutes
        self.last_auto_save = time.time()
        
        # Backup settings
        self.max_backups = 5
        self.backup_interval = 3600  # 1 hour
        
        # Preloading settings
        self.preload_enabled = True
        self.preload_queue = queue.Queue(maxsize=5)
        self.preload_thread = Thread(target=self._preload_worker, daemon=True)
        self.preload_thread.start()
        
    def _preload_worker(self):
        """Background worker for preloading images."""
        while True:
            try:
                image_path = self.preload_queue.get()
                if image_path not in self.image_cache:
                    loader = LazyImageLoader(image_path)
                    # Only load metadata, not the actual image
                    _ = loader.metadata
                    self.image_cache[image_path] = loader
                self.preload_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error in preload worker: {str(e)}")
            
    def load_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Load image with lazy loading and caching."""
        try:
            # Check cache first
            if image_path in self.image_cache:
                loader = self.image_cache[image_path]
            else:
                loader = LazyImageLoader(image_path)
                self.image_cache[image_path] = loader
            
            # Trigger actual image loading
            image = loader.image
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Queue next images for preloading
            self._queue_next_images(image_path)
            
            return image, loader.metadata
            
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise
            
    def _queue_next_images(self, current_path: str): 
        """Queue next few images for preloading."""
        if not self.preload_enabled:
            return
            
        try:
            current_name = os.path.basename(current_path)
            image_files = sorted([f for f in os.listdir(self.images_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            current_idx = image_files.index(current_name)
            
            # Queue next 3 images
            for idx in range(current_idx + 1, min(current_idx + 4, len(image_files))):
                next_path = os.path.join(self.images_path, image_files[idx])
                if next_path not in self.image_cache:
                    try:
                        self.preload_queue.put_nowait(next_path)
                    except queue.Full:
                        break
                        
        except Exception as e:
            self.logger.error(f"Error queueing next images: {str(e)}")
        
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
            elif format == 'pascal':
                from .exporters.pascal_exporter import PascalVOCExporter
                exporter = PascalVOCExporter(self.dataset_path, export_path)
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
        """Auto-save with optimized saving frequency."""
        current_time = time.time()
        if not self.auto_save_enabled:
            return
            
        if current_time - self.last_auto_save >= self.auto_save_interval:
            try:
                self.save_annotations(annotations, image_path)
                self.last_auto_save = current_time
            except Exception as e:
                self.logger.error(f"Error in auto-save: {str(e)}")
            
    
    
    
    def save_annotations(self, annotations: List[Dict], image_path: str) -> None:
        """Save annotations with caching."""
        try:
            # Get label path
            image_name = os.path.basename(image_path)
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(self.labels_path, label_name)
            
            # Create annotation strings
            annotation_lines = []
            for annotation in annotations:
                points_str = ' '.join([f"{pt[0][0]} {pt[0][1]}" 
                                     for pt in annotation['contour_points']])
                line = f"{annotation['class_id']} {len(annotation['contour_points'])} {points_str}"
                annotation_lines.append(line)
            
            # Write to file
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, 'w') as f:
                f.write('\n'.join(annotation_lines))
            
            # Update caches
            self.annotation_cache[image_path] = annotations.copy()
            self._update_dataset_info()
            
        except Exception as e:
            self.logger.error(f"Error saving annotations: {str(e)}")
            raise
    
    
    
    
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
        """Load polygon annotations with caching."""
        try:
            # Check cache first
            if image_path in self.annotation_cache:
                return self.annotation_cache[image_path]
            
            annotations = []
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.labels_path, f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                return annotations
            
            # Get image dimensions
            if image_path in self.image_cache:
                img_metadata = self.image_cache[image_path].metadata
                orig_height, orig_width = img_metadata['height'], img_metadata['width']
            else:
                img = cv2.imread(image_path)
                if img is None:
                    self.logger.error(f"Could not read image: {image_path}")
                    return annotations
                orig_height, orig_width = img.shape[:2]
            
            # Load annotations
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        if len(parts) < 3:  # Need at least class_id and one point
                            continue
                        
                        # First value is class ID
                        class_id = int(float(parts[0]))
                        
                        # Rest of values are x,y coordinates
                        polygon_points = []
                        for i in range(1, len(parts), 2):
                            if i + 1 >= len(parts):
                                break
                            x = float(parts[i]) * orig_width
                            y = float(parts[i + 1]) * orig_height
                            polygon_points.append([[int(x), int(y)]])
                        
                        if len(polygon_points) < 3:  # Need at least 3 points for a polygon
                            continue
                        
                        # Convert points to numpy array
                        contour = np.array(polygon_points, dtype=np.int32)
                        
                        # Calculate bounding box directly from the contour
                        x, y, w, h = cv2.boundingRect(contour)
                        box = [x, y, x + w, y + h]
                        
                        # Create mask using uint8 instead of bool
                        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
                        cv2.fillPoly(mask, [contour], 1)
                        
                        # Convert mask to boolean after filling
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
                        
                        annotations.append({
                            'id': len(annotations),
                            'class_id': class_id,
                            'class_name': f'Class {class_id}',  # Default class name
                            'contour_points': contour,  # Original cv2 contour format
                            'contour': flattened_contour,  # Flattened points for visualization
                            'box': box,
                            'display_box': box,  # Same as box initially
                            'mask': mask_bool,
                            'area': area,
                            'original_shape': (orig_height, orig_width),
                            'metadata': {
                                'annotation_mode': 'imported',
                                'timestamp': time.time()
                            }
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Error parsing annotation line: {line.strip()}, Error: {str(e)}")
                        continue
            
            # Cache the results
            self.annotation_cache[image_path] = annotations
            return annotations
            
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
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
    
    
    def monitor_memory_usage(self) -> bool:
        """Monitor memory usage and clear cache if needed."""
        try:
            process = psutil.Process()
            memory_percent = process.memory_percent()
            
            if memory_percent > 75:  # Over 75% memory usage
                self.logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
                self.clear_cache()
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error monitoring memory: {str(e)}")
            return False
        
        
        
    def clear_cache(self, current_image_path: Optional[str] = None) -> None:
        """Clear cache with option to keep current image."""
        try:
            if current_image_path and current_image_path in self.image_cache:
                # Keep only current image
                current_loader = self.image_cache[current_image_path]
                current_annot = self.annotation_cache.get(current_image_path)
                
                self.image_cache.clear()
                self.annotation_cache.clear()
                self.metadata_cache.clear()
                
                self.image_cache[current_image_path] = current_loader
                if current_annot is not None:
                    self.annotation_cache[current_image_path] = current_annot
            else:
                self.image_cache.clear()
                self.annotation_cache.clear()
                self.metadata_cache.clear()
                
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            
            
    
        
    