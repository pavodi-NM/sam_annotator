# from typing import Dict, List
# import json
# import cv2
# import numpy as np


# class YoloExporter:
#     """Handles export to YOLO format."""
    
#     @staticmethod
#     def export(dataset_path: str, export_path: str) -> str:
#         """Export dataset in YOLO format."""
#         # Create necessary directories
#         os.makedirs(os.path.join(export_path, 'images'), exist_ok=True)
#         os.makedirs(os.path.join(export_path, 'labels'), exist_ok=True)
        
#         # Process images and annotations
#         labels_dir = os.path.join(self.dataset_path, 'labels')
#         images_dir = os.path.join(self.dataset_path, 'images')
        
#         # Create classes.txt
#         dataset_info = self.load_dataset_info()
#         with open(os.path.join(export_path, 'classes.txt'), 'w') as f:
#             for class_id, class_info in sorted(dataset_info['classes'].items()):
#                 f.write(f"{class_info['name']}\n")
        
#         for image_file in os.listdir(images_dir):
#             if not image_file.endswith(('.jpg', '.png', '.jpeg')):
#                 continue
                
#             # Copy image
#             shutil.copy2(
#                 os.path.join(images_dir, image_file),
#                 os.path.join(export_path, 'images', image_file)
#             )
            
#             # Get image dimensions
#             img_path = os.path.join(images_dir, image_file)
#             img = cv2.imread(img_path)
#             img_height, img_width = img.shape[:2]
            
#             # Process annotations
#             label_file = os.path.splitext(image_file)[0] + '.json'
#             yolo_annotations = []
            
#             if os.path.exists(os.path.join(labels_dir, label_file)):
#                 with open(os.path.join(labels_dir, label_file), 'r') as f:
#                     annotations = json.load(f)
                    
#                     for ann in annotations:
#                         if 'contour_points' not in ann:
#                             continue
                            
#                         # Calculate bbox from contour points
#                         contour = np.array(ann['contour_points'])
#                         x, y, w, h = cv2.boundingRect(contour)
                        
#                         # Convert to YOLO format (normalized coordinates)
#                         x_center = (x + w/2) / img_width
#                         y_center = (y + h/2) / img_height
#                         width = w / img_width
#                         height = h / img_height
                        
#                         # YOLO format: <class> <x_center> <y_center> <width> <height>
#                         yolo_annotations.append(
#                             f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
#                         )
            
#             # Save YOLO annotations
#             yolo_label_path = os.path.join(
#                 export_path, 'labels',
#                 os.path.splitext(image_file)[0] + '.txt'
#             )
#             with open(yolo_label_path, 'w') as f:
#                 f.write('\n'.join(yolo_annotations))
        
#         return export_path


import logging 
from typing import Dict, List
import json
import cv2
import numpy as np
import os
import shutil
import logging
from datetime import datetime
from .base_exporter import BaseExporter

class YoloExporter(BaseExporter):
    """Handles export to YOLO format."""
    
    def __init__(self, dataset_path: str):
        """Initialize the YOLO exporter."""
        super().__init__(dataset_path)
        self.logger = logging.getLogger(__name__)

    def load_dataset_info(self) -> Dict:
        """Load dataset information."""
        try:
            info_path = os.path.join(self.dataset_path, 'dataset_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    return json.load(f)
            return {'classes': {}}
        except Exception as e:
            self.logger.error(f"Error loading dataset info: {str(e)}")
            return {'classes': {}}
    
    def export(self) -> str:
        """Export dataset in YOLO format."""
        try:
            # Create export directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_path = os.path.join(self.dataset_path, 'exports', f'yolo_export_{timestamp}')
            
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
            
            self.logger.info(f"Successfully exported dataset to {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Error during YOLO export: {str(e)}")
            raise