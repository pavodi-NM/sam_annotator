import cv2
import os
import logging
from datetime import datetime
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple
from .base_exporter import BaseExporter
import numpy as np

# """Exports dataset to Pascal VOC format with segmentation support."""


class PascalVOCExporter(BaseExporter):
    def __init__(self, dataset_path: str, copy_images: bool = False):
        """Initialize the Pascal VOC exporter. Exports dataset to Pascal VOC format with segmentation support.
        
        Args:
            dataset_path: Path to dataset
            copy_images: Whether to copy images to export directory (default: False)
        """
        super().__init__(dataset_path)
        self.logger = logging.getLogger(__name__)
        self.copy_images = copy_images

    def _get_annotated_images(self) -> List[str]:
        """Get list of images that have corresponding annotation files."""
        image_files = self._get_image_files()
        annotated_images = []
        
        for image_file in image_files:
            annotation_file = self._get_annotation_file(image_file)
            if os.path.exists(annotation_file):
                if os.path.getsize(annotation_file) > 0:
                    annotated_images.append(image_file)
        
        return annotated_images

    def _parse_yolo_line(self, line: str, image_width: int, image_height: int) -> Dict[str, Any]:
        """Parse a line from YOLO format and convert to absolute coordinates."""
        try:
            parts = line.strip().split()
            class_id = int(parts[0])
            points = []
            
            # Parse normalized coordinates and convert to absolute pixels
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    x = float(parts[i]) * image_width
                    y = float(parts[i + 1]) * image_height
                    points.append([round(x), round(y)])  # Round to nearest pixel
            
            # Calculate bounding box from points
            points_array = np.array(points)
            x_min = np.min(points_array[:, 0])
            y_min = np.min(points_array[:, 1])
            x_max = np.max(points_array[:, 0])
            y_max = np.max(points_array[:, 1]) 
            
            return {
                'class_id': class_id,
                'points': points,
                'bbox': [x_min, y_min, x_max, y_max]
            }
        except Exception as e:
            self.logger.error(f"Error parsing YOLO line: {str(e)}")
            return None

    def _create_voc_xml(self, image_file: str, image_size: Tuple[int, int, int], 
                       annotations: List[Dict[str, Any]]) -> ET.Element:
        """Create Pascal VOC XML structure for an image with segmentation support."""
        root = ET.Element("annotation")
        
        # Add basic image information
        folder = ET.SubElement(root, "folder")
        folder.text = "images"
        
        filename = ET.SubElement(root, "filename")
        filename.text = image_file
        
        # Add source information
        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "SAM Annotator Export"
        
        # Add image size information
        size = ET.SubElement(root, "size")
        width = ET.SubElement(size, "width")
        width.text = str(image_size[1])
        height = ET.SubElement(size, "height")
        height.text = str(image_size[0])
        depth = ET.SubElement(size, "depth")
        depth.text = str(image_size[2])
        
        # Set segmented flag to 1 since we have segmentation data
        segmented = ET.SubElement(root, "segmented")
        segmented.text = "1"
        
        # Add each object annotation
        for ann in annotations:
            if ann is None or 'points' not in ann:
                continue
                
            obj = ET.SubElement(root, "object")
            
            name = ET.SubElement(obj, "name")
            name.text = f"class_{ann['class_id']}"
            
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            
            # Add bounding box
            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(ann['bbox'][0]))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(ann['bbox'][1]))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(ann['bbox'][2]))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(ann['bbox'][3]))
            
            # Add polygon points for segmentation
            polygon = ET.SubElement(obj, "polygon")
            points_str = ' '.join([f"{int(pt[0])},{int(pt[1])}" for pt in ann['points']])
            points_elem = ET.SubElement(polygon, "points")
            points_elem.text = points_str
        
        return root

    def export(self) -> str:
        """Export dataset to Pascal VOC format."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = os.path.join(self.dataset_path, 'exports', f'voc_export_{timestamp}')
            os.makedirs(export_dir, exist_ok=True)
            
            # Create Annotations directory
            annotations_dir = os.path.join(export_dir, 'Annotations')
            os.makedirs(annotations_dir, exist_ok=True)
            
            # Only create JPEGImages directory if copying images
            if self.copy_images:
                images_dir = os.path.join(export_dir, 'JPEGImages')
                os.makedirs(images_dir, exist_ok=True)
            
            annotated_images = self._get_annotated_images()
            self.logger.info(f"Found {len(annotated_images)} annotated images")
            
            if not annotated_images:
                self.logger.warning("No annotated images found!")
                return export_dir

            for image_file in annotated_images:
                try:
                    # Read image just to get dimensions
                    image_path = os.path.join(self.dataset_path, 'images', image_file)
                    img = cv2.imread(image_path)
                    if img is None:
                        self.logger.warning(f"Could not read image: {image_path}")
                        continue
                    
                    height, width, channels = img.shape
                    
                    # Process annotations and create XML
                    annotations = []
                    annotation_file = self._get_annotation_file(image_file)
                    
                    with open(annotation_file, 'r') as f:
                        for line in f:
                            try:
                                ann_data = self._parse_yolo_line(line, width, height)
                                if ann_data:
                                    annotations.append(ann_data)
                            except Exception as e:
                                self.logger.warning(f"Error processing annotation in {annotation_file}: {str(e)}")
                                continue
                    
                    xml_root = self._create_voc_xml(image_file, (height, width, channels), annotations)
                    
                    # Save XML file
                    xml_path = os.path.join(annotations_dir, f"{os.path.splitext(image_file)[0]}.xml")
                    tree = ET.ElementTree(xml_root)
                    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                    
                    # Copy image only if specified
                    if self.copy_images:
                        dst_path = os.path.join(images_dir, image_file)
                        cv2.imwrite(dst_path, img)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing image {image_file}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully exported annotations for {len(annotated_images)} images")
            return export_dir
            
        except Exception as e:
            self.logger.error(f"Error during Pascal VOC export: {str(e)}")
            raise
        
        
    