import cv2
import os
import logging
from datetime import datetime
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple
from .base_exporter import BaseExporter

class PascalVOCExporter(BaseExporter):
    """Exports dataset to Pascal VOC format."""
    
    def __init__(self, dataset_path: str):
        """Initialize the Pascal VOC exporter."""
        super().__init__(dataset_path)
        self.logger = logging.getLogger(__name__)

    def _create_voc_xml(self, image_file: str, image_size: Tuple[int, int, int], 
                       annotations: List[Dict[str, Any]]) -> ET.Element:
        """Create Pascal VOC XML structure for an image.
        
        Args:
            image_file: Name of the image file
            image_size: Tuple of (height, width, channels)
            annotations: List of annotation dictionaries containing bounding boxes
            
        Returns:
            ET.Element: Root element of the XML tree
        """
        root = ET.Element("annotation")
        
        # Add basic image information
        folder = ET.SubElement(root, "folder")
        folder.text = "images"
        
        filename = ET.SubElement(root, "filename")
        filename.text = image_file
        
        # Add image size information
        size = ET.SubElement(root, "size")
        width = ET.SubElement(size, "width")
        width.text = str(image_size[1])
        height = ET.SubElement(size, "height")
        height.text = str(image_size[0])
        depth = ET.SubElement(size, "depth")
        depth.text = str(image_size[2])
        
        # Add segmented flag (0 for object detection)
        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"
        
        # Add each object annotation
        for ann in annotations:
            obj = ET.SubElement(root, "object")
            
            name = ET.SubElement(obj, "name")
            name.text = f"class_{ann['class_id']}"
            
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(ann['bbox'][0]))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(ann['bbox'][1]))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(ann['bbox'][0] + ann['bbox'][2]))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(ann['bbox'][1] + ann['bbox'][3]))
        
        return root

    def _parse_yolo_line(self, line: str, image_width: int, image_height: int) -> Dict[str, Any]:
        """Parse a line from YOLO format and convert to Pascal VOC format.
        
        Args:
            line: Single line from YOLO annotation file
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            Dictionary containing class_id and bounding box coordinates
        """
        parts = line.strip().split()
        class_id = int(parts[0])
        points = []
        
        # Parse normalized coordinates and convert to absolute pixels
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                x = float(parts[i]) * image_width
                y = float(parts[i + 1]) * image_height
                points.append([x, y])
        
        # Convert points to bounding box
        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            xmin = min(x_coords)
            ymin = min(y_coords)
            width = max(x_coords) - xmin
            height = max(y_coords) - ymin
            
            return {
                'class_id': class_id,
                'bbox': [xmin, ymin, width, height]
            }
        return None

    def export(self) -> str:
        """Export dataset to Pascal VOC format.
        
        Returns:
            str: Path to exported dataset
        """
        try:
            # Create export directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = os.path.join(self.dataset_path, 'exports', f'voc_export_{timestamp}')
            os.makedirs(export_dir, exist_ok=True)
            
            # Create necessary subdirectories
            annotations_dir = os.path.join(export_dir, 'Annotations')
            images_dir = os.path.join(export_dir, 'JPEGImages')
            os.makedirs(annotations_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            
            # Get list of annotated images
            annotated_images = self._get_annotated_images()
            self.logger.info(f"Found {len(annotated_images)} annotated images")
            
            if not annotated_images:
                self.logger.warning("No annotated images found!")
                return export_dir
            
            # Process each annotated image
            for image_file in annotated_images:
                try:
                    # Read image to get dimensions
                    image_path = os.path.join(self.dataset_path, 'images', image_file)
                    img = cv2.imread(image_path)
                    if img is None:
                        self.logger.warning(f"Could not read image: {image_path}")
                        continue
                    
                    height, width, channels = img.shape
                    
                    # Parse annotations
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
                    
                    # Create XML annotation
                    xml_root = self._create_voc_xml(image_file, (height, width, channels), annotations)
                    
                    # Save XML file
                    xml_path = os.path.join(annotations_dir, f"{os.path.splitext(image_file)[0]}.xml")
                    tree = ET.ElementTree(xml_root)
                    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                    
                    # Copy image to export directory
                    dst_path = os.path.join(images_dir, image_file)
                    cv2.imwrite(dst_path, img)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing image {image_file}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully exported {len(annotated_images)} images to Pascal VOC format")
            return export_dir
            
        except Exception as e:
            self.logger.error(f"Error during Pascal VOC export: {str(e)}")
            raise