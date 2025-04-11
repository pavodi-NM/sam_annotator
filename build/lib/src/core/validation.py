from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
import json
import os

class ValidationManager:
    """Manages validation rules and quality control for annotations."""
    
    def __init__(self, visualization_manager):
        self.viz_manager = visualization_manager
        self.validation_rules = {
            'min_size': 100,  # Minimum area in pixels
            'max_overlap': 0.3,  # Maximum allowed overlap ratio
            'required_fields': ['class_id', 'mask', 'box'],  # Updated to use mask instead of contour_points
            'max_annotations_per_class': 50,
            'min_annotations_per_class': 1
        }
        
        # Auto-save settings
        self.auto_save_interval = 300  # seconds
        self.last_auto_save = 0
        
    def validate_annotation(self, annotation: Dict, image_shape: Tuple[int, int]) -> Tuple[bool, str]:
        """Validate a single annotation against all rules."""
        # Check required fields
        for field in self.validation_rules['required_fields']:
            if field not in annotation:
                return False, f"Missing required field: {field}"
        
        # Validate mask size
        if 'mask' in annotation:
            area = np.count_nonzero(annotation['mask'])
            if area < self.validation_rules['min_size']:
                return False, f"Annotation too small: {area} pixels"
        # For backward compatibility, also check contour_points
        elif 'contour_points' in annotation:
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [annotation['contour_points']], -1, 255, -1)
            area = cv2.countNonZero(mask)
            if area < self.validation_rules['min_size']:
                return False, f"Annotation too small: {area} pixels"
        # For new structure, check contour
        elif 'contour' in annotation:
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            try:
                contour_array = np.array(annotation['contour'], dtype=np.int32)
                # Handle different possible shapes
                if len(contour_array.shape) == 2:  # [x,y] format
                    contour_array = contour_array.reshape(-1, 1, 2)
                cv2.drawContours(mask, [contour_array], -1, 255, -1)
                area = cv2.countNonZero(mask)
                if area < self.validation_rules['min_size']:
                    return False, f"Annotation too small: {area} pixels"
            except Exception as e:
                return False, f"Invalid contour format: {str(e)}"
        
        return True, "Valid annotation"
    
    def check_overlap(self, annotations: List[Dict], new_annotation: Dict, 
                     image_shape: Tuple[int, int]) -> Tuple[bool, float]:
        """Check overlap ratio between annotations."""
        # Create mask for new annotation
        new_mask = None
        
        # Check if we already have a mask
        if 'mask' in new_annotation and new_annotation['mask'] is not None:
            new_mask = new_annotation['mask'].astype(np.uint8) * 255
            if new_mask.shape[:2] != image_shape[:2]:
                new_mask = cv2.resize(new_mask, (image_shape[1], image_shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        # For backward compatibility, check contour_points
        elif 'contour_points' in new_annotation:
            new_mask = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.drawContours(new_mask, [new_annotation['contour_points']], -1, 255, -1)
        # For new structure, check contour
        elif 'contour' in new_annotation:
            new_mask = np.zeros(image_shape[:2], dtype=np.uint8)
            try:
                contour_array = np.array(new_annotation['contour'], dtype=np.int32)
                # Handle different possible shapes
                if len(contour_array.shape) == 2:  # [x,y] format
                    contour_array = contour_array.reshape(-1, 1, 2)
                cv2.drawContours(new_mask, [contour_array], -1, 255, -1)
            except Exception:
                return False, 0.0
        
        if new_mask is None:
            return True, 0.0
            
        new_area = cv2.countNonZero(new_mask)
        if new_area == 0:
            return True, 0.0
        
        max_overlap_ratio = 0.0
        for existing in annotations:
            existing_mask = None
            
            # Check if we already have a mask
            if 'mask' in existing and existing['mask'] is not None:
                existing_mask = existing['mask'].astype(np.uint8) * 255
                if existing_mask.shape[:2] != image_shape[:2]:
                    existing_mask = cv2.resize(existing_mask, (image_shape[1], image_shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
            # For backward compatibility, check contour_points
            elif 'contour_points' in existing:
                existing_mask = np.zeros(image_shape[:2], dtype=np.uint8)
                cv2.drawContours(existing_mask, [existing['contour_points']], -1, 255, -1)
            # For new structure, check contour
            elif 'contour' in existing:
                existing_mask = np.zeros(image_shape[:2], dtype=np.uint8)
                try:
                    contour_array = np.array(existing['contour'], dtype=np.int32)
                    # Handle different possible shapes
                    if len(contour_array.shape) == 2:  # [x,y] format
                        contour_array = contour_array.reshape(-1, 1, 2)
                    cv2.drawContours(existing_mask, [contour_array], -1, 255, -1)
                except Exception:
                    continue
            
            if existing_mask is None:
                continue
                
            # Calculate overlap
            overlap = cv2.bitwise_and(new_mask, existing_mask)
            overlap_area = cv2.countNonZero(overlap)
            overlap_ratio = overlap_area / new_area if new_area > 0 else 0
            
            max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)
            
            if overlap_ratio > self.validation_rules['max_overlap']:
                return False, overlap_ratio
                
        return True, max_overlap_ratio
    
    def validate_dataset(self, dataset_path: str) -> Dict:
        """Validate entire dataset against rules."""
        validation_results = {
            'valid_count': 0,
            'invalid_count': 0,
            'errors': [],
            'class_distribution': {},
            'size_distribution': []
        }
        
        # Iterate through annotation files
        for root, _, files in os.walk(os.path.join(dataset_path, 'labels')):
            for file in files:
                if not file.endswith('.json'):
                    continue
                    
                with open(os.path.join(root, file), 'r') as f:
                    try:
                        annotations = json.load(f)
                        
                        # Check each annotation
                        for ann in annotations:
                            is_valid, message = self.validate_annotation(
                                ann, (1000, 1000))  # Default size for now
                            
                            if is_valid:
                                validation_results['valid_count'] += 1
                                # Update class distribution
                                class_id = ann.get('class_id', -1)
                                validation_results['class_distribution'][class_id] = \
                                    validation_results['class_distribution'].get(class_id, 0) + 1
                            else:
                                validation_results['invalid_count'] += 1
                                validation_results['errors'].append({
                                    'file': file,
                                    'error': message
                                })
                                
                    except Exception as e:
                        validation_results['errors'].append({
                            'file': file,
                            'error': f"Failed to parse annotation file: {str(e)}"
                        })
        
        return validation_results
    
    def should_auto_save(self, current_time: float) -> bool:
        """Check if it's time for auto-save."""
        if current_time - self.last_auto_save >= self.auto_save_interval:
            self.last_auto_save = current_time
            return True
        return False
    
    def update_validation_rules(self, new_rules: Dict) -> None:
        """Update validation rules with new values."""
        self.validation_rules.update(new_rules)
        
    def get_validation_summary(self, annotations: List[Dict], 
                             image_shape: Tuple[int, int]) -> Dict:
        """Generate summary of validation status for current image."""
        summary = {
            'total_annotations': len(annotations),
            'valid_annotations': 0,
            'invalid_annotations': 0,
            'class_counts': {},
            'overlapping_pairs': [],
            'error_messages': []
        }
        
        # Check each annotation
        for i, ann in enumerate(annotations):
            is_valid, message = self.validate_annotation(ann, image_shape)
            if is_valid:
                summary['valid_annotations'] += 1
            else:
                summary['invalid_annotations'] += 1
                summary['error_messages'].append(f"Annotation {i}: {message}")
            
            # Update class counts
            class_id = ann.get('class_id', -1)
            summary['class_counts'][class_id] = \
                summary['class_counts'].get(class_id, 0) + 1
        
        # Check overlap between all pairs
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                # Use the existing annotations and just check pairs
                is_valid, overlap_ratio = self.check_overlap(
                    [annotations[i]], annotations[j], image_shape)
                if not is_valid:
                    summary['overlapping_pairs'].append({
                        'annotation1': i,
                        'annotation2': j,
                        'overlap_ratio': overlap_ratio
                    })
        
        return summary