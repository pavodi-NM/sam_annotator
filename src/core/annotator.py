import cv2
import numpy as np
import torch
import os
import time
import logging
from typing import Optional, List, Dict, Tuple
import pandas as pd

from ..ui.window_manager import WindowManager
from ..ui.event_handler import EventHandler
from ..utils.visualization import VisualizationManager
from .predictor import SAMPredictor
from .weight_manager import SAMWeightManager 
from ..utils.image_utils import ImageProcessor 

from ..core.validation import ValidationManager
from ..data.dataset_manager import DatasetManager

from .command_manager import (
    CommandManager,
    AddAnnotationCommand,
    DeleteAnnotationCommand,
    ModifyAnnotationCommand
)

class SAMAnnotator:
    """Main class for SAM-based image annotation."""
    
    def __init__(self, checkpoint_path: str, category_path: str, classes_csv: str):
        """Initialize the SAM annotator."""
        

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize image processor (add this)
        self.image_processor = ImageProcessor(target_size=1024, min_size=600)
        
        
        # Initialize managers
        self.window_manager = WindowManager(logger=self.logger)
        self.event_handler = EventHandler(self.window_manager, logger=self.logger)
        self.vis_manager = VisualizationManager()
        
        """ New """
        # Initialize managers
        self.dataset_manager = DatasetManager(category_path)
        self.validation_manager = ValidationManager(self.vis_manager)
      
        
        # Load SAM model
        self._initialize_model(checkpoint_path)
        
        # Load classes and setup paths
        self._load_classes(classes_csv)
        self._setup_paths(category_path)
        
        # Initialize state
        self.current_idx = 0
        self.current_image_path: Optional[str] = None
        self.image: Optional[np.ndarray] = None
        self.annotations: List[Dict] = []
        self.current_class_id = 0
        
        # add command manager
        self.command_manager = CommandManager()
        
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _initialize_model(self, checkpoint_path: str) -> None:
        """Initialize SAM model."""
        try:
            weight_manager = SAMWeightManager()
            verified_checkpoint_path = weight_manager.get_checkpoint_path(checkpoint_path)
            
            self.predictor = SAMPredictor()
            self.predictor.initialize(verified_checkpoint_path)
            self.logger.info("SAM model initialized successfully!")
        except Exception as e:
            self.logger.error(f"Error initializing SAM model: {str(e)}")
            raise
    
    def _load_classes(self, classes_csv: str) -> None:
        """Load class names from CSV."""
        try:
            df = pd.read_csv(classes_csv)
            self.class_names = df['class_name'].tolist()[:15]
            self.logger.info(f"Loaded {len(self.class_names)} classes")
        except Exception as e:
            self.logger.error(f"Error loading classes: {str(e)}")
            raise
    
    def _setup_paths(self, category_path: str) -> None:
        """Setup paths for images and annotations."""
        self.images_path = os.path.join(category_path, 'images')
        self.annotations_path = os.path.join(category_path, 'labels')
        os.makedirs(self.annotations_path, exist_ok=True)  
    
    def _setup_callbacks(self) -> None:
        """Setup event callbacks."""
        # Event handler callbacks
        self.event_handler.register_callbacks(
            on_mask_prediction=self._handle_mask_prediction,
            on_class_selection=self._handle_class_selection
        )
        
        # Review panel callbacks
        review_callbacks = {
            'delete': self._on_annotation_delete,
            'select': self._on_annotation_select,
            'class_change': self._on_annotation_class_change
        }
        
        # Window manager callbacks
        self.window_manager.setup_windows(
            mouse_callback=self.event_handler.handle_mouse_event,
            class_callback=self.event_handler.handle_class_window_event,
            review_callbacks=review_callbacks
        )
        

    def _on_annotation_select(self, idx: int) -> None:
        """Handle annotation selection."""
        if 0 <= idx < len(self.annotations):
            self.selected_annotation_idx = idx
            # Update the main window to highlight selected annotation
            # This will be handled through the window manager's update mechanism

    
    def _on_annotation_class_change(self, idx: int, new_class_id: int) -> None:
        """Handle annotation class change."""
        if 0 <= idx < len(self.annotations) and 0 <= new_class_id < len(self.class_names):
            try:
                # Create new state
                new_state = self.annotations[idx].copy()
                new_state['class_id'] = new_class_id
                new_state['class_name'] = self.class_names[new_class_id]
                
                # Create and execute modify command
                command = ModifyAnnotationCommand(self.annotations, idx, new_state, self.window_manager)
                self.command_manager.execute(command)
                
            except Exception as e:
                self.logger.error(f"Error changing annotation class: {str(e)}")
 
    def _handle_mask_prediction(self, 
                          box_start: Tuple[int, int],
                          box_end: Tuple[int, int],
                          drawing: bool = False) -> None:
        """Handle mask prediction from box input."""
        if drawing:
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=len(self.image_files),
                box_start=box_start,
                box_end=box_end
            )
            return
        
        try:
            # Get display and original dimensions
            display_height, display_width = self.image.shape[:2]
            original_image = cv2.imread(self.current_image_path)
            original_height, original_width = original_image.shape[:2]
            
            # Calculate scale factors
            scale_x = original_width / display_width
            scale_y = original_height / display_height
            
            # Scale box coordinates to original image size
            orig_box_start = (
                int(box_start[0] * scale_x),
                int(box_start[1] * scale_y)
            )
            orig_box_end = (
                int(box_end[0] * scale_x),
                int(box_end[1] * scale_y)
            )
            
            # Calculate center point in original coordinates
            center_x = (orig_box_start[0] + orig_box_end[0]) // 2
            center_y = (orig_box_start[1] + orig_box_end[1]) // 2
            
            input_points = np.array([[center_x, center_y]])
            input_labels = np.array([1])
            
            # Create input box in original coordinates
            input_box = np.array([
                min(orig_box_start[0], orig_box_end[0]),
                min(orig_box_start[1], orig_box_end[1]),
                max(orig_box_start[0], orig_box_end[0]),
                max(orig_box_start[1], orig_box_end[1])
            ])
            
            # Predict mask
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_box,
                multimask_output=True
            )
            
            if len(scores) > 0:
                best_mask_idx = np.argmax(scores)
                
                # Scale the mask to display size
                best_mask = masks[best_mask_idx]
                display_mask = cv2.resize(
                    best_mask.astype(np.uint8),
                    (display_width, display_height),
                    interpolation=cv2.INTER_NEAREST
                )
                
                self.window_manager.set_mask(display_mask.astype(bool))
                self.window_manager.update_main_window(
                    image=self.image,
                    annotations=self.annotations,
                    current_class=self.class_names[self.current_class_id],
                    current_class_id=self.current_class_id,
                    current_image_path=self.current_image_path,
                    current_idx=self.current_idx,
                    total_images=len(self.image_files),
                    status="Mask predicted - press 'a' to add"
                )
                
        except Exception as e:
            self.logger.error(f"Error in mask prediction: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _handle_class_selection(self, class_id: int) -> None:
        """Handle class selection."""
        if 0 <= class_id < len(self.class_names):
            self.current_class_id = class_id
            self.window_manager.update_class_window(
                self.class_names, 
                self.current_class_id
            )
    
    def _load_image(self, image_path: str) -> None:
        """Load image and prepare for annotation."""
        try:
            # Load original image
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            # Process image for display
            display_image, metadata = self.image_processor.process_image(original_image)
            
            self.image = display_image
            self.current_image_path = image_path
            
            # Set original image in predictor
            self.predictor.set_image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            
            # Reset annotations and mask
            self.annotations = []
            self.window_manager.set_mask(None)
            
            self.logger.info(f"Loaded image: {image_path}")
            self.logger.info(f"Original size: {metadata['original_size']}")
            self.logger.info(f"Display size: {metadata['display_size']}")
            
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise
   
    def _add_annotation(self) -> None:
        """Add current annotation to the list with proper scaling."""
        self.logger.info("Attempting to add annotation...")
        
        current_mask = self.window_manager.current_mask
        if current_mask is None:
            self.logger.warning("No region selected! Draw a box first.")
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=len(self.image_files),
                status="No region selected! Draw a box first."
            )
            return

        try:
            # Get current box from event handler
            box_start = self.event_handler.box_start
            box_end = self.event_handler.box_end
            
            if not box_start or not box_end:
                self.logger.warning("Box coordinates not found")
                return

            # Get display dimensions
            display_height, display_width = self.image.shape[:2]
            
            # Convert mask to uint8 and find contours at display size
            mask_uint8 = (current_mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, 
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_TC89_KCOS)
            
            if not contours:
                self.logger.warning("No valid contours found in mask")
                return
                
            # Get the largest contour at display size
            display_contour = max(contours, key=cv2.contourArea)
            
            # Calculate bounding box at display size
            display_box = [
                min(box_start[0], box_end[0]),
                min(box_start[1], box_end[1]),
                max(box_start[0], box_end[0]),
                max(box_start[1], box_end[1])
            ]
            
            # Scale contour and box to original image size
            original_contour = self.image_processor.scale_to_original(
                display_contour, 'contour'
            )
            original_box = self.image_processor.scale_to_original(
                display_box, 'box'
            )
            
            # Create the annotation dictionary with both display and original coordinates
            annotation = {
                'class_id': self.current_class_id,
                'class_name': self.class_names[self.current_class_id],
                'mask': current_mask.copy(),
                'contour_points': display_contour,
                'original_contour': original_contour,
                'box': display_box,
                'original_box': original_box
            }

            # Validate the annotation before adding
            is_valid, message = self.validation_manager.validate_annotation(
                annotation, self.image.shape)
            
            if not is_valid:
                self.logger.warning(f"Invalid annotation: {message}")
                self.window_manager.update_main_window(
                    image=self.image,
                    annotations=self.annotations,
                    status=f"Invalid annotation: {message}"
                )
                return
            
            # Check for overlap with existing annotations
            is_valid, overlap_ratio = self.validation_manager.check_overlap(
                self.annotations, annotation, self.image.shape)
            
            if not is_valid:
                self.logger.warning(f"Excessive overlap detected: {overlap_ratio:.2f}")
                self.window_manager.update_main_window(
                    image=self.image,
                    annotations=self.annotations,
                    status=f"Too much overlap: {overlap_ratio:.2f}"
                )
                return
            
            # Create and execute command
            command = AddAnnotationCommand(self.annotations, annotation, self.window_manager)
            if self.command_manager.execute(command):
                self.logger.info(f"Successfully added annotation. Total annotations: {len(self.annotations)}")
                
                # Auto-save if enabled
                self.dataset_manager.auto_save(self.annotations, self.current_image_path)
                
                # Clear current selection
                self.event_handler.reset_state()
                self.window_manager.set_mask(None)
                
                # Update displays
                self.window_manager.update_main_window(
                    image=self.image,
                    annotations=self.annotations,
                    current_class=self.class_names[self.current_class_id],
                    current_class_id=self.current_class_id,
                    current_image_path=self.current_image_path,
                    current_idx=self.current_idx,
                    total_images=len(self.image_files),
                    status=f"Annotation {len(self.annotations)} added!"
                )
            
        except Exception as e:
            self.logger.error(f"Error adding annotation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _save_annotations(self) -> bool:
        """Save annotations with proper scaling for export formats."""
        try:
            self.logger.info(f"Starting save_annotations. Number of annotations: {len(self.annotations)}")
            
            if not self.annotations:
                self.logger.warning("No annotations to save!")
                return False
            
            # Get validation summary before saving
            summary = self.validation_manager.get_validation_summary(
                self.annotations, self.image.shape)
            
            if summary['invalid_annotations'] > 0:
                self.logger.warning(f"Found {summary['invalid_annotations']} invalid annotations")
                self.window_manager.update_main_window(
                    image=self.image,
                    annotations=self.annotations,
                    status=f"Warning: {summary['invalid_annotations']} invalid annotations"
                )
            
            # Get image dimensions and base paths
            original_image = cv2.imread(self.current_image_path)
            original_height, original_width = original_image.shape[:2]
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            original_ext = os.path.splitext(self.current_image_path)[1]
            
            # Setup directories
            masks_dir = os.path.join(os.path.dirname(self.annotations_path), 'masks')
            metadata_dir = os.path.join(os.path.dirname(self.annotations_path), 'metadata')
            os.makedirs(masks_dir, exist_ok=True)
            os.makedirs(metadata_dir, exist_ok=True)
            
            # 1. Save normalized contour coordinates to labels/*.txt
            label_path = os.path.join(self.annotations_path, f"{base_name}.txt")
            with open(label_path, 'w') as f:
                for annotation in self.annotations:
                    # Get display contour and scale it to original size
                    display_contour = annotation['contour_points']
                    
                    # Scale contour points to original image space
                    scale_x = original_width / self.image.shape[1]
                    scale_y = original_height / self.image.shape[0]
                    
                    original_contour = display_contour.copy()
                    original_contour = original_contour.astype(np.float32)
                    original_contour[:, :, 0] *= scale_x
                    original_contour[:, :, 1] *= scale_y
                    original_contour = original_contour.astype(np.int32)
                    
                    # Write class_id and normalized coordinates
                    line = f"{annotation['class_id']}"
                    for point in original_contour:
                        x, y = point[0]
                        x_norm = x / original_width
                        y_norm = y / original_height
                        line += f" {x_norm:.6f} {y_norm:.6f}"
                    f.write(line + '\n')
            
            # 2. Create visualization at original size
            combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            overlay = original_image.copy()
            
            for annotation in self.annotations:
                # Scale display contour to original size for visualization
                display_contour = annotation['contour_points']
                original_contour = display_contour.copy()
                original_contour = original_contour.astype(np.float32)
                original_contour[:, :, 0] *= scale_x
                original_contour[:, :, 1] *= scale_y
                original_contour = original_contour.astype(np.int32)
                
                # Create mask using the scaled contour
                mask = np.zeros((original_height, original_width), dtype=np.uint8)
                cv2.fillPoly(mask, [original_contour], 1)
                
                # Update combined mask
                combined_mask = np.logical_or(combined_mask, mask)
                
                # Create green overlay
                mask_area = mask > 0
                green_overlay = overlay.copy()
                green_overlay[mask_area] = (0, 255, 0)
                overlay = cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0)
            
            # Convert binary mask to RGB for visualization
            combined_mask_rgb = cv2.cvtColor(combined_mask.astype(np.uint8) * 255, 
                                        cv2.COLOR_GRAY2BGR)
            
            # Create side-by-side visualization
            side_by_side = np.hstack((combined_mask_rgb, overlay))
            
            # Add separator line
            separator_x = original_width
            cv2.line(side_by_side, 
                    (separator_x, 0), 
                    (separator_x, original_height),
                    (0, 0, 255),  # Red line
                    2)
            
            # Save visualization
            mask_path = os.path.join(masks_dir, f"{base_name}_mask{original_ext}")
            cv2.imwrite(mask_path, side_by_side)
            
            # 3. Save metadata with scaling information
            metadata = {
                'num_annotations': len(self.annotations),
                'class_distribution': {str(i): 0 for i in range(len(self.class_names))},
                'image_dimensions': {
                    'original': (original_width, original_height),
                    'display': self.image.shape[:2]
                },
                'scale_factors': {
                    'x': scale_x,
                    'y': scale_y
                }
            }
            
            # Count instances of each class
            for annotation in self.annotations:
                class_id = str(annotation['class_id'])
                metadata['class_distribution'][class_id] = \
                    metadata['class_distribution'].get(class_id, 0) + 1
            
            metadata_path = os.path.join(metadata_dir, f"{base_name}.txt")
            with open(metadata_path, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            self.logger.info(f"Successfully saved annotations to {label_path}")
            self.logger.info(f"Saved mask and overlay to {masks_dir}")
            self.logger.info(f"Saved metadata to {metadata_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in save_annotations: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _load_annotations(self, image_path: str) -> List[Dict]:
        """Load annotations from label file and reconstruct masks."""
        annotations = []
        try:
            # Get paths and check if label exists
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.annotations_path, f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                return annotations
            
            # Get original image dimensions
            original_image = cv2.imread(image_path)
            orig_height, orig_width = original_image.shape[:2]
            
            # Get display dimensions (current image is already resized)
            display_height, display_width = self.image.shape[:2]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    
                    # First convert normalized coordinates to original pixel space
                    orig_points = []
                    for i in range(1, len(parts), 2):
                        x = float(parts[i]) * orig_width
                        y = float(parts[i + 1]) * orig_height
                        orig_points.append([[int(x), int(y)]])
                    
                    # Convert to numpy array for processing
                    orig_contour = np.array(orig_points, dtype=np.int32)
                    
                    # Scale points to display size
                    scale_x = display_width / orig_width
                    scale_y = display_height / orig_height
                    display_contour = orig_contour.copy()
                    display_contour[:, :, 0] = orig_contour[:, :, 0] * scale_x
                    display_contour[:, :, 1] = orig_contour[:, :, 1] * scale_y
                    display_contour = display_contour.astype(np.int32)
                    
                    # Create mask at display size
                    mask = np.zeros((display_height, display_width), dtype=bool)
                    cv2.fillPoly(mask, [display_contour], 1)
                    
                    # Calculate bounding box for display size
                    x, y, w, h = cv2.boundingRect(display_contour)
                    box = [x, y, x + w, y + h]
                    
                    annotations.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'mask': mask,
                        'contour_points': display_contour,
                        'box': box,
                        'original_contour': orig_contour  # Keep original for saving
                    })
                    
            self.logger.info(f"Loaded {len(annotations)} annotations from {label_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        return annotations

    def _prev_image(self) -> None:
        """Move to previous image."""
        if self.current_idx > 0: # check if we can move backward
            # clear current state 
            self.event_handler.reset_state()
            self.window_manager.set_mask(None)
            self.annotations = [] # clear annotations for previous image
            
            # Move to the previous iamge
            self.current_idx -= 1
            self._load_image(os.path.join(self.images_path, self.image_files[self.current_idx]))
            
    def _next_image(self) -> None:
        """ Move to the next image """
        if self.current_idx < len(self.image_files) - 1: # check if we can move forward
            # clear current state 
            self.event_handler.reset_state()
            self.window_manager.set_mask(None)
            self.annotations = [] # clear annotations 
            
            # move to the next image 
            self.current_idx +=1
            self._load_image(os.path.join(self.images_path, self.image_files[self.current_idx]))
     
    def _remove_last_annotation(self) -> None:
        """Remove the last added annotation."""
        if self.annotations:
            self.annotations.pop()
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=len(self.image_files),
                status="Last annotation removed"
            )
    
    def _get_label_path(self, image_path: str) -> str:
            """Get the corresponding label file path for an image."""
            # Assuming image_path is like: test_data/s2/images/img1.jpg
            # We want: test_data/s2/labels/img1.txt
            base_dir = os.path.dirname(os.path.dirname(image_path))  # Gets test_data/s2
            image_name = os.path.basename(image_path)  # Gets img1.jpg
            image_name_without_ext = os.path.splitext(image_name)[0]  # Gets img1
            
            # Construct label path
            label_path = os.path.join(base_dir, 'labels', f"{image_name_without_ext}.txt")
            return label_path

    def _save_annotations_to_file(self) -> None:
            """Save current annotations to label file."""
            try:
                label_path = self._get_label_path(self.current_image_path)
                
                # Ensure the labels directory exists
                os.makedirs(os.path.dirname(label_path), exist_ok=True)
                
                # Create annotation strings
                annotation_lines = []
                for annotation in self.annotations:
                    # Convert contour points to string format
                    points_str = ' '.join([f"{pt[0][0]} {pt[0][1]}" for pt in annotation['contour_points']])
                    # Format: class_id num_points x1 y1 x2 y2 ...
                    line = f"{annotation['class_id']} {len(annotation['contour_points'])} {points_str}"
                    annotation_lines.append(line)
                
                # Write to file
                with open(label_path, 'w') as f:
                    f.write('\n'.join(annotation_lines))
                    
                self.logger.info(f"Saved {len(self.annotations)} annotations to {label_path}")
                
            except Exception as e:
                self.logger.error(f"Error saving annotations to file: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
         
    def _on_annotation_delete(self, idx: int) -> None:
        """Handle annotation deletion."""
        if 0 <= idx < len(self.annotations):
            try:
                # Create and execute delete command
                command = DeleteAnnotationCommand(self.annotations, idx, self.window_manager)
                if self.command_manager.execute(command):
                    # Update main window
                    self.window_manager.update_main_window(
                        image=self.image,
                        annotations=self.annotations,
                        current_class=self.class_names[self.current_class_id],
                        current_class_id=self.current_class_id,
                        current_image_path=self.current_image_path,
                        current_idx=self.current_idx,
                        total_images=len(self.image_files),
                        status=f"Deleted annotation {idx + 1}"
                    )
                    
                    self.logger.info(f"Successfully deleted annotation {idx + 1}")
                    
            except Exception as e:
                self.logger.error(f"Error deleting annotation: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                
    def _handle_undo(self) -> None:
        """Handle undo command."""
        if self.command_manager.undo():
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=len(self.image_files),
                status="Undo successful"
            )
            
    def _handle_redo(self) -> None:
        """Handle redo command."""
        if self.command_manager.redo():
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=len(self.image_files),
                status="Redo successful"
            )
    
   
    """ New export """
    def _handle_export(self, format: str = 'coco') -> None:
        """Handle dataset export request."""
        try:
            # Save current annotations if any
            if self.annotations:
                self._save_annotations()
                
            # Update status
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=len(self.image_files),
                status=f"Exporting dataset to {format.upper()} format..."
            )
            
            # Get base path (directory containing images/labels folders)
            base_path = os.path.dirname(os.path.dirname(self.current_image_path))
            
            if format.lower() == 'coco':
                from ..data.exporters.coco_exporter import CocoExporter
                exporter = CocoExporter(base_path)
            elif format.lower() == 'yolo':
                from ..data.exporters.yolo_exporter import YoloExporter
                exporter = YoloExporter(base_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            # Perform export
            export_path = exporter.export()
            
            # Update status
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=len(self.image_files),
                status=f"Dataset exported to: {export_path}"
            )
            
            self.logger.info(f"Successfully exported dataset to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting dataset: {str(e)}")
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=len(self.image_files),
                status=f"Export failed: {str(e)}"
            )
    
    def run(self) -> None:
            """Run the main annotation loop."""
            try:
                # Get image files
                self.image_files = [f for f in os.listdir(self.images_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not self.image_files:
                    self.logger.error(f"No images found in {self.images_path}")
                    return
                
                # Load first image
                self._load_image(os.path.join(self.images_path, self.image_files[0]))
                
                # Initialize windows
                self.window_manager.update_class_window(
                    self.class_names,
                    self.current_class_id
                )
                
                """ Handle the review annotation panel """

                # Initialize review panel with current annotations
                self.window_manager.update_review_panel(self.annotations)
                """ End """
                
                while True:
                    # Update display
                    self.window_manager.update_main_window(
                        image=self.image,
                        annotations=self.annotations,
                        current_class=self.class_names[self.current_class_id],
                        current_class_id=self.current_class_id,
                        current_image_path=self.current_image_path,
                        current_idx=self.current_idx,
                        total_images=len(self.image_files),
                        box_start=self.event_handler.box_start,
                        box_end=self.event_handler.box_end
                    )
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == -1:
                        continue
                        
                    action = self.event_handler.handle_keyboard_event(key)
                    self.logger.debug(f"Keyboard action: {action}")
                    
                    # Process actions
                    if action == "update_view":
                        self.logger.info("Updating view based on control change")
                        # Force a refresh of the main window
                        self.window_manager.update_main_window(
                            image=self.image,
                            annotations=self.annotations,
                            current_class=self.class_names[self.current_class_id],
                            current_class_id=self.current_class_id,
                            current_image_path=self.current_image_path,
                            current_idx=self.current_idx,
                            total_images=len(self.image_files),
                            box_start=self.event_handler.box_start,
                            box_end=self.event_handler.box_end
                        )
                    elif action == 'quit':
                        break
                    elif action == 'next':
                        self._next_image()
                    elif action == 'prev':
                        self._prev_image()
                    elif action == 'save':
                        self._save_annotations()
                    elif action == 'clear_selection':
                        self.event_handler.reset_state()
                        self.window_manager.set_mask(None)
                    elif action == 'add':
                        self._add_annotation()
                    elif action == 'undo':
                        self._handle_undo()
                        #self._remove_last_annotation()
                    elif action == "redo":
                        self._handle_redo()
                    elif action == 'clear_all':
                        self.annotations = []
                    elif action == "export_coco":
                        self.logger.info("Starting COCO export")
                        self._handle_export('coco')
                    elif action == "export_yolo":
                        self.logger.info("Starting YOLO export")
                        self._handle_export('yolo')
                        
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
            finally:
                # Cleanup
                try:
                    self.window_manager.destroy_windows()
                except Exception as e:
                    self.logger.error(f"Error while cleaning up: {str(e)}")

    
    
