import cv2
import numpy as np
import torch
import os
import time
import logging
from typing import Optional, List, Dict, Tuple
import pandas as pd 

from .file_manager import FileManager
from ..ui.window_manager import WindowManager
from ..ui.event_handler import EventHandler
from ..utils.visualization import VisualizationManager
#from .predictor import SAMPredictor
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

from .base_predictor import BaseSAMPredictor 
from .predictor import SAM1Predictor, SAM2Predictor
from .session_manager import SessionManager


class SAMAnnotator:
    """Main class for SAM-based image annotation."""
    
    def __init__(self, 
                 checkpoint_path: str, 
                 category_path: str, 
                 classes_csv: str,
                 sam_version: str = 'sam1',
                 model_type: str = None
                 ):
        """Initialize the SAM annotator."""
        

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Store SAM version and model type
        self.sam_version = sam_version
        self.model_type = model_type or ('vit_h' if sam_version == 'sam1' else 'small_v2')
        self.logger.info(f"Using SAM version: {sam_version} with model type: {self.model_type}")
        
        # Initialize file manager first (replaces direct DatasetManager initialization)
        self.file_manager = FileManager(category_path, logger=self.logger)
        
        # Initialize image processor
        self.image_processor = ImageProcessor(target_size=1024, min_size=600)
        
        # Initialize managers
        self.window_manager = WindowManager(logger=self.logger)
        self.event_handler = EventHandler(self.window_manager, logger=self.logger)
        self.vis_manager = VisualizationManager()
        self.validation_manager = ValidationManager(self.vis_manager)
        
        # Load SAM model
        self._initialize_model(checkpoint_path)
        
        # Load classes through file manager
        self.class_names = self.file_manager.load_classes(classes_csv)
        
        # Get image files through file manager
        self.dataset_manager = DatasetManager(category_path)
        self.image_files = self.file_manager.get_image_list()
        self.total_images = len(self.image_files)
        
        # Initialize state
        self.current_idx = 0
        self.current_image_path: Optional[str] = None
        self.image: Optional[np.ndarray] = None
        self.annotations: List[Dict] = []
        self.current_class_id = 0
        
        # Add command manager
        self.command_manager = CommandManager()
        
         # Initialize session manager with minimal dependencies
        self.session_manager = SessionManager(
            file_manager=self.file_manager,
            window_manager=self.window_manager,
            event_handler=self.event_handler,
            logger=self.logger
        )
        
        # Setup callbacks
        self._setup_callbacks()
        
          
    def _initialize_model(self, checkpoint_path: str) -> None:
        """Create and initialize appropriate SAM predictor based on version."""
        try:
            # Initialize weight manager
            weight_manager = SAMWeightManager()
            
            # Get appropriate checkpoint path
            verified_checkpoint = weight_manager.get_checkpoint_path(
                user_checkpoint_path=checkpoint_path,
                version=self.sam_version,
                model_type=self.model_type
            )
            
            if self.sam_version == 'sam1':
                self.predictor = SAM1Predictor()
            else:
                self.predictor = SAM2Predictor()
                
            # Initialize the predictor with verified checkpoint
            self.predictor.initialize(verified_checkpoint)
            self.logger.info(f"Successfully initialized {self.sam_version.upper()} predictor with {self.model_type} model")
            
        except Exception as e:
            self.logger.error(f"Error Initializing SAM model: {str(e)}")
            raise
   
    
    """ _setup_paths is an unused method """
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
            on_class_selection=self._handle_class_selection,
            on_point_prediction=self._handle_point_prediction
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
 
  
    """  
        Interesting fact to know about mask prediction Interface:
                - Is version-agnostic at the high level
                - Handles conversion between formats internally
                - Maintains consistent input/output interfaces
    """
    
    def _handle_mask_prediction(self, 
                          box_start: Tuple[int, int],
                          box_end: Tuple[int, int],
                          drawing: bool = False) -> None:
        """Handle mask prediction from box inputs."""
        # If still drawing box, just update display
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
                box_end=box_end,
                annotation_mode=self.event_handler.mode
            )
            return
        
        try:
            # Get memory info before prediction - use safe method
            memory_info = self.predictor.memory_manager.safe_get_memory_info()
            self.logger.info(f"Memory before prediction: {memory_info['formatted']}")
            
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
            
            # Predict mask using either SAM1 or SAM2
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_box,
                multimask_output=True
            )
            
            if len(masks) > 0:
                best_mask_idx = np.argmax(scores) if scores.size > 0 else 0
                best_mask = masks[best_mask_idx]
                
                # Scale the mask to display size
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
                    status="Mask predicted - press 'a' to add",
                    box_start=box_start,
                    box_end=box_end,
                    annotation_mode=self.event_handler.mode
                )
                
            # Get memory info after prediction - use safe method
            memory_info = self.predictor.memory_manager.safe_get_memory_info()
            self.logger.info(f"Memory after prediction: {memory_info['formatted']}")
                
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
        """Load image and its existing annotations."""
        try:
            # Load original image through DatasetManager via FileManager
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            # Process image for display with caching
            display_image, metadata = self.image_processor.process_image(original_image)
            
            self.image = display_image
            self.current_image_path = image_path
            
            # Clear current state
            self.annotations = []  # Clear current annotations
            self.window_manager.set_mask(None)
            
            # Load annotations through FileManager which delegates to DatasetManager
            loaded_annotations = self.file_manager.load_annotations(
                image_path=image_path,
                original_dimensions=(original_image.shape[0], original_image.shape[1]),
                display_dimensions=(display_image.shape[0], display_image.shape[1])
            )
            
            if loaded_annotations:
                self.annotations = loaded_annotations
                self.logger.info(f"Loaded {len(loaded_annotations)} existing annotations")
            
            # Set image in predictor
            self.predictor.set_image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            
            self.logger.info(f"Loaded image: {image_path}")
            self.logger.info(f"Original size: {metadata['original_size']}")
            self.logger.info(f"Display size: {metadata['display_size']}")
            
            # Update windows with loaded annotations
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=self.total_images,
                annotation_mode=self.event_handler.mode
            )
            
            # Update review panel
            self.window_manager.update_review_panel(self.annotations)
            
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise
   
    def _add_annotation(self) -> None:
        """Add current mask as an annotation."""
        try:
            # Ensure we have a valid mask
            if self.window_manager.current_mask is None:
                self.logger.warning("No mask available to add as annotation")
                return
                
            # Convert mask to proper format for contour finding
            mask = self.window_manager.current_mask.astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(
                mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                self.logger.warning("No contours found in mask")
                return
                
            # Get largest contour by area
            display_contour = max(contours, key=cv2.contourArea)
            
            # Get display dimensions
            display_height, display_width = self.image.shape[:2]
            
            # Get original image dimensions
            original_image = cv2.imread(self.current_image_path)
            original_height, original_width = original_image.shape[:2]
            
            # Calculate scale factors
            scale_x = original_width / display_width
            scale_y = original_height / display_height
            
            # Calculate boxes in display coordinates
            x, y, w, h = cv2.boundingRect(display_contour)
            display_box = [x, y, x + w, y + h]
            
            # Calculate box in original coordinates
            original_box = [
                int(x * scale_x),
                int(y * scale_y),
                int((x + w) * scale_x),
                int((y + h) * scale_y)
            ]
            
            # Process contour for storage
            # Store original cv2 contour format (useful for some operations)
            contour_points = display_contour
            
            # Convert to flat list for simpler storage/visualization
            contour_list = display_contour.flatten().tolist()
            
            # Ensure mask is boolean for consistent storage
            clean_mask = self.window_manager.current_mask.copy()
            
            # Get the current SAM version being used
            sam_version = getattr(self.predictor, 'sam_version', 'unknown')
            
            # Create annotation structure
            annotation = {
                'id': len(self.annotations),
                'class_id': self.current_class_id,
                'class_name': self.class_names[self.current_class_id],
                'box': original_box,  # Box in original image coordinates
                'display_box': display_box,  # Box in display coordinates
                'contour_points': contour_points,  # OpenCV contour format
                'contour': contour_list,  # Flattened points for visualization
                'mask': clean_mask,  # Boolean mask
                'area': cv2.contourArea(display_contour),
                'metadata': {
                    'annotation_mode': self.event_handler.mode,
                    'sam_version': sam_version,
                    'timestamp': time.time()
                }
            }
            
            # Create and execute add command
            command = AddAnnotationCommand(self.annotations, annotation, self.window_manager)
            self.command_manager.execute(command)
            
            # Reset state 
            self.window_manager.set_mask(None)
            self.event_handler.reset_state()
            
            # Update main window
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=len(self.image_files),
                status=f"Added {self.class_names[self.current_class_id]} annotation",
                annotation_mode=self.event_handler.mode
            )
            
            # Update review panel
            self.window_manager.update_review_panel(self.annotations)
            
            self.logger.info(f"Added annotation for class {self.class_names[self.current_class_id]}")
            
        except Exception as e:
            self.logger.error(f"Error adding annotation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _save_annotations(self) -> bool:
        """Save annotations to file."""
        self.logger.info(f"Starting save_annotations. Number of annotations: {len(self.annotations)}")
        
        try:
            # Skip validation if there are annotations that need to be saved
            if len(self.annotations) > 0:
                self.logger.info(f"Attempting to save {len(self.annotations)} annotations")
                
                # Make sure all annotations have the required fields
                valid_annotations = []
                for annotation in self.annotations:
                    # Ensure we have the mask
                    if 'mask' not in annotation or annotation['mask'] is None:
                        self.logger.warning(f"Missing mask in annotation {annotation.get('id', '?')}")
                        continue
                        
                    # Ensure we have box coordinates
                    if 'box' not in annotation and 'display_box' not in annotation:
                        self.logger.warning(f"Missing box coordinates in annotation {annotation.get('id', '?')}")
                        continue
                        
                    valid_annotations.append(annotation)
                
                if not valid_annotations:
                    self.logger.warning("No valid annotations to save")
                    self.window_manager.update_main_window(
                        image=self.image,
                        annotations=self.annotations,
                        current_class=self.class_names[self.current_class_id],
                        current_class_id=self.current_class_id,
                        current_image_path=self.current_image_path,
                        current_idx=self.current_idx,
                        total_images=len(self.image_files),
                        status="No valid annotations to save",
                        annotation_mode=self.event_handler.mode
                    )
                    return False
                
                # Save through FileManager
                self.logger.info(f"Saving {len(valid_annotations)} annotations")
                
                # Get the original image dimensions
                original_image = cv2.imread(self.current_image_path)
                if original_image is None:
                    self.logger.error(f"Could not load original image to get dimensions: {self.current_image_path}")
                    return False
                    
                original_dimensions = (original_image.shape[0], original_image.shape[1])
                display_dimensions = (self.image.shape[0], self.image.shape[1])
                
                saved = self.file_manager.save_annotations(
                    annotations=valid_annotations,
                    image_name=os.path.basename(self.current_image_path),
                    original_dimensions=original_dimensions,
                    display_dimensions=display_dimensions,
                    class_names=self.class_names
                )
                
                if saved:
                    self.logger.info(f"Successfully saved {len(valid_annotations)} annotations")
                    self.window_manager.update_main_window(
                        image=self.image,
                        annotations=self.annotations,
                        current_class=self.class_names[self.current_class_id],
                        current_class_id=self.current_class_id,
                        current_image_path=self.current_image_path,
                        current_idx=self.current_idx,
                        total_images=len(self.image_files),
                        status=f"Saved {len(valid_annotations)} annotations",
                        annotation_mode=self.event_handler.mode
                    )
                    return True
                else:
                    self.logger.error("Failed to save annotations")
                    self.window_manager.update_main_window(
                        image=self.image,
                        annotations=self.annotations,
                        current_class=self.class_names[self.current_class_id],
                        current_class_id=self.current_class_id,
                        current_image_path=self.current_image_path,
                        current_idx=self.current_idx,
                        total_images=len(self.image_files),
                        status="Failed to save annotations",
                        annotation_mode=self.event_handler.mode
                    )
                    return False
            else:
                self.logger.warning("No annotations to save")
                self.window_manager.update_main_window(
                    image=self.image,
                    annotations=self.annotations,
                    current_class=self.class_names[self.current_class_id],
                    current_class_id=self.current_class_id,
                    current_image_path=self.current_image_path,
                    current_idx=self.current_idx,
                    total_images=len(self.image_files),
                    status="No annotations to save",
                    annotation_mode=self.event_handler.mode
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error in save_annotations: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=len(self.image_files),
                status=f"Error saving: {str(e)}",
                annotation_mode=self.event_handler.mode
            )
            return False
    
    
    def _load_annotations(self, image_path: str) -> List[Dict]:
        """Load annotations from label file and reconstruct masks."""
        annotations = []
        try:
            # Get paths and check if label exists
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.annotations_path, f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                self.logger.info(f"No annotations file found at {label_path}")
                return annotations
            
            self.logger.info(f"Loading annotations from {label_path}")
            
            # Get original image dimensions
            original_image = cv2.imread(image_path)
            if original_image is None:
                self.logger.error(f"Failed to read original image at {image_path}")
                return annotations
            
            orig_height, orig_width = original_image.shape[:2]
            
            # Get display dimensions (current image is already resized)
            display_height, display_width = self.image.shape[:2]
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
                self.logger.info(f"Found {len(lines)} annotation lines")
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 3:  # Need at least class_id and one point (2 numbers)
                        self.logger.warning(f"Invalid annotation line: {line}")
                        continue
                        
                    class_id = int(parts[0])
                    
                    # First convert normalized coordinates to original pixel space
                    orig_points = []
                    for i in range(1, len(parts), 2):
                        if i+1 < len(parts):  # Ensure we have both x and y
                            x = float(parts[i]) * orig_width
                            y = float(parts[i + 1]) * orig_height
                            orig_points.append([[int(x), int(y)]])
                    
                    if not orig_points:
                        self.logger.warning(f"No valid points found in line: {line}")
                        continue
                    
                    # Convert to numpy array for processing
                    orig_contour = np.array(orig_points, dtype=np.int32)
                    
                    # Scale points to display size
                    scale_x = display_width / orig_width
                    scale_y = display_height / orig_height
                    display_contour = orig_contour.copy()
                    display_contour[:, :, 0] = (orig_contour[:, :, 0] * scale_x).astype(np.int32)
                    display_contour[:, :, 1] = (orig_contour[:, :, 1] * scale_y).astype(np.int32)
                    
                    # Create mask at display size
                    mask = np.zeros((display_height, display_width), dtype=np.uint8)
                    try:
                        cv2.fillPoly(mask, [display_contour], 1)
                        # Convert to boolean for consistency
                        mask_bool = mask.astype(bool)
                        
                        # Debug mask creation
                        if not np.any(mask):
                            self.logger.warning(f"Empty mask created from contour with {len(display_contour)} points")
                            continue
                    except Exception as e:
                        self.logger.error(f"Error creating mask from contour: {e}")
                        continue
                    
                    # Calculate bounding box for display size
                    x, y, w, h = cv2.boundingRect(display_contour)
                    display_box = [x, y, x + w, y + h]
                    
                    # Calculate original box
                    orig_x = int(x / scale_x)
                    orig_y = int(y / scale_y)
                    orig_w = int(w / scale_x)
                    orig_h = int(h / scale_y)
                    original_box = [orig_x, orig_y, orig_x + orig_w, orig_y + orig_h]
                    
                    # Prepare contour in format expected by visualization
                    # Flatten contour for compatibility with visualization
                    contour_list = display_contour.tolist()
                    if len(contour_list) > 0 and isinstance(contour_list[0], list) and len(contour_list[0]) == 1:
                        contour_list = [point[0] for point in contour_list]
                    
                    # Create annotation with all required fields for both new and old format
                    annotation = {
                        'id': len(annotations),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id] if class_id in self.class_names else f"Class {class_id}",
                        'mask': mask_bool,  # Boolean mask
                        'contour': contour_list,  # Flattened contour for new format
                        'contour_points': display_contour,  # Original format expected by some functions
                        'box': original_box,  # Box in original image coordinates
                        'display_box': display_box,  # Box in display coordinates
                        'area': cv2.contourArea(display_contour),
                        'original_contour': orig_contour,  # Keep for reference
                        'metadata': {
                            'annotation_mode': 'imported',
                            'timestamp': time.time()
                        }
                    }
                    
                    annotations.append(annotation)
                    
            self.logger.info(f"Successfully loaded {len(annotations)} annotations from {label_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        return annotations

    def _prev_image(self) -> None:
        """Move to previous image."""
        if next_path := self.session_manager.prev_image():
            # Clear local annotations
            self.annotations = []
            
            # Update index and load new image
            self.current_idx = self.session_manager.current_idx
            self._load_image(next_path)

    def _next_image(self) -> None:
        """Move to next image."""
        if next_path := self.session_manager.next_image():
            # Clear local annotations
            self.annotations = []
            
            # Update index and load new image
            self.current_idx = self.session_manager.current_idx
            self._load_image(next_path)
         
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
                        status=f"Deleted annotation {idx + 1}",
                        annotation_mode=self.event_handler.mode
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
                status="Undo successful",
                annotation_mode=self.event_handler.mode
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
                status="Redo successful",
                annotation_mode=self.event_handler.mode
            )
    
   
    def _handle_export(self, format: str = 'coco') -> None:
        """Handle dataset export using delegating FileManager."""
        try:
            self.logger.info(f"Initiating export to {format} format")  # Add this debug line
            # Save current annotations if any
            if self.annotations:
                self._save_annotations()
                
            # Update status before export
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=self.total_images,
                status=f"Exporting dataset to {format.upper()} format...",
                annotation_mode=self.event_handler.mode
            )
            
            # Export through FileManager
            export_path = self.file_manager.handle_export(
                format=format,
                class_names=self.class_names
            )
            
            if export_path:
                status_msg = f"Dataset exported to: {export_path}"
                self.logger.info(status_msg)
            else:
                status_msg = "Export failed"
                self.logger.error(status_msg)
                
            # Update status after export
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=self.total_images,
                status=status_msg,
                annotation_mode=self.event_handler.mode
            )
            
        except Exception as e:
            self.logger.error(f"Error exporting dataset: {str(e)}")
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=self.total_images,
                status=f"Export failed: {str(e)}",
                annotation_mode=self.event_handler.mode
            )
    
    def _handle_point_prediction(self, points: List[List[int]], point_labels: List[int]) -> None:
        """Handle mask prediction from point inputs.
        
        Args:
            points: List of point coordinates
            point_labels: List of point labels (1=foreground, 0=background)
        """
        if not points or len(points) == 0:
            return
            
        # Import traceback at the top level to avoid UnboundLocalError
        import traceback
        
        try:
            # Get memory info before prediction - use safe method
            memory_info = self.predictor.memory_manager.safe_get_memory_info()
            self.logger.info(f"Memory before prediction: {memory_info['formatted']}")
            
            # Get display and original dimensions
            display_height, display_width = self.image.shape[:2]
            original_image = cv2.imread(self.current_image_path)
            original_height, original_width = original_image.shape[:2]
            
            # Calculate scale factors
            scale_x = original_width / display_width
            scale_y = original_height / display_height
            
            # Check which SAM version we're using
            is_sam2 = hasattr(self.predictor, 'sam_version') and self.predictor.sam_version == 'sam2'
            
            if is_sam2:
                # Scale the coordinates to original image size for SAM2
                if len(points) > 0:
                    # Focus on the first point only as SAM2 has issues with multiple points
                    point = points[0]
                    label = point_labels[0]
                    
                    # Convert to original image coordinates
                    orig_x = float(point[0] * scale_x)
                    orig_y = float(point[1] * scale_y)
                    
                    self.logger.info(f"Using SAM2 with point prompt: [{orig_x}, {orig_y}], label={label}")
                    
                    try:
                        # Direct call to Ultralytics SAM2 model following documentation format
                        results = self.predictor.model(
                            source=self.predictor.current_image,
                            points=[orig_x, orig_y],  # single point [x, y]
                            labels=[label]            # single label
                        )
                        
                        # Process the results
                        if len(results) > 0 and results[0].masks is not None:
                            # Get the mask data
                            masks = results[0].masks.data.cpu().numpy()
                            
                            # Handle single mask case
                            if len(masks.shape) == 2:
                                masks = np.expand_dims(masks, 0)
                            
                            # Use confidence scores if available
                            scores = results[0].conf.cpu().numpy() if hasattr(results[0], 'conf') else np.ones(len(masks))
                            
                            # Select best mask
                            best_mask_idx = np.argmax(scores) if scores.size > 0 else 0
                            best_mask = masks[best_mask_idx]
                            
                            # Scale to display size
                            display_mask = cv2.resize(
                                best_mask.astype(np.uint8),
                                (display_width, display_height),
                                interpolation=cv2.INTER_NEAREST
                            )
                            
                            # Update UI
                            self.window_manager.set_mask(display_mask.astype(bool))
                            self.logger.info("Successfully generated mask using point-based prompt for SAM2")
                        else:
                            self.logger.warning("No masks returned from SAM2 prediction with point prompt")
                            
                            # Fall back to box-based approach if point-based fails
                            self.logger.info("Falling back to box-based approach")
                            
                            # Create a box around the point
                            padding = 20
                            min_x = max(0, point[0] - padding)
                            min_y = max(0, point[1] - padding)
                            max_x = min(display_width, point[0] + padding)
                            max_y = min(display_height, point[1] + padding)
                            
                            # Scale box to original coordinates
                            orig_box = np.array([
                                min_x * scale_x,
                                min_y * scale_y,
                                max_x * scale_x,
                                max_y * scale_y
                            ])
                            
                            # Call with bounding box
                            results = self.predictor.model(
                                source=self.predictor.current_image,
                                bboxes=[orig_box.tolist()]
                            )
                            
                            if len(results) > 0 and results[0].masks is not None:
                                masks = results[0].masks.data.cpu().numpy()
                                if len(masks.shape) == 2:
                                    masks = np.expand_dims(masks, 0)
                                
                                scores = results[0].conf.cpu().numpy() if hasattr(results[0], 'conf') else np.ones(len(masks))
                                best_mask_idx = np.argmax(scores) if scores.size > 0 else 0
                                best_mask = masks[best_mask_idx]
                                
                                display_mask = cv2.resize(
                                    best_mask.astype(np.uint8),
                                    (display_width, display_height),
                                    interpolation=cv2.INTER_NEAREST
                                )
                                
                                self.window_manager.set_mask(display_mask.astype(bool))
                                self.logger.info("Successfully generated mask using box-based fallback for SAM2")
                    except Exception as e:
                        self.logger.error(f"Error with SAM2 point prediction: {str(e)}")
                        self.logger.error(traceback.format_exc())
            else:
                # Format for SAM1 - separate points and labels
                input_points = []
                for point in points:
                    orig_x = int(point[0] * scale_x)
                    orig_y = int(point[1] * scale_y)
                    input_points.append([orig_x, orig_y])
                
                # Convert to numpy arrays
                input_points = np.array(input_points)
                input_labels = np.array(point_labels)
                
                # Predict mask using SAM1
                masks, scores, _ = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )
                
                if len(masks) > 0:
                    best_mask_idx = np.argmax(scores) if scores.size > 0 else 0
                    best_mask = masks[best_mask_idx]
                    
                    # Scale the mask to display size
                    display_mask = cv2.resize(
                        best_mask.astype(np.uint8),
                        (display_width, display_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    self.window_manager.set_mask(display_mask.astype(bool))
            
            # Update UI with the new mask (for both SAM1 and SAM2)
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=len(self.image_files),
                status="Mask predicted - press 'a' to add",
                input_points=points,
                input_point_labels=point_labels,
                annotation_mode=self.event_handler.mode
            )
                
            # Get memory info after prediction - use safe method
            memory_info = self.predictor.memory_manager.safe_get_memory_info()
            self.logger.info(f"Memory after prediction: {memory_info['formatted']}")
                
        except Exception as e:
            self.logger.error(f"Error in point-based mask prediction: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def run(self) -> None:
        """Run the main annotation loop with FileManager integration."""
        try:
            # Get image files through FileManager
            self.image_files = self.file_manager.get_image_list()
            if not self.image_files:
                self.logger.error(f"No images found in {self.file_manager.structure['images']}")
                return
            
            # Find the first unannotated image or last image if all are annotated
            self.current_idx = self.file_manager.get_last_annotated_index()
            
            # Load first image
            self._load_image(str(self.file_manager.structure['images'] / self.image_files[self.current_idx]))
            
            # Initialize windows
            self.window_manager.update_class_window(
                self.class_names,
                self.current_class_id
            )
            
            # Initialize review panel with current annotations
            self.window_manager.update_review_panel(self.annotations)
            
            while True:
                # Periodically check memory usage
                if hasattr(self.image_processor, 'get_memory_usage'):
                    memory_usage = self.image_processor.get_memory_usage()
                    if memory_usage > 1e9:  # More than 1GB
                        self.logger.info("Clearing image cache due to high memory usage")
                        self.image_processor.clear_cache()
                        
                # Check GPU memory periodically
                if hasattr(self.predictor, 'get_memory_usage'):
                    gpu_memory = self.predictor.get_memory_usage()
                    if gpu_memory > 0.8:  # Over 80% GPU memory
                        self.predictor.optimize_memory()
                        
                # Update display
                self.window_manager.update_main_window(
                    image=self.image,
                    annotations=self.annotations,
                    current_class=self.class_names[self.current_class_id],
                    current_class_id=self.current_class_id,
                    current_image_path=self.current_image_path,
                    current_idx=self.current_idx,
                    total_images=self.total_images,
                    box_start=self.event_handler.box_start,
                    box_end=self.event_handler.box_end,
                    input_points=self.event_handler.points if self.event_handler.mode == 'point' else None,
                    input_point_labels=self.event_handler.point_labels if self.event_handler.mode == 'point' else None,
                    annotation_mode=self.event_handler.mode
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
                        box_end=self.event_handler.box_end,
                        input_points=self.event_handler.points if self.event_handler.mode == 'point' else None,
                        input_point_labels=self.event_handler.point_labels if self.event_handler.mode == 'point' else None,
                        annotation_mode=self.event_handler.mode
                    )
                elif action == 'switch_mode_point' or action == 'switch_mode_box':
                    # Update window with new mode
                    mode = 'point' if action == 'switch_mode_point' else 'box'
                    self.window_manager.update_main_window(
                        image=self.image,
                        annotations=self.annotations,
                        current_class=self.class_names[self.current_class_id],
                        current_class_id=self.current_class_id,
                        current_image_path=self.current_image_path,
                        current_idx=self.current_idx,
                        total_images=len(self.image_files),
                        annotation_mode=mode,
                        status=f"Switched to {mode} annotation mode"
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
                elif action == "export_pascal":
                    self.logger.info("Starting Pascal VOC export")
                    self._handle_export('pascal')
                                    
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
        finally:
            # Cleanup
            try:
                self.window_manager.destroy_windows()
            except Exception as e:
                self.logger.error(f"Error while cleaning up: {str(e)}")

    
    
