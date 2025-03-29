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
        self._create_predictor(checkpoint_path)
        
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
        
        # Setup callbacks
        self._setup_callbacks()
        
        
        
    def _create_predictor(self, checkpoint_path: str) -> None:
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
   
    """ Similar method being used by _create_predictor"""
    def _initialize_model(self, checkpoint_path: str) -> None:
        """Initialize SAM model based on version."""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Using device: {device}")

            if self.sam_version == 'sam1':
                # Initialize original SAM
                weight_manager = SAMWeightManager()
                verified_checkpoint_path = weight_manager.get_checkpoint_path(checkpoint_path)
                
                sam = sam_model_registry["vit_h"](checkpoint=verified_checkpoint_path)
                sam.to(device=device)
                self.predictor = SamPredictor(sam)
                
            else:  # sam2
                # Initialize SAM2 through Ultralytics
                self.predictor = SAM2(checkpoint_path)
                self.latest_results = None  # Store latest SAM2 results

            self.logger.info(f"SAM{self.sam_version[-1]} model initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Error initializing SAM model: {str(e)}")
            raise
    
    """ _load_classes function moved to file_manager. This one is unused"""
    def _load_classes(self, classes_csv: str) -> None:
        """Load class names from CSV."""
        try:
            df = pd.read_csv(classes_csv)
            self.class_names = df['class_name'].tolist()[:15]
            self.logger.info(f"Loaded {len(self.class_names)} classes")
        except Exception as e:
            self.logger.error(f"Error loading classes: {str(e)}")
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
            # Get memory info before prediction
            memory_info = self.predictor.memory_manager.get_gpu_memory_info()
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
                    status="Mask predicted - press 'a' to add"
                )
                
            # Get memory info after prediction
            memory_info = self.predictor.memory_manager.get_gpu_memory_info()
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
    
    # In annotator.py, in _load_image method:
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
                total_images=self.total_images
            )
            
            # Update review panel
            self.window_manager.update_review_panel(self.annotations)
            
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
        """Save annotations using FileManager."""
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
            
            # Get original image dimensions
            original_image = cv2.imread(self.current_image_path)
            if original_image is None:
                raise ValueError(f"Could not load original image: {self.current_image_path}")
                
            success = self.file_manager.save_annotations(
                annotations=self.annotations,
                image_name=os.path.basename(self.current_image_path),
                original_dimensions=original_image.shape[:2],
                display_dimensions=self.image.shape[:2],
                class_names=self.class_names,
                save_visualization=True
            )
            
            if success:
                self.logger.info("Successfully saved annotations")
                self.window_manager.update_main_window(
                    image=self.image,
                    annotations=self.annotations,
                    current_class=self.class_names[self.current_class_id],
                    current_class_id=self.current_class_id,
                    current_image_path=self.current_image_path,
                    current_idx=self.current_idx,
                    total_images=self.total_images,
                    status="Annotations saved successfully"
                )
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error in save_annotations: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
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
            #self._load_image(os.path.join(self.images_path, self.image_files[self.current_idx]))
            # Use file_manager's structure to get the full path
            image_path = str(self.file_manager.structure['images'] / self.image_files[self.current_idx])
            #self._load_image(os.path.join(self.images_path, self.image_files[self.current_idx]))
            
            self._load_image(image_path)

            
    def _next_image(self) -> None:
        """ Move to the next image """
        if self.current_idx < len(self.image_files) - 1: # check if we can move forward
            # clear current state 
            self.event_handler.reset_state()
            self.window_manager.set_mask(None)
            self.annotations = [] # clear annotations 
            
            # move to the next image 
            self.current_idx +=1
            #self._load_image(os.path.join(self.images_path, self.image_files[self.current_idx]))
            # Use file_manager's structure to get the full path
            image_path = str(self.file_manager.structure['images'] / self.image_files[self.current_idx])
            
            #self._load_image(os.path.join(self.images_path, self.image_files[self.current_idx]))
            self._load_image(image_path)
        
     
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
                status=f"Exporting dataset to {format.upper()} format..."
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
                status=status_msg
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
                status=f"Export failed: {str(e)}"
            )
    
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

    
    
