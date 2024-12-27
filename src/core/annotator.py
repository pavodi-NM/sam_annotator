import cv2
import numpy as np
import torch
import os
import logging
from typing import Optional, List, Dict, Tuple
import pandas as pd

from ..ui.window_manager import WindowManager
from ..ui.event_handler import EventHandler
from ..utils.visualization import VisualizationManager
from .predictor import SAMPredictor
from .weight_manager import SAMWeightManager 

class SAMAnnotator:
    """Main class for SAM-based image annotation."""
    
    def __init__(self, checkpoint_path: str, category_path: str, classes_csv: str):
        """Initialize the SAM annotator."""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize managers
        self.window_manager = WindowManager(logger=self.logger)
        self.event_handler = EventHandler(self.window_manager, logger=self.logger)
        self.vis_manager = VisualizationManager()
        
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
        
        # Initialize UI components with logger
        # self.window_manager = WindowManager(logger=self.logger)
        # self.event_handler = EventHandler(self.window_manager, logger=self.logger)
        
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
        
        def _on_annotation_delete(self, idx: int) -> None:
            """Handle annotation deletion."""
            if 0 <= idx < len(self.annotations):
                del self.annotations[idx]
                self.window_manager.update_review_panel(self.annotations)

        def _on_annotation_select(self, idx: int) -> None:
            """Handle annotation selection."""
            if 0 <= idx < len(self.annotations):
                self.selected_annotation_idx = idx
                # Update the main window to highlight selected annotation
                # This will be handled through the window manager's update mechanism

        def _on_annotation_class_change(self, idx: int, new_class_id: int) -> None:
            """Handle annotation class change."""
            if 0 <= idx < len(self.annotations):
                self.annotations[idx]['class_id'] = new_class_id
                self.annotations[idx]['class_name'] = self.class_names[new_class_id]
                self.window_manager.update_review_panel(self.annotations)
    
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
            # Prepare points and box
            center_x = (box_start[0] + box_end[0]) // 2
            center_y = (box_start[1] + box_end[1]) // 2
            input_points = np.array([[center_x, center_y]])
            input_labels = np.array([1])
            input_box = np.array([
                min(box_start[0], box_end[0]),
                min(box_start[1], box_end[1]),
                max(box_start[0], box_end[0]),
                max(box_start[1], box_end[1])
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
                self.window_manager.set_mask(masks[best_mask_idx])
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
            self.image = cv2.imread(image_path)
            self.current_image_path = image_path
            self.predictor.set_image(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            self.annotations = []
            self.window_manager.set_mask(None)
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise
     
    
    def _add_annotation(self) -> None:
        """Add current annotation to the list."""
        self.logger.info("Attempting to add annotation...")
        
        # Get current mask from window manager
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
            # Convert mask to more memory-efficient format
            mask_uint8 = (current_mask.astype(np.uint8)) * 255
            
            # Find contours immediately and store only the points
            contours, _ = cv2.findContours(mask_uint8, 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.logger.warning("No valid contours found in mask")
                return
                
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get current box from event handler
            box_start = self.event_handler.box_start
            box_end = self.event_handler.box_end
            
            if not box_start or not box_end:
                self.logger.warning("Box coordinates not found")
                return

            # Store only the approximated contour points instead of full mask
            annotation = {
                'class_id': self.current_class_id,
                'class_name': self.class_names[self.current_class_id],  # Add class_name for review panel
                'contour_points': approx.copy(),
                'box': [min(box_start[0], box_end[0]),
                       min(box_start[1], box_end[1]),
                       max(box_start[0], box_end[0]),
                       max(box_start[1], box_end[1])] 
            }
            self.annotations.append(annotation)
            self.logger.info(f"Successfully added annotation. Total annotations: {len(self.annotations)}")
            
            # Clear current selection
            self.event_handler.reset_state()
            self.window_manager.set_mask(None)
            
            # Update both main window and review panel
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
            
            # Update review panel with current annotations
            self.window_manager.update_review_panel(self.annotations)
            
        except Exception as e:
            self.logger.error(f"Error adding annotation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
      
    
    def _save_annotations(self) -> bool:
        """Save all annotations for the current image."""
        try:
            self.logger.info(f"Starting save_annotations. Number of annotations: {len(self.annotations)}")
            
            if not self.annotations:
                self.logger.warning("No annotations to save!")
                return False

            height, width = self.image.shape[:2]
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            anno_path = os.path.join(self.annotations_path, f"{base_name}.txt")
            
            self.logger.info(f"Saving to: {anno_path}")
            
            # Clear existing annotations for this image
            with open(anno_path, 'w') as f:
                for idx, annotation in enumerate(self.annotations):
                    self.logger.info(f"Processing annotation {idx + 1}")
                    contour_points = annotation['contour_points']
                    class_id = annotation['class_id']
                    
                    # Write annotation line
                    line = f"{class_id}"
                    for point in contour_points:
                        x, y = point[0]
                        x_norm = x / width
                        y_norm = y / height
                        line += f" {x_norm:.6f} {y_norm:.6f}"
                    f.write(line + '\n')
                    self.logger.info(f"Wrote annotation {idx + 1} with {len(contour_points)} points")
            
            self.logger.info(f"Successfully saved {len(self.annotations)} annotations to {anno_path}")
            
            # Show save confirmation
            self.window_manager.update_main_window(
                image=self.image,
                annotations=self.annotations,
                current_class=self.class_names[self.current_class_id],
                current_class_id=self.current_class_id,
                current_image_path=self.current_image_path,
                current_idx=self.current_idx,
                total_images=len(self.image_files),
                status="All annotations saved!"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Error in save_annotations: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
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
    
    
    """ Handle the annotation review panel """


    def _on_annotation_select(self, idx: int) -> None:
        """Handle annotation selection."""
        # Highlight the selected annotation in the main window
        self.selected_annotation_idx = idx
        # Update visualization...

    def _on_annotation_class_change(self, idx: int, new_class_id: int) -> None:
        """Handle annotation class change."""
        if 0 <= idx < len(self.annotations):
            self.annotations[idx]['class_id'] = new_class_id
            self.annotations[idx]['class_name'] = self.class_names[new_class_id]
            self.window_manager.update_review_panel(self.annotations)    
    
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
                # Store info for logging
                deleted_class = self.annotations[idx]['class_id']
                
                # Remove from annotations list
                del self.annotations[idx]
                
                # Update review panel
                self.window_manager.update_review_panel(self.annotations)
                
                # Save updated annotations to file
                self._save_annotations_to_file()
                
                # Update main window
                self.window_manager.update_main_window(
                    image=self.image,
                    annotations=self.annotations,
                    current_class=self.class_names[self.current_class_id],
                    current_class_id=self.current_class_id,
                    current_image_path=self.current_image_path,
                    current_idx=self.current_idx,
                    total_images=len(self.image_files),
                    status=f"Deleted annotation {idx + 1} (Class: {self.class_names[deleted_class]})"
                )
                
                self.logger.info(f"Successfully deleted annotation {idx + 1} of class {self.class_names[deleted_class]}")
                
            except Exception as e:
                self.logger.error(f"Error deleting annotation: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
    
    
    
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
                        self._remove_last_annotation()
                    elif action == 'clear_all':
                        self.annotations = []
                        
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
            finally:
                # Cleanup
                try:
                    self.window_manager.destroy_windows()
                except Exception as e:
                    self.logger.error(f"Error while cleaning up: {str(e)}")

    