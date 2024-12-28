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
        """Add current annotation to the list with full mask."""
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
            # Get current box from event handler
            box_start = self.event_handler.box_start
            box_end = self.event_handler.box_end
            
            if not box_start or not box_end:
                self.logger.warning("Box coordinates not found")
                return

            # Convert mask to uint8 and find contours
            mask_uint8 = (current_mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            
            if not contours:
                self.logger.warning("No valid contours found in mask")
                return
                
            # Get the largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour to reduce noise while keeping accuracy
            epsilon = 0.0005 * cv2.arcLength(contour, True)  # Reduced epsilon for more accuracy
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Store annotation with both mask and contour
            annotation = {
                'class_id': self.current_class_id,
                'class_name': self.class_names[self.current_class_id],
                'mask': current_mask.copy(),
                'contour_points': approx_contour,
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
            self.window_manager.update_review_panel(self.annotations)
            
        except Exception as e:
            self.logger.error(f"Error adding annotation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    
    
    def _save_annotations(self) -> bool:
        """Save annotations in multiple formats:
        1. labels/*.txt: normalized contour coordinates
        2. masks/*.png: visual mask representation
        3. metadata/*: additional information
        """
        try:
            self.logger.info(f"Starting save_annotations. Number of annotations: {len(self.annotations)}")
            
            if not self.annotations:
                self.logger.warning("No annotations to save!")
                return False

            # Get image dimensions and base paths
            height, width = self.image.shape[:2]
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
                    # Write class_id and normalized contour points
                    contour = annotation['contour_points']
                    line = f"{annotation['class_id']}"
                    for point in contour:
                        x, y = point[0]
                        x_norm = x / width
                        y_norm = y / height
                        line += f" {x_norm:.6f} {y_norm:.6f}"
                    f.write(line + '\n')
            
            # 2. Save visual mask and overlay side by side
            # Create combined mask image
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            overlay = self.image.copy()
            
            for annotation in self.annotations:
                mask = annotation['mask']
                combined_mask = np.logical_or(combined_mask, mask)
                
                # Create green overlay for this mask
                mask_area = mask > 0
                green_overlay = overlay.copy()
                green_overlay[mask_area] = (0, 255, 0)  # Pure green for masked areas
                
                # Blend with original image
                overlay = cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0)
            
            # Convert binary mask to 3-channel for visualization
            combined_mask_rgb = cv2.cvtColor(combined_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            
            # Create side-by-side visualization
            side_by_side = np.hstack((combined_mask_rgb, overlay))
            
            # Add a vertical line between the images
            separator_x = width
            cv2.line(side_by_side, 
                    (separator_x, 0), 
                    (separator_x, height), 
                    (0, 0, 255),  # Red line
                    2)  # Line thickness
            
            # Save combined visualization with same extension as original
            mask_path = os.path.join(masks_dir, f"{base_name}_mask{original_ext}")
            cv2.imwrite(mask_path, side_by_side)
            
            # 3. Save metadata (optional)
            metadata = {
                'num_annotations': len(self.annotations),
                'class_distribution': {str(i): 0 for i in range(len(self.class_names))}
            }
            
            # Count instances of each class
            for annotation in self.annotations:
                class_id = str(annotation['class_id'])
                metadata['class_distribution'][class_id] += 1
                
            metadata_path = os.path.join(metadata_dir, f"{base_name}.txt")
            with open(metadata_path, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            self.logger.info(f"Successfully saved annotations to {label_path}")
            self.logger.info(f"Saved mask and overlay to {masks_dir}")
            self.logger.info(f"Saved metadata to {metadata_path}")
            
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
    
    
    def _load_annotations(self, image_path: str) -> List[Dict]:
        """Load annotations from label file and reconstruct masks."""
        annotations = []
        try:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.annotations_path, f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                return annotations
                
            height, width = self.image.shape[:2]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    
                    # Convert normalized coordinates back to pixel space
                    points = []
                    for i in range(1, len(parts), 2):
                        x = float(parts[i]) * width
                        y = float(parts[i + 1]) * height
                        points.append([[int(x), int(y)]])
                    
                    contour = np.array(points, dtype=np.int32)
                    
                    # Create mask from contour
                    mask = np.zeros((height, width), dtype=bool)
                    cv2.fillPoly(mask, [contour], 1)
                    
                    # Calculate bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    box = [x, y, x + w, y + h]
                    
                    annotations.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'mask': mask,
                        'contour_points': contour,
                        'box': box
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

    