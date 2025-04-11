import logging
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from ..ui.window_manager import WindowManager
from ..ui.event_handler import EventHandler
from ..utils.visualization import VisualizationManager
from ..utils.image_utils import ImageProcessor
from ..core.validation import ValidationManager
from ..core.command_manager import CommandManager
from ..core.weight_manager import SAMWeightManager
from ..core.base_predictor import BaseSAMPredictor
from ..core.predictor import SAM1Predictor, SAM2Predictor


from .file_manager import FileManager
from .annotation_manager import AnnotationManager 
from .session_manager import SessionManager
"""Initialize SAM annotator with all components.

Args:
    checkpoint_path: Path to SAM model checkpoint
    category_path: Path to category folder
    classes_csv: Path to class definitions
    sam_version: SAM version to use ('sam1' or 'sam2')
    model_type: Model type specification
"""

class SAMAnnotator:
    """Main class coordinating SAM-based image annotation."""
    def __init__(self, 
                checkpoint_path: str,
                category_path: str,
                classes_csv: str,
                sam_version: str = 'sam1',
                model_type: str = None):
        """Initialize SAM annotator with all components."""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Store SAM configuration
        self.sam_version = sam_version
        self.model_type = model_type or ('vit_h' if sam_version == 'sam1' else 'small_v2')
        
        # Initialize image processor
        self.image_processor = ImageProcessor(target_size=1024, min_size=600)
        
        # Initialize managers
        self.window_manager = WindowManager(logger=self.logger)
        self.event_handler = EventHandler(self.window_manager, logger=self.logger)
        self.vis_manager = VisualizationManager()
        
        # Initialize managers and validation
        self.validation_manager = ValidationManager(self.vis_manager)
        self.file_manager = FileManager(category_path, self.image_processor)
        self.command_manager = CommandManager()
        
        # Initialize annotation manager
        self.annotation_manager = AnnotationManager(
            self.validation_manager,
            self.window_manager,
            self.command_manager
        )

        # Initialize session manager without predictor
        self.session_manager = SessionManager(
            self.file_manager,
            self.annotation_manager,
            self.window_manager
        )

        # Load SAM model and initialize predictor
        self._create_predictor(checkpoint_path)
        
        # Set predictor in session manager
        self.session_manager.set_predictor(self.predictor)
        
        # Load classes
        self._load_classes(classes_csv)

        # Setup callbacks
        self._setup_callbacks()
        
        self.logger.info("SAM Annotator initialized successfully")

    def _create_predictor(self, checkpoint_path: str) -> None:
        """Create and initialize appropriate predictor based on version."""
        self.logger.info(f"Creating predictor for {self.sam_version}")
        try:
            # Create appropriate predictor based on version
            if self.sam_version == 'sam1':
                self.predictor = SAM1Predictor()
                self.logger.info("Created SAM1Predictor")
            else:
                self.predictor = SAM2Predictor()
                self.logger.info("Created SAM2Predictor")
                
            # Initialize with checkpoint
            self.predictor.initialize(checkpoint_path)
            self.logger.info("Predictor initialized with checkpoint")
            
        except Exception as e:
            self.logger.error(f"Error creating predictor: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    
            
    def _load_classes(self, classes_csv: str) -> None:
        """Load class definitions.
        
        Args:
            classes_csv: Path to CSV containing class definitions
        """
        try:
            df = pd.read_csv(classes_csv)
            self.class_names = df['class_name'].tolist()[:15]  # Limit to 15 classes
            self.annotation_manager.class_names = self.class_names
            self.logger.info(f"Loaded {len(self.class_names)} classes")
            
        except Exception as e:
            self.logger.error(f"Error loading classes: {str(e)}")
            raise
            
    def _setup_callbacks(self) -> None:
        """Setup event callbacks."""
        try:
            # Event handler callbacks
            self.event_handler.register_callbacks(
                on_mask_prediction=self._handle_mask_prediction,
                on_class_selection=self._handle_class_selection
            )
            
            # Review panel callbacks
            review_callbacks = {
                'delete': self.annotation_manager.delete_annotation,
                'select': self.annotation_manager.select_annotation,
                'class_change': self.annotation_manager.handle_class_change
            }
            
            # Setup window manager callbacks
            self.window_manager.setup_windows(
                mouse_callback=self.event_handler.handle_mouse_event,
                class_callback=self.event_handler.handle_class_window_event,
                review_callbacks=review_callbacks
            )
            
            self.logger.info("Callbacks setup complete")
            
        except Exception as e:
            self.logger.error(f"Error setting up callbacks: {str(e)}")
            raise
            
    def _handle_mask_prediction(self, 
                              box_start: Tuple[int, int],
                              box_end: Tuple[int, int],
                              drawing: bool = False) -> None:
        """Handle mask prediction from box input.
        
        Args:
            box_start: Starting coordinates of box
            box_end: Ending coordinates of box
            drawing: Whether currently drawing box
        """
        if drawing:
            self.window_manager.update_main_window(
                image=self.window_manager.current_image,
                annotations=self.annotation_manager.annotations,
                current_class=self.class_names[self.annotation_manager.current_class_id],
                current_class_id=self.annotation_manager.current_class_id,
                current_image_path=self.session_manager.current_image_path,
                current_idx=self.session_manager.current_idx,
                total_images=self.session_manager.total_images,
                box_start=box_start,
                box_end=box_end
            )
            return
            
        try:
            # Get memory info before prediction
            memory_info = self.predictor.memory_manager.get_gpu_memory_info()
            self.logger.info(f"Memory before prediction: {memory_info.get('formatted', 'Memory stats not available')}")
            
            # Get display and original dimensions
            display_height, display_width = self.window_manager.current_image.shape[:2]
            original_image = cv2.imread(self.session_manager.current_image_path)
            original_height, original_width = original_image.shape[:2]
            
            # Calculate scale factors
            scale_x = original_width / display_width
            scale_y = original_height / display_height
            
            # Scale coordinates to original size
            orig_box_start = (
                int(box_start[0] * scale_x),
                int(box_start[1] * scale_y)
            )
            orig_box_end = (
                int(box_end[0] * scale_x),
                int(box_end[1] * scale_y)
            )
            
            # Prepare input for predictor
            center_x = (orig_box_start[0] + orig_box_end[0]) // 2
            center_y = (orig_box_start[1] + orig_box_end[1]) // 2
            
            input_points = np.array([[center_x, center_y]])
            input_labels = np.array([1])
            
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
            
            if len(masks) > 0:
                # Get best mask
                best_mask_idx = np.argmax(scores) if scores.size > 0 else 0
                best_mask = masks[best_mask_idx]
                
                # Scale mask to display size
                display_mask = cv2.resize(
                    best_mask.astype(np.uint8),
                    (display_width, display_height),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Update display
                self.window_manager.set_mask(display_mask.astype(bool))
                self.window_manager.update_main_window(
                    image=self.window_manager.current_image,
                    annotations=self.annotation_manager.annotations,
                    current_class=self.class_names[self.annotation_manager.current_class_id],
                    current_class_id=self.annotation_manager.current_class_id,
                    current_image_path=self.session_manager.current_image_path,
                    current_idx=self.session_manager.current_idx,
                    total_images=self.session_manager.total_images,
                    status="Mask predicted - press 'a' to add"
                )
                
            # Get memory info after prediction
            memory_info = self.predictor.memory_manager.get_gpu_memory_info()
            self.logger.info(f"Memory after prediction: {memory_info.get('formatted', 'Memory stats not available')}")
            
        except Exception as e:
            self.logger.error(f"Error in mask prediction: {str(e)}")
            self.window_manager.update_main_window(
                status=f"Error predicting mask: {str(e)}"
            )
            
    def _handle_class_selection(self, class_id: int) -> None:
        """Handle class selection.
        
        Args:
            class_id: Selected class ID
        """
        try:
            self.annotation_manager.current_class_id = class_id
            self.window_manager.update_class_window(
                self.class_names,
                class_id
            )
        except Exception as e:
            self.logger.error(f"Error handling class selection: {str(e)}")
            
    def _handle_export(self, format: str) -> None:
        """Handle dataset export.
        
        Args:
            format: Export format ('coco' or 'yolo')
        """
        try:
            # Update status
            self.window_manager.update_main_window(
                status=f"Exporting dataset to {format.upper()} format..."
            )
            
            # Perform export
            export_path = self.file_manager.export_dataset(format)
            
            # Update status
            self.window_manager.update_main_window(
                status=f"Dataset exported to: {export_path}"
            )
            
        except Exception as e:
            self.logger.error(f"Error exporting dataset: {str(e)}")
            self.window_manager.update_main_window(
                status=f"Export failed: {str(e)}"
            )
            
    def _handle_undo(self) -> None:
        """Handle undo command."""
        if self.command_manager.undo():
            self.window_manager.update_main_window(
                status="Undo successful"
            )
            
    def _handle_redo(self) -> None:
        """Handle redo command."""
        if self.command_manager.redo():
            self.window_manager.update_main_window(
                status="Redo successful"
            )
            
    def run(self) -> None:
        """Run the annotation session."""
        try:
            # Initialize session
            self.session_manager.initialize_session()
            
            # Main event loop
            while True:
                # Check memory usage
                if hasattr(self.image_processor, 'get_memory_usage'):
                    memory_usage = self.image_processor.get_memory_usage()
                    if memory_usage > 1e9:  # More than 1GB
                        self.logger.info("Clearing image cache")
                        self.image_processor.clear_cache()
                        
                # Check GPU memory
                if hasattr(self.predictor, 'get_memory_usage'):
                    gpu_memory = self.predictor.get_memory_usage()
                    if gpu_memory > 0.8:  # Over 80% GPU memory
                        self.predictor.optimize_memory()
                        
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == -1:
                    continue
                    
                action = self.event_handler.handle_keyboard_event(key)
                self.logger.debug(f"Keyboard action: {action}")
                
                # Process actions
                if action == "quit":
                    break
                elif action == "save":
                    self.file_manager.save_annotations(
                        self.annotation_manager.annotations,
                        self.session_manager.current_image_path
                    )
                elif action == "next":
                    self.session_manager.next_image()
                elif action == "prev":
                    self.session_manager.prev_image()
                elif action == "undo":
                    self._handle_undo()
                elif action == "redo":
                    self._handle_redo()
                elif action == "export_coco":
                    self._handle_export('coco')
                elif action == "export_yolo":
                    self._handle_export('yolo')
                    
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
            
        finally:
            self.cleanup()
            
    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Save session state
            self.session_manager.save_current_state()
            
            # Save any unsaved annotations
            if self.annotation_manager.annotations:
                self.file_manager.save_annotations(
                    self.annotation_manager.annotations,
                    self.session_manager.current_image_path
                )
                
            # Cleanup windows
            self.window_manager.destroy_windows()
            
            # Clean up GPU memory
            if hasattr(self.predictor, 'cleanup'):
                self.predictor.cleanup()
                
            self.logger.info("Cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")