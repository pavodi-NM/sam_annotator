# SAM Annotator API Reference

This document provides a comprehensive guide to the SAM Annotator API, enabling programmatic access to the annotation functionality.

## Overview

The SAM Annotator API allows you to:
- Load and initialize the SAM model
- Process images programmatically
- Generate masks using box or point prompts
- Manage annotations 
- Export annotations to various formats

## Installation

To use the SAM Annotator API, install the package:

```bash
pip install sam-annotator
```

## Example Scripts

The SAM Annotator package includes example scripts to help you get started quickly:

### Simple Example

`simple_api_example.py` demonstrates the core functionality in a minimal script:

```python
import os
import cv2
import numpy as np
from sam_annotator.core import SAMAnnotator

# Initialize the annotator
annotator = SAMAnnotator(
    checkpoint_path=None,  # Will use default
    category_path="work_dir",
    classes_csv="classes.csv",
    sam_version="sam1"
)

# Load an image
image = cv2.imread("path/to/image.jpg")

# Set the image in the predictor
predictor = annotator.predictor
predictor.set_image(image)

# Generate a mask using a box prompt
box = np.array([100, 100, 300, 300]).reshape(1, 4)  # [x1, y1, x2, y2]
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=box,
    multimask_output=True
)

# Get the best mask
mask = masks[np.argmax(scores)]

# Add the mask as an annotation
annotation = {
    'mask': mask,
    'class_id': 1,
    'box': box[0],
    'area': np.sum(mask),
    'metadata': {'annotation_mode': 'box'}
}
annotation_id = annotator.annotation_manager.add_annotation(annotation)

# Set the current image path (needed for saving)
annotator.session_manager.current_image_path = "path/to/image.jpg"

# Save the annotation
annotator.file_manager.save_annotations(
    annotations=[annotation],
    image_name="path/to/image.jpg",
    original_dimensions=image.shape[:2],
    display_dimensions=image.shape[:2],
    class_names=["class1"],
    save_visualization=True
)
```

### Comprehensive Example

`api_example.py` is a full-featured example that demonstrates:
- Command-line argument handling
- Multiple annotation methods (box and point prompts)
- Working with multiple classes
- Exporting to all supported formats
- Creating visualizations

These example scripts can be found in the `examples/` directory of the SAM Annotator repository.

## Core Components

The API is organized into several core components:

### 1. SAMAnnotator

The main class that coordinates the annotation functionality.

```python
from sam_annotator.core import SAMAnnotator

# Initialize the annotator
annotator = SAMAnnotator(
    checkpoint_path="path/to/checkpoint.pth",
    category_path="path/to/category",
    classes_csv="path/to/classes.csv",
    sam_version="sam1",  # or "sam2"
    model_type="vit_h"   # depends on the SAM version
)

# Access to components
predictor = annotator.predictor  # The SAM model predictor
annotations = annotator.annotations  # List of annotations
file_manager = annotator.file_manager  # Handles file operations
session_manager = annotator.session_manager  # Manages the current session
command_manager = annotator.command_manager  # Manages undo/redo and commands
```

### 2. Predictor

The interface to the SAM model for generating masks.

```python
# Get the predictor from the annotator
predictor = annotator.predictor

# Load an image
image = cv2.imread("path/to/image.jpg")

# Important: Always set the image in the predictor before prediction
predictor.set_image(image)

# Predict with a box
box = np.array([100, 100, 300, 300]).reshape(1, 4)  # [x1, y1, x2, y2]
masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=box,
    multimask_output=True
)

# Predict with points
point_coords = np.array([[100, 100], [200, 200]])
point_labels = np.array([1, 1])  # 1 for foreground, 0 for background
masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    box=None,
    multimask_output=True
)
```

### 3. Command Manager

Manages operations on annotations, including adding, deleting, and modifying annotations.

```python
# Import the necessary command
from sam_annotator.core.command_manager import AddAnnotationCommand

# Create an annotation structure
mask = masks[np.argmax(scores)]  # Get the best mask

# Need to create contours from the mask
mask_uint8 = mask.astype(np.uint8) * 255
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
contour = max(contours, key=cv2.contourArea)  # Get largest contour

# Create flattened contour list
contour_list = contour.tolist()
if len(contour_list) > 0 and isinstance(contour_list[0], list) and len(contour_list[0]) == 1:
    contour_list = [point[0] for point in contour_list]

# Create annotation dictionary
annotation = {
    'id': len(annotator.annotations),  # Next available ID
    'class_id': 1,
    'class_name': annotator.class_names[1],
    'box': [100, 100, 300, 300],
    'contour': contour_list,  # Flattened points
    'contour_points': contour,  # Original OpenCV contour
    'mask': mask,  # Boolean numpy array
    'display_box': [100, 100, 300, 300],
    'area': np.sum(mask),
    'metadata': {'annotation_mode': 'box'}
}

# Add annotation using the command manager
command = AddAnnotationCommand(annotator.annotations, annotation, annotator.window_manager)
annotator.command_manager.execute(command)

# Note: There's no direct "annotation_manager" - annotations are stored directly in annotator.annotations
```

### 4. SessionManager

Manages the current annotation session, including storing current image path and navigating between images.

```python
# Get the session manager from the annotator
session_manager = annotator.session_manager

# Set the current image path (required for saving)
session_manager.current_image_path = "path/to/image.jpg"

# Save annotations for the current image
session_manager.save_annotations()

# Navigate to next/previous image
next_path = session_manager.next_image()
prev_path = session_manager.previous_image()

# Check if navigation is possible
can_go_next = session_manager.can_move_next()
can_go_prev = session_manager.can_move_prev()

# Get the current image path
current_path = session_manager.get_current_image_path()
```

### 5. FileManager

Handles file operations like loading/saving annotations and exporting to different formats.

```python
# Get the file manager from the annotator
file_manager = annotator.file_manager

# Export annotations to different formats
coco_path = file_manager.handle_export("coco", class_names)
yolo_path = file_manager.handle_export("yolo", class_names)
pascal_path = file_manager.handle_export("pascal", class_names)
```

## Basic Usage

### Initializing the Annotator

```python
from sam_annotator.core import SAMAnnotator

# Initialize the annotator
annotator = SAMAnnotator(
    checkpoint_path="path/to/checkpoint.pth",
    category_path="path/to/category",
    classes_csv="path/to/classes.csv",
    sam_version="sam1",
    model_type="vit_h"
)
```

### Loading an Image

```python
# Set the current image path in session manager
annotator.session_manager.current_image_path = "path/to/image.jpg"

# Or load directly for prediction
image = cv2.imread("path/to/image.jpg")
annotator.predictor.set_image(image)
```

### Generating a Mask with a Box Prompt

```python
# Define a bounding box [x1, y1, x2, y2]
box = np.array([100, 100, 300, 300]).reshape(1, 4)

# Set the image in the predictor
predictor = annotator.predictor
predictor.set_image(image)

# Generate masks
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=box,
    multimask_output=True
)

# Get the best mask
mask_idx = np.argmax(scores)
mask = masks[mask_idx]
```

### Generating a Mask with Point Prompts

```python
# Define foreground points [[x1, y1], [x2, y2], ...]
foreground_points = np.array([[150, 150], [200, 200]])
foreground_labels = np.array([1, 1])  # 1 for foreground

# Define background points (optional)
background_points = np.array([[50, 50], [350, 350]])
background_labels = np.array([0, 0])  # 0 for background

# Combine points and labels
point_coords = np.vstack((foreground_points, background_points))
point_labels = np.hstack((foreground_labels, background_labels))

# Generate masks
masks, scores, _ = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    box=None,
    multimask_output=True
)

# Get the best mask
mask_idx = np.argmax(scores)
mask = masks[mask_idx]
```

### Adding an Annotation

```python
from sam_annotator.core.command_manager import AddAnnotationCommand

# Create contours from the mask
mask_uint8 = mask.astype(np.uint8) * 255
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
if contours:
    contour = max(contours, key=cv2.contourArea)
    
    # Create flattened contour list
    contour_list = contour.tolist()
    if len(contour_list) > 0 and isinstance(contour_list[0], list) and len(contour_list[0]) == 1:
        contour_list = [point[0] for point in contour_list]
    
    # Create an annotation structure
    annotation = {
        'id': len(annotator.annotations),
        'class_id': 1,
        'class_name': annotator.class_names[1],
        'box': [100, 100, 300, 300],
        'contour': contour_list,
        'contour_points': contour,
        'mask': mask,
        'display_box': [100, 100, 300, 300],
        'area': np.sum(mask),
        'metadata': {'annotation_mode': 'box'}
    }
    
    # Add the annotation using the command manager
    command = AddAnnotationCommand(annotator.annotations, annotation, annotator.window_manager)
    annotator.command_manager.execute(command)
```

### Saving Annotations

```python
# Make sure the session manager knows about the current image
annotator.session_manager.current_image_path = "path/to/image.jpg"

# Save annotations
success = annotator.file_manager.save_annotations(
    annotations=[annotation],
    image_name="path/to/image.jpg",
    original_dimensions=image.shape[:2],
    display_dimensions=image.shape[:2],
    class_names=["class1"],
    save_visualization=True
)
```

### Exporting Annotations

```python
# Export all annotations to COCO format
coco_path = annotator.file_manager.handle_export("coco", annotator.class_names)

# Export to YOLO format
yolo_path = annotator.file_manager.handle_export("yolo", annotator.class_names)

# Export to Pascal VOC format
pascal_path = annotator.file_manager.handle_export("pascal", annotator.class_names)
```

## Advanced Usage

### Batch Processing

Process multiple images in a folder:

```python
import os
import cv2
import numpy as np
from sam_annotator.core import SAMAnnotator

# Initialize
annotator = SAMAnnotator(
    checkpoint_path="path/to/checkpoint.pth",
    category_path="path/to/category",
    classes_csv="path/to/classes.csv"
)

# Get all images in the images folder
image_folder = os.path.join(annotator.category_path, "images")
image_files = [f for f in os.listdir(image_folder) 
               if f.endswith(('.jpg', '.jpeg', '.png'))]

# Get the predictor
predictor = annotator.predictor

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    # Load image
    image = cv2.imread(image_path)
    predictor.set_image(image)
    
    # Set the current image path in the session manager
    annotator.session_manager.current_image_path = image_path
    
    # Example: generate a mask for center of the image
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    box_size = min(width, height) // 3
    
    box = np.array([
        center_x - box_size // 2, 
        center_y - box_size // 2,
        center_x + box_size // 2, 
        center_y + box_size // 2
    ]).reshape(1, 4)
    
    # Generate mask
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=True
    )
    
    # Get the best mask
    if scores.size > 0:
        mask_idx = np.argmax(scores)
        mask = masks[mask_idx]
        
        # Add annotation
        annotation = {
            'mask': mask,
            'class_id': 1,
            'box': box[0].tolist(),
            'area': np.sum(mask),
            'metadata': {'annotation_mode': 'box'}
        }
        
        annotator.annotation_manager.add_annotation(annotation)
    
    # Save annotations
    annotator.file_manager.save_annotations(
        annotations=[annotation],
        image_name=image_file,
        original_dimensions=(height, width),
        display_dimensions=(height, width),
        class_names=["class1"],
        save_visualization=True
    )

# Export all annotations
annotator.file_manager.handle_export("coco", annotator.class_names)
```

## API Reference

### SAMAnnotator Class

```python
class SAMAnnotator:
    """Main class coordinating SAM-based image annotation."""
    def __init__(self, 
                checkpoint_path: str,
                category_path: str,
                classes_csv: str,
                sam_version: str = 'sam1',
                model_type: str = None):
        """Initialize SAM annotator with all components."""
        
    # Internal method, not meant for direct API use:
    def _load_image(self, image_path: str) -> None:
        """Internal method to load image and its existing annotations."""
```

### Predictor Classes

```python
class BaseSAMPredictor:
    """Base class for SAM predictors."""
    def initialize(self, checkpoint_path: str) -> None:
        """Initialize the predictor with a model checkpoint."""
        
    def predict(self, 
               point_coords: np.ndarray = None,
               point_labels: np.ndarray = None,
               box: np.ndarray = None,
               multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate masks from the provided prompts."""
```

### AnnotationManager Class

```python
class AnnotationManager:
    """Manages annotations and their operations."""
    def add_annotation(self, annotation: Dict) -> int:
        """Add a new annotation."""
        
    def delete_annotation(self, annotation_id: int) -> bool:
        """Delete an annotation by ID."""
        
    def select_annotation(self, annotation_id: int) -> Dict:
        """Select an annotation by ID."""
        
    def modify_annotation(self, annotation_id: int, properties: Dict) -> bool:
        """Modify properties of an annotation."""
```

### FileManager Class

```python
class FileManager:
    """Manages file operations for annotations."""
    def load_annotations(self, image_path: str) -> List[Dict]:
        """Load annotations for the specified image."""
        
    def save_annotations(self, 
                        annotations: List[Dict],
                        image_name: str,
                        original_dimensions: Tuple[int, int],
                        display_dimensions: Tuple[int, int],
                        class_names: List[str]) -> bool:
        """Save annotations for the specified image."""
        
    def handle_export(self, format: str, class_names: List[str]) -> str:
        """Export annotations to the specified format."""
```

## Building Custom Extensions

### Creating a Custom Predictor

You can extend the predictor functionality by creating a custom predictor:

```python
from sam_annotator.core import BaseSAMPredictor
import numpy as np

class CustomPredictor(BaseSAMPredictor):
    def initialize(self, checkpoint_path: str) -> None:
        """Initialize with custom logic."""
        # Your custom initialization code
        
    def predict(self, 
               point_coords: np.ndarray = None,
               point_labels: np.ndarray = None,
               box: np.ndarray = None,
               multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Custom prediction implementation."""
        # Your custom prediction logic
        
        # Return format: (masks, scores, logits)
        return masks, scores, logits
```

### Creating a Custom Exporter

You can create a custom exporter for a new annotation format:

```python
from sam_annotator.data.exporters import BaseExporter

class CustomExporter(BaseExporter):
    def __init__(self, base_path: str):
        super().__init__(base_path)
        
    def export(self) -> str:
        """Export annotations to a custom format."""
        # Get annotation data
        annotations = self.load_all_annotations()
        
        # Process annotations into custom format
        # ...
        
        # Save to file
        export_path = self._get_export_path("custom")
        with open(export_path, 'w') as f:
            # Write your custom format
            pass
            
        return export_path
```

## Error Handling

The API includes robust error handling for various common issues:

```python
try:
    # Initialize the annotator
    annotator = SAMAnnotator(
        checkpoint_path="path/to/checkpoint.pth",
        category_path="path/to/category",
        classes_csv="path/to/classes.csv"
    )
    
    # Try to load an image that might not exist
    try:
        image = annotator.load_image("non_existent_image.jpg")
    except FileNotFoundError as e:
        print(f"Error loading image: {e}")
        
    # Try to generate a mask with invalid inputs
    try:
        mask = annotator.predict_mask_from_box([-100, -100, 100, 100])
    except ValueError as e:
        print(f"Invalid box coordinates: {e}")
        
except Exception as e:
    print(f"General error: {e}")
```

## Performance Considerations

When using the API programmatically, consider the following for optimal performance:

1. **Batch Processing**: Process images in batches rather than one by one to amortize model loading time
2. **Memory Management**: Clear unused objects to free memory, especially after processing large images
3. **GPU Utilization**: SAM benefits significantly from GPU acceleration; ensure CUDA is properly configured
4. **Image Sizing**: Consider resizing large images before processing to improve performance
5. **Error Handling**: Implement robust error handling to avoid interruptions during batch processing

## Coming Soon

The following API features are planned for future releases:

1. **Automatic Annotation**: Functionality for automatic annotation suggestions
2. **Annotation Refinement**: Methods to refine existing annotations
3. **Multi-Model Support**: Integration with additional segmentation models
4. **Async Processing**: Asynchronous processing for improved performance
5. **Web API**: REST API for remote access to annotation functionality 