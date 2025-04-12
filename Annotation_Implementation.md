# SAM Annotator Implementation Details

This document provides an in-depth explanation of how the point-based and box-based annotation features are implemented in the SAM Annotator tool.

## Table of Contents

1. [Overview](#overview)
2. [Components](#components)
3. [Box-Based Annotation](#box-based-annotation)
4. [Point-Based Annotation](#point-based-annotation)
5. [Annotation Data Structure](#annotation-data-structure)
6. [Mask Processing](#mask-processing)
7. [Saving and Loading Annotations](#saving-and-loading-annotations)
8. [Visualization](#visualization)

## Overview

The SAM Annotator is built around the Segment Anything Model (SAM), developed by Meta AI. SAM is designed to generate segmentation masks from various prompts including points and bounding boxes. Our application provides an interface to interact with SAM for efficient image annotation.

Two primary annotation methods are implemented:
1. **Box-based annotation**: Drawing a bounding box around an object to generate a segmentation mask
2. **Point-based annotation**: Placing foreground and background points to guide the segmentation

## Components

The annotation system is composed of several interacting components:

- **SAMAnnotator**: Main orchestrator class that coordinates the entire annotation workflow
- **EventHandler**: Manages user interactions with the interface
- **WindowManager**: Handles window operations and visualization
- **Predictor**: Interfaces with the SAM model to generate masks
- **FileManager**: Manages loading and saving of annotations
- **CommandManager**: Implements undo/redo functionality for annotation operations

## Box-Based Annotation

### Implementation Flow

1. **User Interaction**:
   - The user draws a box by clicking and dragging
   - `EventHandler.handle_mouse_event()` captures the mouse events
   - Box coordinates are stored in `box_start` and `box_end` variables

2. **Mask Prediction**:
   - Upon mouse release, `_handle_mask_prediction()` is called
   - The method scales the box coordinates from display size to original image size
   - A center point is calculated from the box for additional reference
   - The predictor is called with both the box and center point:
     ```python
     masks, scores, _ = self.predictor.predict(
         point_coords=input_points,
         point_labels=input_labels,
         box=input_box,
         multimask_output=True
     )
     ```

3. **Processing Results**:
   - The best mask is selected based on confidence scores
   - The mask is resized to match display dimensions
   - The mask is set in WindowManager: `self.window_manager.set_mask(display_mask)`
   - The interface is updated to show the predicted mask

### Key Functions:

- `_handle_mask_prediction()`: Processes the box input and generates a mask
- `EventHandler.handle_mouse_event()`: Captures mouse interactions for drawing the box
- `EventHandler.reset_state()`: Clears the current selection state

## Point-Based Annotation

### Implementation Flow

1. **User Interaction**:
   - The mode is switched to 'point' (using 'p' key)
   - The user clicks to place foreground points (left click) or background points (right click)
   - `EventHandler.handle_mouse_event()` captures these points and their labels
   - Points are stored in the `points` list and labels in the `point_labels` list

2. **Mask Prediction**:
   - After points are placed, pressing 'space' triggers `_handle_point_prediction()`
   - The method scales the point coordinates from display size to original image size
   - The predictor is called with the points and their labels:
     ```python
     masks, scores, _ = self.predictor.predict(
         point_coords=input_points,
         point_labels=input_labels,
         multimask_output=True
     )
     ```

3. **Processing Results**:
   - The best mask is selected based on confidence scores
   - The mask is resized to match display dimensions
   - The mask is set in WindowManager: `self.window_manager.set_mask(display_mask)`
   - The interface is updated to show the predicted mask with the input points

### Key Functions:

- `_handle_point_prediction()`: Processes the point inputs and generates a mask
- `EventHandler.handle_mouse_event()`: Captures mouse interactions for placing points
- `VisualizationManager.draw_input_points()`: Draws the points with appropriate colors (green for foreground, red for background)

## Annotation Data Structure

When an annotation is added using 'a' key, it is converted to a structured format:

```python
annotation = {
    'id': len(self.annotations),
    'class_id': self.current_class_id,
    'class_name': self.class_names[self.current_class_id],
    'box': original_box,            # Box in original image coordinates
    'display_box': display_box,     # Box in display coordinates
    'contour_points': contour_points,  # OpenCV contour format
    'contour': contour_list,        # Flattened points for visualization
    'mask': clean_mask,             # Boolean mask
    'area': cv2.contourArea(display_contour),
    'metadata': {
        'annotation_mode': self.event_handler.mode,
        'timestamp': time.time()
    }
}
```

## Mask Processing

After a mask is predicted, `_add_annotation()` handles the following steps:

1. **Contour Extraction**:
   - The boolean mask is converted to uint8
   - Contours are extracted using `cv2.findContours()`
   - The largest contour is selected

2. **Bounding Box Calculation**:
   - A bounding box is calculated from the contour using `cv2.boundingRect()`
   - The box is scaled for both display and original image dimensions

3. **Mask Cleaning**:
   - A clean boolean mask is created
   - The contour is processed into two formats:
     - `contour_points`: Original cv2 contour format
     - `contour`: Flattened list for visualization

## Saving and Loading Annotations

### Saving Process

The `_save_annotations()` method handles saving annotations to disk:

1. Annotations are validated to ensure they have required fields
2. Original image dimensions are obtained
3. The FileManager's `save_annotations()` method is called with:
   - The annotations list
   - Image name
   - Original and display dimensions
   - Class names

The FileManager then:
1. Scales contour points back to original image space
2. Writes normalized coordinates to a text file
3. Creates visualization images of the masks
4. Saves metadata about the annotations

### Loading Process

When loading an image with existing annotations via `_load_image()`:

1. The image is loaded and processed to display dimensions
2. The FileManager's `load_annotations()` method is called to fetch existing annotations
3. Annotations are scaled to match the display dimensions
4. The interface is updated to show the annotations

## Visualization

The `VisualizationManager` handles all rendering of annotations:

1. **create_composite_view()**: Main method that creates a visualization with:
   - Original image as background
   - Colored mask overlays with adjustable opacity
   - Bounding boxes
   - Class labels
   - Interactive points (when in point mode)

2. **Drawing Functions**:
   - `_draw_mask()`: Renders a mask with the class color and proper opacity
   - `_draw_box()`: Draws a bounding box with the class color
   - `_draw_label()`: Adds a class label with a semi-transparent background
   - `draw_input_points()`: Visualizes input points with numbers and colors indicating foreground/background

## Command Pattern Implementation

Annotation operations use a command pattern for undo/redo functionality:

1. **Add Annotation**: `AddAnnotationCommand` adds a new annotation to the list
2. **Delete Annotation**: `DeleteAnnotationCommand` removes an annotation
3. **Modify Annotation**: `ModifyAnnotationCommand` changes properties of an annotation

Each command handles both the execution and its reverse operation, allowing for robust undo/redo capabilities. 