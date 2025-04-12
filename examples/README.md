# SAM Annotator API Examples

This directory contains example scripts demonstrating how to use the SAM Annotator API programmatically.

> **Important Note**: The API examples have been updated to match the actual implementation of SAM Annotator. The API works by directly using the components of the SAM Annotator (predictor, annotation_manager, session_manager, file_manager) rather than through convenience methods. This provides more flexibility and control over the annotation process.

## Available Examples

1. `simple_api_example.py`: A minimal example showing the core functionality
2. `api_example.py`: A comprehensive example showing various features

## Requirements

Make sure you have SAM Annotator installed:

```bash
pip install sam-annotator
```

You'll also need additional dependencies for the examples:
```bash
pip install numpy matplotlib opencv-python
```

## Running the Simple Example

The simple example demonstrates how to:
- Load an image
- Generate a mask using a box prompt
- Save an annotation
- Export to COCO format
- Create a visualization

To run it:

1. Edit the script to point to your test image:
   ```python
   # Change this to your image path
   IMAGE_PATH = "path/to/your/image.jpg"
   ```

2. Run the script:
   ```bash
   python simple_api_example.py
   ```

The script will create a directory called `sam_annotator_test` with all outputs.

## Running the Comprehensive Example

The comprehensive example accepts command-line arguments and demonstrates:
- Loading an image
- Generating masks using both box and point prompts
- Adding multiple annotations with different classes
- Saving annotations
- Exporting to multiple formats (COCO, YOLO, Pascal VOC)
- Creating visualizations with matplotlib

To run it:

```bash
python api_example.py --image path/to/your/image.jpg [--checkpoint path/to/sam/checkpoint.pth] [--output_dir ./output] [--sam_version sam1|sam2]
```

## API Structure

The SAM Annotator API is organized into several components:

- **SAMAnnotator**: The main class that coordinates the entire process
- **predictor**: Interfaces with the SAM model to generate masks
- **annotation_manager**: Manages the annotations (add, delete, modify)
- **session_manager**: Handles the current annotation session
- **file_manager**: Manages file operations (loading, saving, exporting)

Example of using these components:

```python
# Initialize the annotator
annotator = SAMAnnotator(
    checkpoint_path=None,  # Will use default
    category_path="work_dir",
    classes_csv="classes.csv",
    sam_version="sam1"
)

# Get the predictor
predictor = annotator.predictor

# Set the image
image = cv2.imread("path/to/image.jpg")
predictor.set_image(image)

# Generate a mask with a box
box = np.array([100, 100, 300, 300]).reshape(1, 4)
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=box,
    multimask_output=True
)

# Get the best mask
mask = masks[np.argmax(scores)]

# Add as an annotation
annotation = {
    'mask': mask,
    'class_id': 1,
    'box': box[0].tolist(),
    'area': np.sum(mask),
    'metadata': {'annotation_mode': 'box'}
}
annotation_id = annotator.annotation_manager.add_annotation(annotation)

# Save the annotation
annotator.session_manager.current_image_path = "path/to/image.jpg"
annotator.file_manager.save_annotations(
    annotations=[annotation],
    image_name="path/to/image.jpg",
    original_dimensions=image.shape[:2],
    display_dimensions=image.shape[:2],
    class_names=annotator.class_names,
    save_visualization=True
)
```

## Example Output

After running the examples, you should see:

1. Annotation files in the `labels` directory
2. Visualization images showing the original image with overlay masks
3. Export files in the `exports` directory for different formats

## Modifying the Examples

Feel free to modify these examples to fit your specific use case. Here are some ideas:

- Process multiple images in a batch
- Implement custom annotation logic
- Integrate with other computer vision libraries
- Set up an automated annotation pipeline

## Further Documentation

For complete API documentation, please refer to:
[SAM Annotator API Reference](https://pavodi-nm.github.io/sam_annotator/api_reference/) 