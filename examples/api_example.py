#!/usr/bin/env python3
"""
SAM Annotator API Example

This example demonstrates how to use the SAM Annotator API to:
1. Load a single image
2. Generate annotations using box and point prompts
3. Save annotations to disk
4. Export annotations to different formats

--  must be reviewed and updated

Usage:
    python api_example.py --image <path_to_image> [--checkpoint <path_to_checkpoint>] [--output_dir <output_directory>]
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Import SAM Annotator components
from sam_annotator.core import SAMAnnotator
from sam_annotator.utils.visualization import VisualizationManager
from sam_annotator.core.command_manager import AddAnnotationCommand


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SAM Annotator API Example")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", default=None, help="Path to SAM checkpoint (if not specified, will use default)")
    parser.add_argument("--output_dir", default="./output", help="Directory to save output")
    parser.add_argument("--sam_version", default="sam1", choices=["sam1", "sam2"], help="SAM version to use")
    parser.add_argument("--model_type", default=None, help="Model type for the selected SAM version")
    return parser.parse_args()


def setup_environment(args):
    """Setup the environment for the example."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create category structure
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    masks_dir = output_dir / "masks"
    exports_dir = output_dir / "exports"
    
    for dir_path in [images_dir, labels_dir, masks_dir, exports_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Copy the input image to the images directory
    image_name = os.path.basename(args.image)
    image_path = images_dir / image_name
    if not image_path.exists():
        import shutil
        shutil.copy2(args.image, image_path)
    
    # Create a simple class CSV file if it doesn't exist
    classes_file = output_dir / "classes.csv"
    if not classes_file.exists():
        with open(classes_file, "w") as f:
            f.write("class_id,class_name,color\n")
            f.write("0,background,\"0,0,0\"\n")
            f.write("1,object,\"255,0,0\"\n")
            f.write("2,person,\"0,255,0\"\n")
            f.write("3,vehicle,\"0,0,255\"\n")
    
    return output_dir, image_path, classes_file


def visualize_results(image, masks, output_path):
    """Visualize the results for better understanding."""
    # Create a visualization manager
    vis_manager = VisualizationManager()
    
    # Convert image to RGB if it's in BGR (OpenCV default)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(masks) + 1, figsize=(15, 5))
    
    # Show original image
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Show each mask
    for i, mask in enumerate(masks):
        # Create a colored overlay
        overlay = image_rgb.copy()
        mask_overlay = np.zeros_like(image_rgb)
        
        # Color depends on the mask index (for visual distinction)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        color = colors[i % len(colors)]
        
        # Apply the mask
        mask_overlay[mask] = color
        
        # Blend with original image
        alpha = 0.5
        result = cv2.addWeighted(overlay, 1 - alpha, mask_overlay, alpha, 0)
        
        # Display
        axes[i + 1].imshow(result)
        axes[i + 1].set_title(f"Mask {i + 1}")
        axes[i + 1].axis("off")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Visualization saved to {output_path}")


def generate_mask_from_box(predictor, image, box):
    """Helper function to generate a mask from a box."""
    # Set the image in the predictor
    predictor.set_image(image)
    
    # Convert box to numpy array
    input_box = np.array(box).reshape(1, 4)
    
    # Generate masks
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=True
    )
    
    # Get the best mask based on score
    if scores.size > 0:
        mask_idx = np.argmax(scores)
        return masks[mask_idx], scores[mask_idx]
    else:
        return None, 0


def generate_mask_from_points(predictor, image, foreground_points, background_points=None):
    """Helper function to generate a mask from points."""
    # Set the image in the predictor
    predictor.set_image(image)
    
    # Prepare points and labels
    fg_points = np.array(foreground_points)
    
    if background_points:
        bg_points = np.array(background_points)
        point_coords = np.vstack((fg_points, bg_points))
        point_labels = np.array([1] * len(fg_points) + [0] * len(bg_points))
    else:
        point_coords = fg_points
        point_labels = np.array([1] * len(fg_points))
    
    # Reshape to match expected format
    point_coords = point_coords.reshape(-1, 2)
    point_labels = point_labels.reshape(-1)
    
    # Generate masks
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=True
    )
    
    # Get the best mask based on score
    if scores.size > 0:
        mask_idx = np.argmax(scores)
        return masks[mask_idx], scores[mask_idx]
    else:
        return None, 0


def create_annotation_from_mask(mask, class_id, box, class_name, mode="box"):
    """Helper function to create an annotation from a mask."""
    # Create contour from mask
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    
    if not contours:
        return None
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Create flattened contour list
    contour_list = contour.tolist()
    if len(contour_list) > 0 and isinstance(contour_list[0], list) and len(contour_list[0]) == 1:
        contour_list = [point[0] for point in contour_list]
    
    # Create annotation dictionary
    annotation = {
        'id': 0,  # Will be updated when added
        'class_id': class_id,
        'class_name': class_name,
        'box': box,
        'contour': contour_list,
        'contour_points': contour,
        'mask': mask,
        'display_box': box,
        'area': np.sum(mask),
        'metadata': {'annotation_mode': mode}
    }
    
    return annotation


def add_annotation(annotator, annotation):
    """Helper function to add an annotation to SAMAnnotator."""
    if annotation:
        # Update ID
        annotation['id'] = len(annotator.annotations)
        
        # Add using command manager
        command = AddAnnotationCommand(annotator.annotations, annotation, annotator.window_manager)
        annotator.command_manager.execute(command)
        return annotation['id']
    return None


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    output_dir, image_path, classes_file = setup_environment(args)
    print(f"Working directory: {output_dir}")
    print(f"Image path: {image_path}")
    
    # Initialize SAM Annotator
    annotator = SAMAnnotator(
        checkpoint_path=args.checkpoint,
        category_path=str(output_dir),
        classes_csv=str(classes_file),
        sam_version=args.sam_version,
        model_type=args.model_type
    )
    print("SAM Annotator initialized successfully")
    
    # Load the image
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Make sure the session manager knows about the current image
    annotator.session_manager.current_image_path = str(image_path)
    print("Set current image path in session manager")
    
    # Get the predictor
    predictor = annotator.predictor
    
    # Store masks for later visualization
    all_masks = []
    
    # Example 1: Generate a mask using a box prompt
    print("\nGenerating mask with box prompt...")
    # Create a box covering the center part of the image
    center_x, center_y = width // 2, height // 2
    box_size = min(width, height) // 3
    box = [
        center_x - box_size // 2, 
        center_y - box_size // 2,
        center_x + box_size // 2, 
        center_y + box_size // 2
    ]
    print(f"Box coordinates: {box}")
    
    # Generate mask
    mask1, score1 = generate_mask_from_box(predictor, image, box)
    print(f"Generated mask with score: {score1}")
    
    # Add as annotation with class "object"
    if mask1 is not None and mask1.any():
        # Create annotation structure
        annotation1 = create_annotation_from_mask(
            mask=mask1,
            class_id=1,
            box=box,
            class_name=annotator.class_names[1],
            mode="box"
        )
        
        # Add annotation
        annotation_id = add_annotation(annotator, annotation1)
        if annotation_id is not None:
            print(f"Added annotation with ID: {annotation_id}")
            all_masks.append(mask1)
        else:
            print("Failed to add annotation")
    else:
        print("No valid mask generated from box prompt")
    
    # Example 2: Generate a mask using point prompts
    print("\nGenerating mask with point prompts...")
    # Create some foreground points (adjust based on what's in your image)
    foreground_points = [
        [width // 4, height // 4],
        [width // 4 + 20, height // 4 + 20]
    ]
    background_points = [
        [width // 4 - 50, height // 4 - 50],
        [width // 4 + 70, height // 4 + 70]
    ]
    print(f"Foreground points: {foreground_points}")
    print(f"Background points: {background_points}")
    
    # Generate mask
    mask2, score2 = generate_mask_from_points(predictor, image, foreground_points, background_points)
    print(f"Generated mask with score: {score2}")
    
    # Add as annotation with class "person"
    if mask2 is not None and mask2.any():
        # Create a box from foreground points
        box2 = [
            min(p[0] for p in foreground_points), 
            min(p[1] for p in foreground_points),
            max(p[0] for p in foreground_points), 
            max(p[1] for p in foreground_points)
        ]
        
        # Create annotation structure
        annotation2 = create_annotation_from_mask(
            mask=mask2,
            class_id=2,
            box=box2,
            class_name=annotator.class_names[2],
            mode="point"
        )
        
        # Add annotation
        annotation_id = add_annotation(annotator, annotation2)
        if annotation_id is not None:
            print(f"Added annotation with ID: {annotation_id}")
            all_masks.append(mask2)
        else:
            print("Failed to add annotation")
    else:
        print("No valid mask generated from point prompts")
    
    # Example 3: Generate another mask for a different object
    print("\nGenerating mask for another object...")
    # Create a box for another object (adjust based on your image)
    box3 = [
        width * 3 // 4 - box_size // 2,
        height * 3 // 4 - box_size // 2,
        width * 3 // 4 + box_size // 2,
        height * 3 // 4 + box_size // 2
    ]
    print(f"Box coordinates: {box3}")
    
    # Generate mask
    mask3, score3 = generate_mask_from_box(predictor, image, box3)
    print(f"Generated mask with score: {score3}")
    
    # Add as annotation with class "vehicle"
    if mask3 is not None and mask3.any():
        # Create annotation structure
        annotation3 = create_annotation_from_mask(
            mask=mask3,
            class_id=3,
            box=box3,
            class_name=annotator.class_names[3],
            mode="box"
        )
        
        # Add annotation
        annotation_id = add_annotation(annotator, annotation3)
        if annotation_id is not None:
            print(f"Added annotation with ID: {annotation_id}")
            all_masks.append(mask3)
        else:
            print("Failed to add annotation")
    else:
        print("No valid mask generated from second box prompt")
    
    # Save annotations
    print("\nSaving annotations...")
    # Get image dimensions
    original_dimensions = (height, width)
    display_dimensions = (height, width)  # Same as original in API case
    
    # Use the file_manager to save annotations
    success = annotator.file_manager.save_annotations(
        annotations=annotator.annotations,
        image_name=os.path.basename(image_path),
        original_dimensions=original_dimensions,
        display_dimensions=display_dimensions,
        class_names=annotator.class_names,
        save_visualization=True
    )
    
    if success:
        print(f"Annotations saved to {output_dir}/labels/{os.path.splitext(os.path.basename(image_path))[0]}.txt")
    else:
        print("Failed to save annotations")
    
    # Export annotations to various formats
    print("\nExporting annotations to different formats...")
    
    try:
        coco_path = annotator.file_manager.handle_export("coco", annotator.class_names)
        print(f"Exported to COCO format: {coco_path}")
    except Exception as e:
        print(f"Failed to export to COCO format: {e}")
    
    try:
        yolo_path = annotator.file_manager.handle_export("yolo", annotator.class_names)
        print(f"Exported to YOLO format: {yolo_path}")
    except Exception as e:
        print(f"Failed to export to YOLO format: {e}")
    
    try:
        pascal_path = annotator.file_manager.handle_export("pascal", annotator.class_names)
        print(f"Exported to Pascal VOC format: {pascal_path}")
    except Exception as e:
        print(f"Failed to export to Pascal VOC format: {e}")
    
    # Visualize results
    if all_masks:
        vis_path = output_dir / "visualization.png"
        visualize_results(image, all_masks, vis_path)
    else:
        print("No masks to visualize")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 