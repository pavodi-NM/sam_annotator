#!/usr/bin/env python3
"""
Simple SAM Annotator API Example

A minimal example that demonstrates how to use the SAM Annotator API.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sam_annotator.core.annotator import SAMAnnotator
from sam_annotator.core.command_manager import AddAnnotationCommand

def main():
    """Simple example script showing how to use the SAM Annotator API."""
    # Setup working directory
    working_dir = Path("./output")
    working_dir.mkdir(exist_ok=True)
    
    # Create category subdirectories
    for subdir in ["images", "labels", "masks", "metadata"]:
        (working_dir / subdir).mkdir(exist_ok=True)
    
    # Create a simple classes.csv file
    classes_file = working_dir / "classes.csv"
    if not classes_file.exists():
        with open(classes_file, "w") as f:
            f.write("class_id,class_name,color\n")
            f.write("0,background,0,0,0\n")
            f.write("1,object,255,0,0\n")
            f.write("2,person,0,255,0\n")
            f.write("3,vehicle,0,0,255\n")
    
    # Copy an example image to the images directory
    image_path = working_dir / "images" / "example.jpg"
    if not image_path.exists():
        # Use any image you have available
        # For this example, we assume you have an image at this path:
        sample_image = Path("./sample_image.jpg")
        if sample_image.exists():
            import shutil
            shutil.copy(sample_image, image_path)
        else:
            # Create a simple test image
            test_image = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.rectangle(test_image, (100, 100), (400, 400), (0, 255, 0), -1)
            cv2.circle(test_image, (256, 256), 100, (0, 0, 255), -1)
            cv2.imwrite(str(image_path), test_image)
    
    # Initialize SAM Annotator
    annotator = SAMAnnotator(
        checkpoint_path="sam_vit_h_4b8939.pth",  # path to your SAM model checkpoint
        category_path=str(working_dir),
        classes_csv=str(classes_file),
        sam_version="sam1"
    )
    
    # Load the image
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    print(f"Loaded image with dimensions: {width}x{height}")
    
    # Make sure the session manager knows about the current image
    annotator.session_manager.current_image_path = str(image_path)
    
    # Specify a box in the image
    box = [100, 100, 400, 400]  # [x1, y1, x2, y2]
    print(f"Using box coordinates: {box}")
    
    # Get the predictor from the annotator
    predictor = annotator.predictor
    
    # Set the image in the predictor
    predictor.set_image(image)
    
    # Generate a mask using the box
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(box),
        multimask_output=True
    )
    
    # Get the best mask
    if len(masks) > 0 and scores.size > 0:
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        print(f"Generated mask with score: {scores[best_mask_idx]}")
        
        # Create contour from mask
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        
        if contours:
            # Get the largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Create flattened contour list
            contour_list = contour.tolist()
            if len(contour_list) > 0 and isinstance(contour_list[0], list) and len(contour_list[0]) == 1:
                contour_list = [point[0] for point in contour_list]
            
            # Create annotation dictionary
            annotation = {
                'id': 0,
                'class_id': 1,  # Use class "object"
                'class_name': annotator.class_names[1],
                'box': box,
                'contour': contour_list,
                'contour_points': contour,
                'mask': mask,
                'display_box': box,
                'area': np.sum(mask),
                'metadata': {'annotation_mode': 'box'}
            }
            
            # Add annotation using command manager
            command = AddAnnotationCommand(annotator.annotations, annotation, annotator.window_manager)
            annotator.command_manager.execute(command)
            print(f"Added annotation with ID: 0")
            
            # Save the annotations using file_manager
            original_dimensions = (height, width)
            display_dimensions = (height, width)  # Same as original in API case
            success = annotator.file_manager.save_annotations(
                annotations=annotator.annotations,
                image_name=os.path.basename(image_path),
                original_dimensions=original_dimensions,
                display_dimensions=display_dimensions,
                class_names=annotator.class_names,
                save_visualization=True
            )
            
            if success:
                print(f"Annotations saved to {working_dir}/labels/{os.path.splitext(os.path.basename(image_path))[0]}.txt")
            else:
                print("Failed to save annotations")
            
            # Export to COCO format
            coco_path = annotator.file_manager.handle_export("coco", annotator.class_names)
            print(f"Exported to COCO format: {coco_path}")
            
            # Visualize the mask
            visualization = image.copy()
            mask_overlay = np.zeros_like(visualization)
            mask_overlay[mask == 1] = [0, 255, 0]  # Green color for the mask
            alpha = 0.5
            visualization = cv2.addWeighted(visualization, 1, mask_overlay, alpha, 0)
            
            # Save the visualization
            vis_path = working_dir / "mask_overlay.jpg"
            cv2.imwrite(str(vis_path), visualization)
            print(f"Visualization saved to {vis_path}")
    else:
        print("No mask generated")
    
    print("Simple API example completed")

if __name__ == "__main__":
    main() 