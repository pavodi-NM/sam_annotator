#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from pathlib import Path
from sam_annotator.core.annotator import SAMAnnotator
from sam_annotator.core.command_manager import AddAnnotationCommand

"""
This is a simple test script to demonstrate how to properly save annotations
using the SAM Annotator API. Use this to test if your code has been fixed.
"""

def main():
    """Test script for saving annotations."""
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
    
    # Create a simple test image
    image_path = working_dir / "images" / "test.jpg"
    if not image_path.exists():
        test_image = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (400, 400), (0, 255, 0), -1)
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
    
    # Create a simple mask (a rectangle in this case)
    mask = np.zeros((height, width), dtype=bool)
    mask[150:350, 150:350] = True
    
    # Create contour from mask
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]  # Should be only one contour
    
    # Create box from contour
    x, y, w, h = cv2.boundingRect(contour)
    box = [x, y, x + w, y + h]
    
    # Create an annotation dictionary
    annotation = {
        'id': 0,
        'class_id': 1,
        'class_name': "object",
        'box': box,
        'contour': contour.tolist(),
        'contour_points': contour,
        'mask': mask,
        'display_box': box,
        'area': np.sum(mask),
        'metadata': {'annotation_mode': 'manual'}
    }
    
    # Add annotation using command manager
    command = AddAnnotationCommand(annotator.annotations, annotation, annotator.window_manager)
    annotator.command_manager.execute(command)
    print(f"Added annotation with ID: 0")
    
    # For generating masks using the predictor, you would need to:
    # 1. Set the image in the predictor
    # predictor = annotator.predictor
    # predictor.set_image(image)
    # 2. Then predict with a box or points
    # masks, scores, _ = predictor.predict(...)
    
    # Save the annotations using file_manager - THIS IS THE CORRECT WAY
    print("\nCorrect way to save annotations:")
    print("-------------------------------")
    print("Using annotator.file_manager.save_annotations()")
    
    success = annotator.file_manager.save_annotations(
        annotations=annotator.annotations,
        image_name=os.path.basename(image_path),
        original_dimensions=(height, width),
        display_dimensions=(height, width),
        class_names=annotator.class_names,
        save_visualization=True
    )
    
    if success:
        print("✓ SUCCESS: Annotations saved correctly")
        print(f"Annotations saved to {working_dir}/labels/{os.path.splitext(os.path.basename(image_path))[0]}.txt")
    else:
        print("✗ ERROR: Failed to save annotations")
    
    # INCORRECT way (will raise an AttributeError)
    print("\nIncorrect way to save annotations:")
    print("---------------------------------")
    print("Using annotator.session_manager.save_annotations()")
    
    try:
        # This will raise an AttributeError
        annotator.session_manager.save_annotations()
        print("✗ ERROR: This should have failed but didn't (your API might be different)")
    except AttributeError as e:
        print(f"✓ EXPECTED ERROR: {e}")
        print("This is the expected behavior - session_manager has no save_annotations method")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 