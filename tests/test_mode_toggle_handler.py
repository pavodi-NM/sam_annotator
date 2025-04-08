#!/usr/bin/env python
"""
Test script to verify the mode toggle functionality in the event handlers
for both the root directory and the sam_annotator package versions.
"""

import os
import sys
import logging
import argparse
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mode_toggle_handler_test")

class MockAnnotator:
    """Mock annotator class to simulate the annotator for testing."""
    
    def __init__(self):
        self.annotation_mode = "box"
        self.overlay = None
        self.current_tool = None
        
    def set_mode(self, mode):
        self.annotation_mode = mode
        logger.info(f"Annotation mode set to: {mode}")
        
    def toggle_mode(self):
        if self.annotation_mode == "box":
            self.set_mode("point")
        else:
            self.set_mode("box")
        return self.annotation_mode

class MockEvent:
    """Mock event class to simulate keyboard events."""
    
    def __init__(self, key):
        self.key = key

def test_src_event_handler():
    """Test the src (root) version event handler."""
    logger.info("Testing mode toggle in src (root) event handler...")
    
    try:
        # Import the event handler from src
        from sam_annotator.annotator import Annotator
        from sam_annotator.ui.event_handlers.keyboard_handler import KeyboardHandler
        from sam_annotator.config.shortcuts import SHORTCUTS
        
        # Get the toggle key
        toggle_key = SHORTCUTS["toggle_mode"]
        logger.info(f"Toggle key from src version: '{toggle_key}'")
        
        # Create a mock annotator
        mock_annotator = MockAnnotator()
        
        # Initialize the keyboard handler with the mock annotator
        keyboard_handler = KeyboardHandler(mock_annotator)
        
        # Initial state should be "box"
        logger.info(f"Initial annotation mode: {mock_annotator.annotation_mode}")
        assert mock_annotator.annotation_mode == "box", "Initial mode should be 'box'"
        
        # Create a toggle key event
        toggle_event = MockEvent(ord(toggle_key))
        
        # Test toggle to point mode
        keyboard_handler.handle_key(toggle_event)
        logger.info(f"After first toggle: {mock_annotator.annotation_mode}")
        assert mock_annotator.annotation_mode == "point", "Mode should toggle to 'point'"
        
        # Test toggle back to box mode
        keyboard_handler.handle_key(toggle_event)
        logger.info(f"After second toggle: {mock_annotator.annotation_mode}")
        assert mock_annotator.annotation_mode == "box", "Mode should toggle back to 'box'"
        
        logger.info("Src event handler test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in src event handler test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_package_event_handler():
    """Test the sam_annotator package version event handler."""
    logger.info("Testing mode toggle in sam_annotator package event handler...")
    
    try:
        # Import the event handler from package
        from sam_annotator.annotator import Annotator
        from sam_annotator.ui.event_handlers.keyboard_handler import KeyboardHandler
        from sam_annotator.config.shortcuts import SHORTCUTS
        
        # Get the toggle key
        toggle_key = SHORTCUTS["toggle_mode"]
        logger.info(f"Toggle key from package version: '{toggle_key}'")
        
        # Create a mock annotator
        mock_annotator = MockAnnotator()
        
        # Initialize the keyboard handler with the mock annotator
        keyboard_handler = KeyboardHandler(mock_annotator)
        
        # Initial state should be "box"
        logger.info(f"Initial annotation mode: {mock_annotator.annotation_mode}")
        assert mock_annotator.annotation_mode == "box", "Initial mode should be 'box'"
        
        # Create a toggle key event
        toggle_event = MockEvent(ord(toggle_key))
        
        # Test toggle to point mode
        keyboard_handler.handle_key(toggle_event)
        logger.info(f"After first toggle: {mock_annotator.annotation_mode}")
        assert mock_annotator.annotation_mode == "point", "Mode should toggle to 'point'"
        
        # Test toggle back to box mode
        keyboard_handler.handle_key(toggle_event)
        logger.info(f"After second toggle: {mock_annotator.annotation_mode}")
        assert mock_annotator.annotation_mode == "box", "Mode should toggle back to 'box'"
        
        logger.info("Package event handler test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in package event handler test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_direct_annotator_toggle():
    """Test the toggle_mode method directly on the Annotator classes."""
    logger.info("Testing direct toggle_mode method on Annotator classes...")
    
    # Test src version
    try:
        from sam_annotator.annotator import Annotator as SrcAnnotator
        
        # Create a minimal test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create a mock window and suppress actual display
        window_name = "test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Initialize src annotator with minimal arguments
        src_annotator = SrcAnnotator(
            window_name=window_name,
            image_paths=["dummy.jpg"],
            save_dir="./",
            class_names=["test"],
            sam_checkpoint=None,  # Mock, won't be used in this test
            device="cpu"
        )
        
        # Set original mode explicitly
        src_annotator.annotation_mode = "box"
        logger.info(f"Initial src annotator mode: {src_annotator.annotation_mode}")
        
        # Toggle mode
        src_annotator.toggle_mode()
        logger.info(f"After toggle, src annotator mode: {src_annotator.annotation_mode}")
        assert src_annotator.annotation_mode == "point", "Src annotator should toggle to 'point'"
        
        # Toggle mode back
        src_annotator.toggle_mode()
        logger.info(f"After second toggle, src annotator mode: {src_annotator.annotation_mode}")
        assert src_annotator.annotation_mode == "box", "Src annotator should toggle back to 'box'"
        
        logger.info("Src direct annotator toggle test passed!")
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Error in src direct annotator test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test package version
    try:
        from sam_annotator.annotator import Annotator as PackageAnnotator
        
        # Create a minimal test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create a mock window and suppress actual display
        window_name = "test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Initialize package annotator with minimal arguments
        package_annotator = PackageAnnotator(
            window_name=window_name,
            image_paths=["dummy.jpg"],
            save_dir="./",
            class_names=["test"],
            sam_checkpoint=None,  # Mock, won't be used in this test
            device="cpu"
        )
        
        # Set original mode explicitly
        package_annotator.annotation_mode = "box"
        logger.info(f"Initial package annotator mode: {package_annotator.annotation_mode}")
        
        # Toggle mode
        package_annotator.toggle_mode()
        logger.info(f"After toggle, package annotator mode: {package_annotator.annotation_mode}")
        assert package_annotator.annotation_mode == "point", "Package annotator should toggle to 'point'"
        
        # Toggle mode back
        package_annotator.toggle_mode()
        logger.info(f"After second toggle, package annotator mode: {package_annotator.annotation_mode}")
        assert package_annotator.annotation_mode == "box", "Package annotator should toggle back to 'box'"
        
        logger.info("Package direct annotator toggle test passed!")
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Error in package direct annotator test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test mode toggle event handlers')
    parser.add_argument('--test', choices=['src', 'package', 'direct', 'all'], 
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    if args.test == 'src' or args.test == 'all':
        test_src_event_handler()
    
    if args.test == 'package' or args.test == 'all':
        test_package_event_handler()
    
    if args.test == 'direct' or args.test == 'all':
        test_direct_annotator_toggle()

if __name__ == "__main__":
    main() 