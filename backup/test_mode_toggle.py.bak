#!/usr/bin/env python
"""
Test script to verify the mode toggle functionality works correctly
in both the root directory and the sam_annotator package.
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
import importlib
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mode_toggle_test")

class ModeToggleTest:
    """Test class for verifying mode toggle functionality."""
    
    def __init__(self):
        """Initialize test environment."""
        self.root_dir = os.getcwd()
        self.test_img = np.zeros((600, 800, 3), dtype=np.uint8)  # Create a blank test image
        
        # Ensure test environment is ready
        sys.path.insert(0, self.root_dir)
        
        # Load configuration for keyboard shortcuts
        try:
            from sam_annotator.config.shortcuts import SHORTCUTS as root_shortcuts
            self.root_shortcuts = root_shortcuts
            logger.info(f"Root shortcuts loaded: toggle_mode = '{root_shortcuts['toggle_mode']}'")
        except ImportError:
            logger.error("Failed to import root shortcuts")
            self.root_shortcuts = {"toggle_mode": "w"}  # Default
            
        try:
            from sam_annotator.config.shortcuts import SHORTCUTS as pkg_shortcuts
            self.pkg_shortcuts = pkg_shortcuts
            logger.info(f"Package shortcuts loaded: toggle_mode = '{pkg_shortcuts['toggle_mode']}'")
        except ImportError:
            logger.error("Failed to import package shortcuts")
            self.pkg_shortcuts = {"toggle_mode": "w"}  # Default
    
    def create_mock_window(self, window_name="Test Window"):
        """Create a mock window for testing."""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, self.test_img)
        return window_name
    
    def test_root_event_handler(self):
        """Test mode toggle in root event handler."""
        logger.info("\n--- Testing mode toggle in root event handler ---")
        
        try:
            # Import event handler from root
            from sam_annotator.ui.event_handler import EventHandler
            from sam_annotator.ui.window_manager import WindowManager
            
            # Create mock window manager and event handler
            window_manager = WindowManager()
            event_handler = EventHandler(window_manager, logger)
            
            # Test mode toggle
            logger.info(f"Initial mode: {event_handler.mode}")
            
            # Simulate pressing 'w' key (ASCII 119)
            action = event_handler.handle_keyboard_event(ord(self.root_shortcuts["toggle_mode"]))
            logger.info(f"After first toggle: mode = {event_handler.mode}, action = {action}")
            
            # Simulate pressing 'w' key again
            action = event_handler.handle_keyboard_event(ord(self.root_shortcuts["toggle_mode"]))
            logger.info(f"After second toggle: mode = {event_handler.mode}, action = {action}")
            
            # Verify toggles worked correctly
            result = (action == 'switch_mode_box')
            logger.info(f"Root mode toggle test {'PASSED' if result else 'FAILED'}")
            return result
            
        except Exception as e:
            logger.error(f"Error testing root event handler: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_package_event_handler(self):
        """Test mode toggle in package event handler."""
        logger.info("\n--- Testing mode toggle in package event handler ---")
        
        try:
            # Import event handler from package
            from sam_annotator.ui.event_handler import EventHandler
            from sam_annotator.ui.window_manager import WindowManager
            
            # Create mock window manager and event handler
            window_manager = WindowManager()
            event_handler = EventHandler(window_manager, logger)
            
            # Test mode toggle
            logger.info(f"Initial mode: {event_handler.mode}")
            
            # Simulate pressing 'w' key
            action = event_handler.handle_keyboard_event(ord(self.pkg_shortcuts["toggle_mode"]))
            logger.info(f"After first toggle: mode = {event_handler.mode}, action = {action}")
            
            # Simulate pressing 'w' key again
            action = event_handler.handle_keyboard_event(ord(self.pkg_shortcuts["toggle_mode"]))
            logger.info(f"After second toggle: mode = {event_handler.mode}, action = {action}")
            
            # Verify toggles worked correctly
            result = (action == 'switch_mode_box')
            logger.info(f"Package mode toggle test {'PASSED' if result else 'FAILED'}")
            return result
            
        except Exception as e:
            logger.error(f"Error testing package event handler: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def test_visual_mode_toggle(self):
        """Visual test for mode toggle (requires manual verification)."""
        logger.info("\n--- Visual test for mode toggle ---")
        logger.info("This test requires manual verification.")
        logger.info("Press 'w' to toggle between box and point mode.")
        logger.info("Press 'q' to exit the test.")
        
        try:
            # Import required modules from root
            from sam_annotator.ui.window_manager import WindowManager
            from sam_annotator.ui.event_handler import EventHandler
            from sam_annotator.ui.widgets.status_overlay import StatusOverlay
            
            # Create test components
            window_name = "Mode Toggle Test"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Create status overlay
            overlay = StatusOverlay()
            
            # Set initial state
            mode = "box"
            
            while True:
                # Create display image
                display = self.test_img.copy()
                
                # Add overlay with current mode
                display = overlay.render(
                    display,
                    status="Press 'w' to toggle mode, 'q' to exit",
                    current_class="Test",
                    current_class_id=0,
                    current_image_path="test.jpg",
                    current_idx=0,
                    total_images=1,
                    num_annotations=0,
                    annotation_mode=mode
                )
                
                # Show display
                cv2.imshow(window_name, display)
                
                # Wait for key press
                key = cv2.waitKey(100) & 0xFF
                
                # Handle key press
                if key == ord('q'):
                    break
                elif key == ord(self.root_shortcuts["toggle_mode"]):
                    mode = "point" if mode == "box" else "box"
                    logger.info(f"Mode toggled to: {mode}")
            
            cv2.destroyAllWindows()
            return True
            
        except Exception as e:
            logger.error(f"Error in visual test: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_tests(self):
        """Run all tests."""
        logger.info("Starting mode toggle tests...")
        
        root_result = self.test_root_event_handler()
        pkg_result = self.test_package_event_handler()
        
        logger.info("\n--- Test Results ---")
        logger.info(f"Root event handler mode toggle: {'PASSED' if root_result else 'FAILED'}")
        logger.info(f"Package event handler mode toggle: {'PASSED' if pkg_result else 'FAILED'}")
        
        if root_result and pkg_result:
            logger.info("\nAll tests PASSED! Mode toggle functionality works correctly in both versions.")
        else:
            logger.warning("\nSome tests FAILED! There may be issues with mode toggle functionality.")
            
        # Ask if user wants to run visual test
        response = input("\nDo you want to run the visual test? (y/n): ")
        if response.lower() == 'y':
            logger.info("Running visual test...")
            self.test_visual_mode_toggle()
        else:
            logger.info("Skipping visual test.")

if __name__ == "__main__":
    tester = ModeToggleTest()
    tester.run_tests() 