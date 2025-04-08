#!/usr/bin/env python3
"""
Visual test script for mode toggle functionality with a focus on the mode indicator display.
This script verifies that the mode indicator is properly displayed when toggling between modes.
"""

import os
import sys
import time
import cv2
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try both import paths to ensure compatibility
def try_imports():
    """Try importing from different module paths and return which one succeeded."""
    try:
        # Try package import first
        from sam_annotator.ui.widgets.status_overlay import StatusOverlay
        from sam_annotator.config.shortcuts import SHORTCUTS
        logger.info("Successfully imported from sam_annotator package")
        return "package", StatusOverlay, SHORTCUTS
    except ImportError as e:
        logger.warning(f"Failed to import from sam_annotator package: {e}")
        try:
            # Fall back to src import
            from sam_annotator.ui.widgets.status_overlay import StatusOverlay
            from sam_annotator.config.shortcuts import SHORTCUTS
            logger.info("Successfully imported from src directory")
            return "src", StatusOverlay, SHORTCUTS
        except ImportError as e:
            logger.error(f"Failed to import from src directory: {e}")
            logger.error("Could not import required modules. Exiting.")
            sys.exit(1)

# Try imports and get required modules
import_source, StatusOverlay, SHORTCUTS = try_imports()

class ModeToggleVisualTest:
    """Visual test for mode toggle functionality and mode indicator display"""
    
    def __init__(self):
        # Create a blank canvas for display
        self.canvas_size = (800, 600)
        self.canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
        
        # Initialize mode and status overlay
        self.mode = "box"  # Start with box mode
        self.overlay = StatusOverlay()
        
        # Create window
        self.window_name = "Mode Toggle Visual Test"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.canvas_size[0], self.canvas_size[1])
        
        # Register keyboard handler
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self.running = True
    
    def _mouse_callback(self, event, x, y, flags, param):
        # Simple mouse callback to allow for interaction
        pass
    
    def toggle_mode(self):
        """Toggle between box and point modes"""
        if self.mode == "box":
            self.mode = "point"
            logger.info("Toggled to POINT mode")
        else:
            self.mode = "box"
            logger.info("Toggled to BOX mode")
    
    def update_display(self):
        """Update the display with current mode"""
        # Create a fresh canvas
        self.canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
        
        # Add instructions
        instructions = [
            f"Mode Toggle Visual Test ({import_source} import)",
            f"Current working directory: {os.getcwd()}",
            "Press 'w' to toggle between Box and Point modes",
            "Press 'q' to quit",
            "",
            f"Current Mode: {self.mode.upper()}"
        ]
        
        y_offset = 50
        for line in instructions:
            cv2.putText(
                self.canvas, line, (50, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA
            )
            y_offset += 40
        
        # Add status overlay with mode indicator
        image_with_overlay = self.overlay.render(
            self.canvas.copy(),
            mode=self.mode,
            class_id=0,
            class_name="Test Class",
        )
        
        # Display the result
        cv2.imshow(self.window_name, image_with_overlay)
    
    def run(self):
        """Run the visual test"""
        logger.info("Starting Mode Toggle Visual Test")
        logger.info(f"Using {import_source} import")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        while self.running:
            self.update_display()
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q'):
                self.running = False
                logger.info("Quitting test")
                break
            elif key == ord('w'):
                self.toggle_mode()
        
        cv2.destroyAllWindows()


def main():
    test = ModeToggleVisualTest()
    test.run()


if __name__ == "__main__":
    main() 