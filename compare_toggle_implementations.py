#!/usr/bin/env python
"""
Script to compare the implementation of the toggle mode functionality
between the root directory and the sam_annotator package.
"""

import os
import sys
import logging
import difflib
import inspect
import importlib
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("toggle_implementations_comparison")

def get_method_source(module_path, class_name, method_name):
    """
    Get the source code of a specific method from a module.
    
    Args:
        module_path (str): The import path of the module
        class_name (str): The name of the class containing the method
        method_name (str): The name of the method to extract
        
    Returns:
        str: The source code of the method
    """
    try:
        module = importlib.import_module(module_path)
        class_obj = getattr(module, class_name)
        method = getattr(class_obj, method_name)
        source = inspect.getsource(method)
        return source
    except Exception as e:
        logger.error(f"Error extracting source for {module_path}.{class_name}.{method_name}: {str(e)}")
        return None

def compare_methods(src_module_path, package_module_path, class_name, method_name):
    """
    Compare the implementation of a method between the src and package versions.
    
    Args:
        src_module_path (str): The import path of the src module
        package_module_path (str): The import path of the package module
        class_name (str): The name of the class containing the method
        method_name (str): The name of the method to compare
    """
    logger.info(f"Comparing {method_name} method in {class_name}...")
    
    src_source = get_method_source(src_module_path, class_name, method_name)
    package_source = get_method_source(package_module_path, class_name, method_name)
    
    if src_source is None or package_source is None:
        logger.error("Could not extract source for comparison")
        return
    
    # Check if the implementations are identical
    if src_source == package_source:
        logger.info(f"✅ The {method_name} implementations are identical")
        return
    
    # Generate a diff
    src_lines = src_source.splitlines()
    package_lines = package_source.splitlines()
    
    diff = list(difflib.unified_diff(
        src_lines, 
        package_lines,
        fromfile=f"src/{class_name}.{method_name}",
        tofile=f"package/{class_name}.{method_name}",
        lineterm=''
    ))
    
    if diff:
        logger.warning(f"⚠️ Differences found in {method_name} implementation:")
        for line in diff:
            print(line)

def compare_handler_methods(src_handler_module, package_handler_module, handler_class, handler_method):
    """
    Compare the implementation of a handler method between src and package versions.
    
    Args:
        src_handler_module (str): The import path of the src handler module
        package_handler_module (str): The import path of the package handler module
        handler_class (str): The name of the handler class
        handler_method (str): The name of the handler method to compare
    """
    logger.info(f"Comparing {handler_method} in {handler_class}...")
    
    src_source = get_method_source(src_handler_module, handler_class, handler_method)
    package_source = get_method_source(package_handler_module, handler_class, handler_method)
    
    if src_source is None or package_source is None:
        logger.error("Could not extract source for comparison")
        return
    
    # Check if the implementations are identical
    if src_source == package_source:
        logger.info(f"✅ The {handler_method} implementations in {handler_class} are identical")
        return
    
    # Generate a diff
    src_lines = src_source.splitlines()
    package_lines = package_source.splitlines()
    
    diff = list(difflib.unified_diff(
        src_lines, 
        package_lines,
        fromfile=f"src/{handler_class}.{handler_method}",
        tofile=f"package/{handler_class}.{handler_method}",
        lineterm=''
    ))
    
    if diff:
        logger.warning(f"⚠️ Differences found in {handler_class}.{handler_method} implementation:")
        for line in diff:
            print(line)

def compare_shortcut_definitions():
    """Compare the shortcut key definitions between src and package."""
    logger.info("Comparing shortcut definitions...")
    
    try:
        # Import both shortcut modules
        from src.config.shortcuts import SHORTCUTS as SRC_SHORTCUTS
        from sam_annotator.config.shortcuts import SHORTCUTS as PACKAGE_SHORTCUTS
        
        # Compare the toggle_mode shortcut
        src_toggle_key = SRC_SHORTCUTS.get("toggle_mode")
        package_toggle_key = PACKAGE_SHORTCUTS.get("toggle_mode")
        
        if src_toggle_key == package_toggle_key:
            logger.info(f"✅ Toggle mode shortcut is identical in both versions: '{src_toggle_key}'")
        else:
            logger.warning(f"⚠️ Toggle mode shortcut differs: src='{src_toggle_key}', package='{package_toggle_key}'")
        
        # Check for other differences in shortcuts that might affect behavior
        all_keys = set(list(SRC_SHORTCUTS.keys()) + list(PACKAGE_SHORTCUTS.keys()))
        for key in all_keys:
            src_value = SRC_SHORTCUTS.get(key, "NOT_DEFINED")
            package_value = PACKAGE_SHORTCUTS.get(key, "NOT_DEFINED")
            
            if src_value != package_value:
                logger.warning(f"⚠️ Shortcut '{key}' differs: src='{src_value}', package='{package_value}'")
        
    except Exception as e:
        logger.error(f"Error comparing shortcuts: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def check_all_toggle_related_code():
    """Check all code related to the toggle functionality."""
    logger.info("Performing comprehensive check of all toggle-related code...")
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    # 1. Compare Annotator.toggle_mode method
    compare_methods(
        "src.annotator", 
        "sam_annotator.annotator", 
        "Annotator", 
        "toggle_mode"
    )
    
    # 2. Compare KeyboardHandler.handle_key method
    compare_handler_methods(
        "src.ui.event_handlers.keyboard_handler",
        "sam_annotator.ui.event_handlers.keyboard_handler",
        "KeyboardHandler",
        "handle_key"
    )
    
    # 3. Compare shortcut definitions
    compare_shortcut_definitions()
    
    # 4. Compare the set_mode method if it exists
    compare_methods(
        "src.annotator", 
        "sam_annotator.annotator", 
        "Annotator", 
        "set_mode"
    )
    
    # 5. Compare the status overlay rendering that might show the current mode
    try:
        compare_handler_methods(
            "src.ui.widgets.status_overlay",
            "sam_annotator.ui.widgets.status_overlay",
            "StatusOverlay",
            "render"
        )
    except Exception as e:
        logger.error(f"Error comparing StatusOverlay.render: {str(e)}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare toggle mode implementations')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Show more detailed comparison output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Perform the comprehensive check
    check_all_toggle_related_code()
    
    logger.info("Comparison complete.")

if __name__ == "__main__":
    main() 