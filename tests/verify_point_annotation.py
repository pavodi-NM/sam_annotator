#!/usr/bin/env python
"""
Script to verify the point-based annotation functionality in both 
the root directory and the sam_annotator package.
"""

import os
import sys
import importlib
import inspect
import re
from pathlib import Path

def check_file_exists(path, filename):
    """Check if a file exists and print its status."""
    file_path = Path(path) / filename
    exists = file_path.exists()
    print(f"  {file_path}: {'✓' if exists else '✗'}")
    return exists

def check_method_exists(module_path, class_name, method_name):
    """Check if a method exists in a class and print its status."""
    try:
        # Import the module dynamically
        module_name = module_path.replace('/', '.')
        module = importlib.import_module(module_name)
        
        # Get the class
        class_obj = getattr(module, class_name)
        
        # Check if method exists
        method_exists = hasattr(class_obj, method_name)
        print(f"  {module_name}.{class_name}.{method_name}: {'✓' if method_exists else '✗'}")
        
        # If it exists, also check the source code
        if method_exists:
            method = getattr(class_obj, method_name)
            source_lines = inspect.getsource(method).count('\n')
            print(f"    - Method has {source_lines} lines of code")
            
        return method_exists
    
    except (ImportError, AttributeError) as e:
        print(f"  Error checking {module_path}.{class_name}.{method_name}: {str(e)}")
        return False

def scan_for_occurrences(directory, pattern):
    """Scan for pattern occurrences in Python files."""
    count = 0
    matched_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        matches = re.findall(pattern, content)
                        if matches:
                            count += len(matches)
                            matched_files.append((file_path, len(matches)))
                except Exception as e:
                    print(f"  Error reading {file_path}: {str(e)}")
    
    return count, matched_files

def main():
    print("Verifying point-based annotation functionality...\n")
    
    # Check files existence
    print("Checking file existence:")
    files_exist_src = check_file_exists("src/ui", "event_handler.py")
    files_exist_pkg = check_file_exists("sam_annotator/ui", "event_handler.py")
    
    # Check method existence in source
    print("\nChecking method existence in src:")
    try:
        sys.path.insert(0, os.getcwd())
        method_exists_src_point_mode = check_method_exists("src.ui.event_handler", "EventHandler", "_handle_point_mode")
        method_exists_src_point_pred = check_method_exists("src.core.annotator", "SAMAnnotator", "_handle_point_prediction") 
    except Exception as e:
        print(f"  Error checking methods in src: {str(e)}")
        method_exists_src_point_mode = False
        method_exists_src_point_pred = False
    
    # Check method existence in package
    print("\nChecking method existence in sam_annotator:")
    try:
        method_exists_pkg_point_mode = check_method_exists("sam_annotator.ui.event_handler", "EventHandler", "_handle_point_mode")
        method_exists_pkg_point_pred = check_method_exists("sam_annotator.core.annotator", "SAMAnnotator", "_handle_point_prediction")
    except Exception as e:
        print(f"  Error checking methods in sam_annotator: {str(e)}")
        method_exists_pkg_point_mode = False
        method_exists_pkg_point_pred = False
    
    # Scan for patterns
    print("\nScanning for key patterns in src directory:")
    patterns = [
        r"_handle_point_mode", 
        r"_handle_point_prediction",
        r"on_point_prediction",
        r"point_labels",
        r"event_handler.mode\s*==\s*'point'"
    ]
    
    for pattern in patterns:
        count_src, matches_src = scan_for_occurrences("src", pattern)
        print(f"  Pattern '{pattern}': {count_src} occurrences")
        for file_path, match_count in matches_src[:3]:  # Show top 3 matches
            print(f"    - {file_path}: {match_count} matches")
    
    print("\nScanning for key patterns in sam_annotator directory:")
    for pattern in patterns:
        count_pkg, matches_pkg = scan_for_occurrences("sam_annotator", pattern)
        print(f"  Pattern '{pattern}': {count_pkg} occurrences")
        for file_path, match_count in matches_pkg[:3]:  # Show top 3 matches
            print(f"    - {file_path}: {match_count} matches")
    
    # Summary
    print("\nSummary:")
    all_checks_passed = (
        files_exist_src and files_exist_pkg and
        method_exists_src_point_mode and method_exists_src_point_pred and
        method_exists_pkg_point_mode and method_exists_pkg_point_pred
    )
    
    if all_checks_passed:
        print("✓ All checks passed. Point-based annotation functionality appears to be properly synchronized.")
    else:
        print("✗ Some checks failed. There may be issues with the point-based annotation functionality.")
        
        # Provide specific recommendations
        if not files_exist_pkg:
            print("  - sam_annotator/ui/event_handler.py is missing!")
        if not method_exists_pkg_point_mode:
            print("  - _handle_point_mode method is missing in the package!")
        if not method_exists_pkg_point_pred:
            print("  - _handle_point_prediction method is missing in the package!")

if __name__ == "__main__":
    main() 