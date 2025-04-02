#!/usr/bin/env python
"""
Script to update version numbers in both pyproject.toml and sam_annotator/__init__.py files.
This ensures consistent versioning across your package.

Usage:
  python update_version.py <new_version>
  
Examples:
  python update_version.py 0.1.0.dev12  # Development version
  python update_version.py 0.1.0a1      # Alpha release
  python update_version.py 0.1.0        # Stable release
  
The script follows the version numbering convention:
  - For development: 0.1.0.dev1, 0.1.0.dev2, etc.
  - For pre-releases: 0.1.0a1, 0.1.0b1, etc.
  - For stable releases: 0.1.0, 0.2.0, etc.
"""

import os
import re
import sys
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("version_updater")

def is_valid_version(version):
    """
    Validate that the version string follows Python's PEP 440 versioning scheme.
    
    Valid formats include:
    - 0.1.0
    - 0.1.0.dev1
    - 0.1.0a1
    - 0.1.0b1
    - 0.1.0rc1
    """
    # This is a simplified regex for PEP 440
    pattern = r"^\d+\.\d+\.\d+((a|b|rc|\.dev)\d+)?$"
    return bool(re.match(pattern, version))

def get_current_versions():
    """
    Get current versions from pyproject.toml and __init__.py files.
    
    Returns:
        tuple: (pyproject_version, init_version)
    """
    pyproject_version = None
    init_version = None
    
    # Get version from pyproject.toml
    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", "r") as f:
            content = f.read()
            match = re.search(r'version\s*=\s*"([^"]+)"', content)
            if match:
                pyproject_version = match.group(1)
                logger.info(f"Current version in pyproject.toml: {pyproject_version}")
            else:
                logger.warning("Could not find version in pyproject.toml")
    else:
        logger.warning("pyproject.toml file not found")
    
    # Get version from sam_annotator/__init__.py
    init_path = "sam_annotator/__init__.py"
    if os.path.exists(init_path):
        with open(init_path, "r") as f:
            content = f.read()
            match = re.search(r"__version__\s*=\s*'([^']+)'", content)
            if match:
                init_version = match.group(1)
                logger.info(f"Current version in {init_path}: {init_version}")
            else:
                logger.warning(f"Could not find __version__ in {init_path}")
    else:
        logger.warning(f"{init_path} file not found")
    
    return pyproject_version, init_version

def update_pyproject_toml(new_version):
    """
    Update the version in pyproject.toml file.
    
    Args:
        new_version (str): The new version to set
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    file_path = "pyproject.toml"
    if not os.path.exists(file_path):
        logger.error(f"{file_path} not found")
        return False
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        # Replace version in pyproject.toml
        updated_content = re.sub(
            r'version\s*=\s*"[^"]+"',
            f'version = "{new_version}"',
            content
        )
        
        # Write updated content back to file
        with open(file_path, "w") as f:
            f.write(updated_content)
        
        logger.info(f"Updated version in {file_path} to {new_version}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")
        return False

def update_init_py(new_version):
    """
    Update the __version__ in sam_annotator/__init__.py file.
    
    Args:
        new_version (str): The new version to set
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    file_path = "sam_annotator/__init__.py"
    if not os.path.exists(file_path):
        logger.error(f"{file_path} not found")
        return False
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        # Replace version in __init__.py
        updated_content = re.sub(
            r"__version__\s*=\s*'[^']+'",
            f"__version__ = '{new_version}'",
            content
        )
        
        # Write updated content back to file
        with open(file_path, "w") as f:
            f.write(updated_content)
        
        logger.info(f"Updated version in {file_path} to {new_version}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")
        return False

def backup_files():
    """
    Create backup copies of the files before modifying them.
    
    Returns:
        tuple: Paths to backup files (pyproject_backup, init_backup)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pyproject_backup = None
    init_backup = None
    
    if os.path.exists("pyproject.toml"):
        pyproject_backup = f"pyproject.toml.{timestamp}.bak"
        with open("pyproject.toml", "r") as src:
            with open(pyproject_backup, "w") as dst:
                dst.write(src.read())
        logger.info(f"Created backup: {pyproject_backup}")
    
    init_path = "sam_annotator/__init__.py"
    if os.path.exists(init_path):
        init_backup = f"{init_path}.{timestamp}.bak"
        with open(init_path, "r") as src:
            with open(init_backup, "w") as dst:
                dst.write(src.read())
        logger.info(f"Created backup: {init_backup}")
    
    return pyproject_backup, init_backup

def suggest_next_versions(current_version):
    """
    Suggest possible next versions based on the current version.
    
    Args:
        current_version (str): Current version string
        
    Returns:
        list: Suggested next versions
    """
    if not current_version:
        return ["0.1.0", "0.1.0.dev1"]
    
    suggestions = []
    
    # Parse the current version
    if ".dev" in current_version:
        # For development versions
        base_version, dev_num = current_version.split(".dev")
        next_dev = int(dev_num) + 1
        suggestions.append(f"{base_version}.dev{next_dev}")  # Next dev version
        suggestions.append(base_version)  # Stable version
        suggestions.append(f"{base_version}a1")  # Alpha version
    
    elif any(x in current_version for x in ["a", "b", "rc"]):
        # For pre-release versions
        match = re.match(r"(\d+\.\d+\.\d+)([a-z]+)(\d+)", current_version)
        if match:
            base_version, pre_type, pre_num = match.groups()
            next_pre = int(pre_num) + 1
            
            # Suggest next pre-release number
            suggestions.append(f"{base_version}{pre_type}{next_pre}")
            
            # Suggest next pre-release type
            if pre_type == "a":
                suggestions.append(f"{base_version}b1")
            elif pre_type == "b":
                suggestions.append(f"{base_version}rc1")
            elif pre_type == "rc":
                suggestions.append(base_version)  # Stable version
    
    else:
        # For stable versions
        parts = current_version.split(".")
        if len(parts) >= 3:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            
            # Suggest patch, minor, major increments
            suggestions.append(f"{major}.{minor}.{patch+1}")  # Patch increment
            suggestions.append(f"{major}.{minor+1}.0")  # Minor increment
            suggestions.append(f"{major+1}.0.0")  # Major increment
            
            # Suggest development version for next patch
            suggestions.append(f"{major}.{minor}.{patch+1}.dev1")
    
    return suggestions

def main():
    """Main function to update version numbers."""
    parser = argparse.ArgumentParser(
        description="Update version numbers in both pyproject.toml and sam_annotator/__init__.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "version", 
        nargs="?", 
        help="New version number (e.g., 0.1.0.dev12, 0.1.0a1, 0.1.0)"
    )
    parser.add_argument(
        "--no-backup", 
        action="store_true", 
        help="Skip creating backup files"
    )
    
    args = parser.parse_args()
    
    # Get current versions
    current_pyproject, current_init = get_current_versions()
    
    # If versions are inconsistent, warn the user
    if current_pyproject and current_init and current_pyproject != current_init:
        logger.warning(f"Version inconsistency detected: pyproject.toml={current_pyproject}, __init__.py={current_init}")
    
    # Use pyproject.toml version as the current version if both exist
    current_version = current_pyproject if current_pyproject else current_init
    
    # If no version is provided, suggest next versions and prompt user
    if not args.version:
        suggestions = suggest_next_versions(current_version)
        
        print("\nSuggested next versions:")
        for i, version in enumerate(suggestions, 1):
            print(f"{i}. {version}")
        
        choice = input("\nEnter a number to select a version, or type a custom version: ")
        
        if choice.isdigit() and 1 <= int(choice) <= len(suggestions):
            new_version = suggestions[int(choice) - 1]
        else:
            new_version = choice.strip()
    else:
        new_version = args.version
    
    # Validate the version
    if not is_valid_version(new_version):
        logger.error(f"Invalid version format: {new_version}")
        logger.info("Version should follow the format: X.Y.Z[.devN] or X.Y.Z[aN/bN/rcN]")
        sys.exit(1)
    
    # Create backups
    if not args.no_backup:
        backup_files()
    
    # Update both files
    pyproject_updated = update_pyproject_toml(new_version)
    init_updated = update_init_py(new_version)
    
    # Print summary
    print("\n=== Version Update Summary ===")
    print(f"Previous version: {current_version}")
    print(f"New version:      {new_version}")
    print(f"pyproject.toml:   {'✓ Updated' if pyproject_updated else '✗ Failed'}")
    print(f"__init__.py:      {'✓ Updated' if init_updated else '✗ Failed'}")
    
    if pyproject_updated and init_updated:
        print("\n✅ Version update completed successfully!")
        print("\nNext steps:")
        print("1. Build the package: rm -rf build/ dist/ *.egg-info/ && python -m build")
        print("2. Publish: python publish_dev_release.py")
    else:
        print("\n❌ Version update encountered some issues. Please check the errors above.")

if __name__ == "__main__":
    main() 