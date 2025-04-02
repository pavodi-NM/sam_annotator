#!/usr/bin/env python
"""
Script to verify that version numbers in pyproject.toml and sam_annotator/__init__.py match.
This can be run as part of a CI/CD process or manually to ensure version consistency.

Usage:
  python verify_versions.py [--fix]
  
Options:
  --fix    Automatically fix version mismatches by updating __init__.py to match pyproject.toml
"""

import os
import re
import sys
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("version_verifier")

def get_version_from_pyproject():
    """
    Extract version from pyproject.toml file.
    
    Returns:
        str: Version string or None if not found
    """
    file_path = "pyproject.toml"
    if not os.path.exists(file_path):
        logger.error(f"{file_path} not found")
        return None
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            version = match.group(1)
            logger.info(f"Version in pyproject.toml: {version}")
            return version
        else:
            logger.error("Could not find version in pyproject.toml")
            return None
    
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return None

def get_version_from_init():
    """
    Extract version from sam_annotator/__init__.py file.
    
    Returns:
        str: Version string or None if not found
    """
    file_path = "sam_annotator/__init__.py"
    if not os.path.exists(file_path):
        logger.error(f"{file_path} not found")
        return None
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        match = re.search(r"__version__\s*=\s*'([^']+)'", content)
        if match:
            version = match.group(1)
            logger.info(f"Version in {file_path}: {version}")
            return version
        else:
            logger.error(f"Could not find __version__ in {file_path}")
            return None
    
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return None

def update_init_version(new_version):
    """
    Update version in __init__.py to match pyproject.toml.
    
    Args:
        new_version (str): Version to set in __init__.py
        
    Returns:
        bool: True if successful, False otherwise
    """
    file_path = "sam_annotator/__init__.py"
    if not os.path.exists(file_path):
        logger.error(f"{file_path} not found")
        return False
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        updated_content = re.sub(
            r"__version__\s*=\s*'[^']+'",
            f"__version__ = '{new_version}'",
            content
        )
        
        with open(file_path, "w") as f:
            f.write(updated_content)
        
        logger.info(f"Updated {file_path} version to {new_version}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")
        return False

def main():
    """Main function to verify version consistency."""
    parser = argparse.ArgumentParser(
        description="Verify that version numbers in pyproject.toml and sam_annotator/__init__.py match",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--fix", 
        action="store_true", 
        help="Automatically fix version mismatches by updating __init__.py"
    )
    
    args = parser.parse_args()
    
    # Get versions from both files
    pyproject_version = get_version_from_pyproject()
    init_version = get_version_from_init()
    
    # Exit if either version couldn't be found
    if pyproject_version is None or init_version is None:
        logger.error("Could not verify versions due to errors reading the files")
        sys.exit(1)
    
    # Compare versions
    if pyproject_version == init_version:
        logger.info("✅ Versions match! Both are: " + pyproject_version)
        sys.exit(0)
    else:
        logger.warning(f"❌ Version mismatch: pyproject.toml={pyproject_version}, __init__.py={init_version}")
        
        if args.fix:
            logger.info("Attempting to fix the mismatch...")
            if update_init_version(pyproject_version):
                logger.info("✅ Version mismatch fixed! Both are now: " + pyproject_version)
                sys.exit(0)
            else:
                logger.error("Failed to fix version mismatch")
                sys.exit(1)
        else:
            logger.info("Run with --fix to automatically update __init__.py to match pyproject.toml")
            sys.exit(1)

if __name__ == "__main__":
    main() 