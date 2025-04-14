#!/usr/bin/env python3
"""
Script to prepare README for PyPI publication by replacing image tags with text links.
Run this before building the package for PyPI.
"""

import re
import os
import shutil

# Configuration
README_PATH = "README.md"
PYPI_README_PATH = "README_PYPI.md"
GITHUB_REPO_URL = "https://github.com/pavodi-NM/sam_annotator"
PRESERVE_ORIGINAL = True

def convert_readme_for_pypi():
    """Convert the GitHub README to a PyPI-friendly version."""
    if not os.path.exists(README_PATH):
        print(f"Error: {README_PATH} not found!")
        return False
    
    with open(README_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace image tags with text links
    img_pattern = r'!\[(.*?)\]\((.*?)\)'
    
    def replace_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        img_filename = os.path.basename(img_path)
        return f"[View {alt_text} on GitHub]({GITHUB_REPO_URL}/blob/main/{img_path})"
    
    pypi_content = re.sub(img_pattern, replace_image, content)
    
    # Add a note about images at the top
    note = """
> **Note:** This package has visual examples available on the [GitHub repository](https://github.com/pavodi-NM/sam_annotator).
> Links to images in this document will redirect you to the GitHub repository.

"""
    pypi_content = pypi_content.replace("# SAM Annotator", "# SAM Annotator" + note, 1)
    
    # Write the modified content
    with open(PYPI_README_PATH, 'w', encoding='utf-8') as f:
        f.write(pypi_content)
    
    print(f"Created PyPI-friendly README at {PYPI_README_PATH}")
    
    # If original should be preserved, make a backup
    if PRESERVE_ORIGINAL:
        return True
    else:
        # Replace the original README with the PyPI version
        shutil.move(PYPI_README_PATH, README_PATH)
        print(f"Replaced {README_PATH} with PyPI-friendly version")
    
    return True

if __name__ == "__main__":
    convert_readme_for_pypi() 