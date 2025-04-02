#!/usr/bin/env python3
"""
Script to fix imports in test files from 'src.' to 'sam_annotator.'
"""

import os
import re
import glob
import logging
from pathlib import Path

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def fix_imports_in_file(file_path, logger):
    """Fix imports in a single file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if file contains src imports
    if 'from src.' in content or 'import src.' in content:
        # Replace imports
        modified_content = re.sub(r'from src\.', 'from sam_annotator.', content)
        modified_content = re.sub(r'import src\.', 'import sam_annotator.', modified_content)
        
        # Create backup
        backup_path = f"{file_path}.bak"
        os.rename(file_path, backup_path)
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        logger.info(f"Fixed imports in {file_path} (backup saved at {backup_path})")
        return True
    return False

def main():
    logger = setup_logging()
    logger.info("Starting import fix process")
    
    # Get the root directory
    root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Find all Python files
    python_files = []
    for pattern in ['test_*.py', 'tests/**/*.py']:
        python_files.extend(glob.glob(str(root_dir / pattern), recursive=True))
    
    # Process each file
    fixed_count = 0
    for file_path in python_files:
        if fix_imports_in_file(file_path, logger):
            fixed_count += 1
    
    logger.info(f"Fixed imports in {fixed_count} files")
    
    # Now check if there are any imports from src remaining
    remaining_count = 0
    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if 'from src.' in content or 'import src.' in content:
            logger.warning(f"File still has src imports: {file_path}")
            remaining_count += 1
    
    if remaining_count == 0:
        logger.info("All imports fixed successfully!")
    else:
        logger.warning(f"{remaining_count} files still have src imports")

if __name__ == "__main__":
    main() 