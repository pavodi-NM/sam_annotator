#!/usr/bin/env python
"""
Script to fix the sam_annotator entry point.
"""

import os
import sys
import stat
import subprocess

def main():
    """Create the entry point script in the bin directory."""
    # Get the bin directory using the command output
    try:
        # For WSL, use this approach to get the correct bin directory
        python_path = subprocess.check_output(["which", "python"], text=True).strip()
        bin_dir = os.path.dirname(python_path)
    except:
        # Fallback to sys.prefix/bin
        bin_dir = os.path.join(sys.prefix, 'bin')
    
    print(f"Using bin directory: {bin_dir}")
    
    # Define the entry point script path
    script_path = os.path.join(bin_dir, 'sam_annotator')
    
    # Define the script content
    script_content = """#!/usr/bin/env python
from sam_annotator.main import main

if __name__ == "__main__":
    main()
"""
    
    # Write the script
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make it executable
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        
        print(f"Entry point script created at: {script_path}")
        print("Now you should be able to run 'sam_annotator' command.")
    except Exception as e:
        print(f"Error creating entry point: {e}")
        print("\nAlternative approach:")
        print("1. Create a file named 'sam_annotator' in your bin directory with this content:")
        print("#!/usr/bin/env python")
        print("from sam_annotator.main import main")
        print("if __name__ == \"__main__\":")
        print("    main()")
        print("\n2. Make it executable with: chmod +x path/to/sam_annotator")

if __name__ == "__main__":
    main() 