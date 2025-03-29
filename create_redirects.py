#!/usr/bin/env python
"""
Script to create redirect HTML files for GitHub Pages.
This helps ensure that old links to docs/file.md will redirect
to the proper GitHub Pages URL.
"""

import os
import glob

REDIRECT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0;url=https://pavodi-nm.github.io/sam_annotator/{target_page}">
    <title>Redirect to SAM Annotator Documentation</title>
</head>
<body>
    <p>The page has moved to: 
       <a href="https://pavodi-nm.github.io/sam_annotator/{target_page}">
          https://pavodi-nm.github.io/sam_annotator/{target_page}
       </a>
    </p>
</body>
</html>
"""

def main():
    # Create redirects directory if it doesn't exist
    if not os.path.exists('redirects'):
        os.makedirs('redirects')
    
    # Get all markdown files in docs
    md_files = glob.glob('docs/*.md')
    
    for md_file in md_files:
        # Extract the filename without path or extension
        filename = os.path.basename(md_file).replace('.md', '')
        
        # Create redirect file
        redirect_path = f'redirects/{filename}.html'
        with open(redirect_path, 'w') as f:
            f.write(REDIRECT_TEMPLATE.format(target_page=filename))
        
        print(f"Created redirect for {md_file} -> {redirect_path}")
    
    # Create a root index.html redirect to docs site
    with open('redirects/index.html', 'w') as f:
        f.write(REDIRECT_TEMPLATE.format(target_page=''))
    
    print("Created redirects/index.html pointing to documentation root")
    
    print("\nTo use these redirects:")
    print("1. Copy the 'redirects' directory to your repository root")
    print("2. Push the changes to GitHub")
    print("3. Old links like https://github.com/pavodi-nm/sam_annotator/blob/main/docs/shortcuts.md")
    print("   should include a notice to visit https://pavodi-nm.github.io/sam_annotator/shortcuts/")

if __name__ == "__main__":
    main() 