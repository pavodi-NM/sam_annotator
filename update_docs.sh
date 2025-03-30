#!/bin/bash

# Navigate to your private repository
cd ~/sam_annotator

# Build the documentation
echo "Building documentation..."
python -m mkdocs build

# Copy to your public repository
echo "Copying files to GitHub Pages repository..."
cp -r site/* ~/mypage/sam_annotator/

# Commit and push the changes
echo "Committing and pushing changes..."
cd ~/mypage
git add .
git commit -m "Update SAM Annotator documentation"
git push

echo "Documentation updated successfully!" 