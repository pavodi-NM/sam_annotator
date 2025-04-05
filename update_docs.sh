#!/bin/bash

# Navigate to your private repository
cd ~/sam_annotator

# Build the documentation
echo "Building documentation..."
python -m mkdocs build

# Prepare the target repository
echo "Preparing clean repository..."
mkdir -p ~/temp_docs
mkdir -p ~/mypage/sam_annotator/

# Clone or reinitialize the repository in a temporary location
if [ -d "~/temp_docs" ]; then
    rm -rf ~/temp_docs
fi

# Create a fresh git repository in the temp directory
mkdir -p ~/temp_docs
cd ~/temp_docs
git init
echo "# SAM Annotator Documentation" > README.md
git add README.md
git commit -m "Initial commit"

# Copy site contents excluding the specified directories
echo "Copying documentation files (excluding large directories)..."
# Create a temporary exclude list
EXCLUDE_FILE=$(mktemp)
echo "weights/" >> $EXCLUDE_FILE
echo "output/" >> $EXCLUDE_FILE
echo "**/weights/" >> $EXCLUDE_FILE
echo "**/output/" >> $EXCLUDE_FILE

# Copy from original build to temp repository
rsync -av --exclude-from=$EXCLUDE_FILE ~/sam_annotator/site/ ~/temp_docs/

# Remove temporary file
rm $EXCLUDE_FILE

# Create .gitignore 
echo "Creating .gitignore file..."
cat > ~/temp_docs/.gitignore << EOL
# Ignore weights directory (large model files)
weights/
**/weights/

# Ignore output directory (generated content)
output/
**/output/

# Other common files to ignore
__pycache__/
*.py[cod]
*$py.class
*.so
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
.DS_Store
EOL

# Verify no large directories were copied
echo "Verifying clean copy..."
if [ -d ~/temp_docs/weights ]; then
    echo "WARNING: weights directory was copied despite exclusion!"
    rm -rf ~/temp_docs/weights
fi

if [ -d ~/temp_docs/output ]; then
    echo "WARNING: output directory was copied despite exclusion!"
    rm -rf ~/temp_docs/output
fi

# Show total size of files being committed
echo "Checking size of documentation files to be committed..."
du -sh ~/temp_docs/

# Commit everything in the temp repository
echo "Committing documentation files..."
cd ~/temp_docs
git add --all .
git commit -m "Update SAM Annotator documentation"

# Rename master branch to main
echo "Renaming master branch to main..."
git branch -m master main

# Move the clean repository to final location
echo "Replacing old repository with clean one..."
rm -rf ~/mypage/sam_annotator/.git
rm -rf ~/mypage/sam_annotator/*
cp -r ~/temp_docs/.git ~/mypage/sam_annotator/
cp -r ~/temp_docs/* ~/mypage/sam_annotator/
cp -r ~/temp_docs/.gitignore ~/mypage/sam_annotator/

# Push the clean repository
echo "Pushing clean repository..."
cd ~/mypage/sam_annotator
git remote add origin git@github.com:pavodi-nm/sam_annotator.git || true
git push -f origin main

# Cleanup
echo "Cleaning up temporary files..."
rm -rf ~/temp_docs

echo "Documentation updated successfully!" 