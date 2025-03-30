#!/bin/bash

# Clear pip cache to ensure fresh metadata
echo "Clearing pip cache..."
pip cache purge

# Install the latest version from TestPyPI
echo "Installing latest SAM Annotator from TestPyPI..."
pip install -i https://test.pypi.org/simple/ --no-cache-dir sam-annotator==0.1.0.dev8

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✓ SAM Annotator installed successfully!"
    
    # Get the version
    version=$(python -c "import sam_annotator; print(sam_annotator.__version__)")
    echo "✓ Installed version: $version"
    
    echo ""
    echo "You can now run 'sam_annotator --version' to verify the installation."
else
    echo "✗ Installation failed."
    echo ""
    echo "Trying alternative installation with extra index URL..."
    pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --no-cache-dir sam-annotator==0.1.0.dev8
    
    if [ $? -eq 0 ]; then
        echo "✓ SAM Annotator installed successfully with extra index!"
        version=$(python -c "import sam_annotator; print(sam_annotator.__version__)")
        echo "✓ Installed version: $version"
    else
        echo "✗ Installation failed. Please check your network connection and try again."
    fi
fi 

# pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sam-annotator==0.1.0.dev7
#    python -m build
#   python -m twine upload --repository testpypi dist/*