#!/bin/bash

echo "Checking SAM Annotator installation..."

# Check if the package is installed
if python -c "import sam_annotator" &>/dev/null; then
    echo "✓ SAM Annotator is installed!"
    
    # Get the version
    version=$(python -c "import sam_annotator; print(sam_annotator.__version__)")
    echo "✓ Version: $version"
    
    # Check if the command-line tool is available
    if command -v sam_annotator &>/dev/null; then
        echo "✓ sam_annotator command-line tool is available!"
        echo ""
        echo "You can run 'sam_annotator --help' for available options."
        echo "Example usage:"
        echo "  sam_annotator --category_path /path/to/categories --classes_csv /path/to/classes.csv"
        echo ""
        echo "For quick version check:"
        echo "  sam_annotator --version"
    else
        echo "✗ sam_annotator command-line tool is not available."
        echo "This might be due to a PATH issue or installation problem."
    fi
else
    echo "✗ SAM Annotator is not installed correctly."
    echo "Try reinstalling with:"
    echo "  pip install -i https://test.pypi.org/simple/ sam-annotator==0.1.0.dev8"
    echo ""
    echo "If you experience dependency issues, try:"
    echo "  pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sam-annotator==0.1.0.dev8"
fi 