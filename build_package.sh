#!/bin/bash
# Script to build and install the sam_annotator package

echo "Cleaning up previous builds..."
rm -rf build/ dist/ *.egg-info/

echo "Building package..."
python -m pip install --upgrade pip
python -m pip install --upgrade build
python -m build

echo "Build completed."
echo "To install the package locally, run:"
echo "pip install dist/*.whl"

echo "To upload to PyPI, run:"
echo "python -m pip install --upgrade twine"
echo "python -m twine upload dist/*" 