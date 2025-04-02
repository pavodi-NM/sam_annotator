#!/bin/bash

# sync_package.sh
# Script to synchronize the content from the root src/ directory to the sam_annotator subdirectory

# Set the source and destination directories
ROOT_DIR="."
PACKAGE_DIR="./sam_annotator"
SRC_DIR="./src"

# Create backup of setup.py and MANIFEST.in
cp setup.py setup.py.bak
cp MANIFEST.in MANIFEST.in.bak
cp README.md README.md.bak

echo "=========================================================="
echo "Syncing content from root directory to package directory"
echo "=========================================================="

# Check if the package directory exists
if [ ! -d "$PACKAGE_DIR" ]; then
    echo "Package directory does not exist. Creating..."
    mkdir -p "$PACKAGE_DIR"
fi

# Clean up the package directory (preserve __init__.py and main.py if they exist)
if [ -f "$PACKAGE_DIR/__init__.py" ]; then
    cp "$PACKAGE_DIR/__init__.py" ./temp_init.py
fi

if [ -f "$PACKAGE_DIR/main.py" ]; then
    cp "$PACKAGE_DIR/main.py" ./temp_main.py
fi

echo "Cleaning package directory..."
rm -rf "$PACKAGE_DIR"/*

# Restore __init__.py and main.py if they existed
mkdir -p "$PACKAGE_DIR"
if [ -f "./temp_init.py" ]; then
    mv ./temp_init.py "$PACKAGE_DIR/__init__.py"
fi

if [ -f "./temp_main.py" ]; then
    mv ./temp_main.py "$PACKAGE_DIR/main.py"
else
    # If no main.py existed, copy from root
    if [ -f "./main.py" ]; then
        cp ./main.py "$PACKAGE_DIR/main.py"
    fi
fi

# Copy main src/__init__.py if it exists
if [ -f "$SRC_DIR/__init__.py" ]; then
    echo "Copying src/__init__.py to package directory..."
    cp "$SRC_DIR/__init__.py" "$PACKAGE_DIR/__init__.py"
fi

# Copy directories from src to the package directory
echo "Copying subdirectories from src/ to package directory..."
for dir in core data utils ui config; do
    if [ -d "$SRC_DIR/$dir" ]; then
        echo "  - Copying $dir directory..."
        cp -r "$SRC_DIR/$dir" "$PACKAGE_DIR/"
        
        # Ensure each subdirectory has an __init__.py file
        if [ ! -f "$PACKAGE_DIR/$dir/__init__.py" ]; then
            echo "  - Creating missing __init__.py in $dir directory..."
            touch "$PACKAGE_DIR/$dir/__init__.py"
        fi
    else
        echo "  - Warning: $dir directory not found in src/, skipping..."
    fi
done

# Verify all key directories have __init__.py files
echo "Verifying __init__.py files in all package directories..."
for dir in core data utils ui config; do
    if [ -d "$PACKAGE_DIR/$dir" ] && [ ! -f "$PACKAGE_DIR/$dir/__init__.py" ]; then
        echo "  - Creating missing __init__.py in $PACKAGE_DIR/$dir"
        touch "$PACKAGE_DIR/$dir/__init__.py"
    fi
done

# Create __init__.py in all subdirectories to ensure proper Python package structure
echo "Creating __init__.py in all subdirectories for proper package structure..."
find "$PACKAGE_DIR" -type d -exec sh -c 'if [ ! -f "{}/__init__.py" ]; then touch "{}/__init__.py"; echo "  - Created __init__.py in {}"; fi' \;

# Update import statements in all Python files
echo "Updating import statements in all Python files..."
find "$PACKAGE_DIR" -type f -name "*.py" | while read -r file; do
    echo "  - Processing $file"
    sed -i 's/from src\./from sam_annotator./g' "$file"
    sed -i 's/import src\./import sam_annotator./g' "$file"
done

# Update import statements in test files
echo "Updating import statements in test files..."
find "$ROOT_DIR" -type f -name "test_*.py" | while read -r file; do
    echo "  - Processing test file $file"
    # Create a backup of the test file
    cp "$file" "${file}.bak"
    sed -i 's/from src\./from sam_annotator./g' "$file"
    sed -i 's/import src\./import sam_annotator./g' "$file"
done

# Also update imports in the tests directory
if [ -d "$ROOT_DIR/tests" ]; then
    echo "Updating import statements in tests directory..."
    find "$ROOT_DIR/tests" -type f -name "*.py" | while read -r file; do
        echo "  - Processing test file $file"
        # Create a backup of the test file
        cp "$file" "${file}.bak"
        sed -i 's/from src\./from sam_annotator./g' "$file"
        sed -i 's/import src\./import sam_annotator./g' "$file"
    done
fi

# Ensure point annotation functionality is properly copied
echo "Verifying point annotation functionality..."
if grep -q "_handle_point_prediction" "$PACKAGE_DIR/core/annotator.py"; then
    echo "  - Point prediction functionality found in annotator.py"
else
    echo "  - WARNING: Point prediction functionality not found in annotator.py!"
fi

if grep -q "_handle_point_mode" "$PACKAGE_DIR/ui/event_handler.py"; then
    echo "  - Point mode handling found in event_handler.py"
else
    echo "  - WARNING: Point mode handling not found in event_handler.py!"
fi

# Verify key components for point-based annotation
echo "Verifying key components for point-based annotation..."
for component in "point_prediction" "on_point_prediction" "point_mode"; do
    count=$(grep -r "$component" "$PACKAGE_DIR" | wc -l)
    echo "  - Found $count references to '$component'"
done

# Create data subdirectories if they exist in src but not in package
if [ -d "$PACKAGE_DIR/data" ]; then
    for subdir in $(find "$SRC_DIR/data" -type d); do
        rel_path=${subdir#"$SRC_DIR/data/"}
        if [ "$rel_path" != "" ]; then
            target_dir="$PACKAGE_DIR/data/$rel_path"
            if [ ! -d "$target_dir" ]; then
                echo "  - Creating missing data subdirectory: $target_dir"
                mkdir -p "$target_dir"
                touch "$target_dir/__init__.py"
            fi
        fi
    done
fi

# Check and sync version numbers
echo "=========================================================="
echo "Checking version numbers in pyproject.toml and __init__.py"
echo "=========================================================="

# Extract version from pyproject.toml
if [ -f "pyproject.toml" ]; then
    PYPROJECT_VERSION=$(grep -oP 'version\s*=\s*"\K[^"]+' pyproject.toml)
    echo "  - Version in pyproject.toml: $PYPROJECT_VERSION"
else
    echo "  - WARNING: pyproject.toml not found!"
    PYPROJECT_VERSION=""
fi

# Extract version from __init__.py
if [ -f "$PACKAGE_DIR/__init__.py" ]; then
    INIT_VERSION=$(grep -oP "__version__\s*=\s*'\K[^']+" "$PACKAGE_DIR/__init__.py")
    echo "  - Version in $PACKAGE_DIR/__init__.py: $INIT_VERSION"
else
    echo "  - WARNING: $PACKAGE_DIR/__init__.py not found!"
    INIT_VERSION=""
fi

# If both files exist and versions are different, update __init__.py
if [ -n "$PYPROJECT_VERSION" ] && [ -n "$INIT_VERSION" ] && [ "$PYPROJECT_VERSION" != "$INIT_VERSION" ]; then
    echo "  - Version mismatch detected!"
    echo "  - Updating version in $PACKAGE_DIR/__init__.py to match pyproject.toml"
    
    # Use sed to update the version in __init__.py
    sed -i "s/__version__ = '[^']*'/__version__ = '$PYPROJECT_VERSION'/" "$PACKAGE_DIR/__init__.py"
    echo "  - Version in $PACKAGE_DIR/__init__.py updated to: $PYPROJECT_VERSION"
elif [ -n "$PYPROJECT_VERSION" ] && [ -n "$INIT_VERSION" ]; then
    echo "  - Versions match. Both are: $PYPROJECT_VERSION"
fi

# Check if we have the verify_versions.py script and suggest using it
if [ -f "verify_versions.py" ]; then
    echo "  - You can also run 'python verify_versions.py --fix' to check and fix versions"
fi

# Print the structure of the package directory after syncing
echo "=========================================================="
echo "Package directory structure after syncing:"
find "$PACKAGE_DIR" -type f -name "*.py" | sort
echo "=========================================================="

# Check for any remaining src. imports in test files
echo "Checking for any remaining src. imports in test files..."
REMAINING_COUNT=$(grep -r "from src\." --include="*.py" "$ROOT_DIR" | wc -l)
REMAINING_COUNT2=$(grep -r "import src\." --include="*.py" "$ROOT_DIR" | wc -l)
TOTAL_REMAINING=$((REMAINING_COUNT + REMAINING_COUNT2))

if [ $TOTAL_REMAINING -eq 0 ]; then
    echo "  - All imports successfully converted!"
else
    echo "  - WARNING: Found $TOTAL_REMAINING remaining src. imports that need to be fixed!"
    echo "  - Consider running fix_test_imports.py to address these issues"
fi

echo "Sync completed."
echo "=========================================================="
echo "Please verify the content of the package directory and make any necessary adjustments."
echo "The following backups were created:"
echo "  - setup.py.bak"
echo "  - MANIFEST.in.bak"
echo "  - README.md.bak"
echo "  - Various test files (*.py.bak)"
echo "=========================================================="

# Remind about updating versions when making a new release
echo "IMPORTANT REMINDER:"
echo "When making a new release, remember to update version numbers in BOTH:"
echo "  - pyproject.toml"
echo "  - $PACKAGE_DIR/__init__.py"
echo "Or use the 'update_version.py' script if available."
echo "=========================================================="

# Make the script executable
# chmod +x sync_package.sh 