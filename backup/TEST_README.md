# Mode Toggle Test Scripts

This directory contains a set of test scripts designed to verify the "mode toggle" functionality in the SAM Annotator. The mode toggle allows switching between "box" and "point" annotation modes by pressing the 'w' key.

The scripts test both the root directory version (`src/`) and the package version (`sam_annotator/`) to ensure consistency between them.

## Test Scripts Overview

### 1. `test_mode_toggle.py`

This is the main test script that programmatically verifies the toggle functionality works correctly in both versions.

**Usage:**
```bash
python test_mode_toggle.py
```

**What it tests:**
- Creates mock objects to simulate the SAM Annotator environment
- Tests that pressing the 'w' key (or the configured toggle key) correctly changes the annotation mode
- Verifies the toggle works in both directions (box → point → box)
- Tests both the root and package implementations

### 2. `test_mode_toggle_visual.py`

This script provides a visual interface to manually verify the mode toggle functionality.

**Usage:**
```bash
python test_mode_toggle_visual.py [--version src|package|both]
```

**What it tests:**
- Displays a window showing the current annotation mode
- Allows you to press the toggle key ('w') to see the mode change visually
- Tests both versions and lets you switch between them
- Helpful for visual confirmation that the UI is updating correctly

### 3. `test_mode_toggle_handler.py`

This script specifically tests the event handler implementation that processes key presses.

**Usage:**
```bash
python test_mode_toggle_handler.py [--test src|package|direct|all]
```

**What it tests:**
- Tests the keyboard handler's handling of the toggle key event
- Tests the direct `toggle_mode` method on the Annotator classes
- More focused on the internal implementation than the visual aspects

### 4. `compare_toggle_implementations.py`

This script analyzes the source code to compare the implementations between versions.

**Usage:**
```bash
python compare_toggle_implementations.py [--verbose]
```

**What it tests:**
- Compares the source code of the `toggle_mode` method between versions
- Compares the event handler implementations
- Checks for differences in shortcut key definitions
- Helps identify any inconsistencies in the code

## Running All Tests

To run a complete verification of the mode toggle functionality, execute all scripts in sequence:

```bash
python test_mode_toggle.py
python test_mode_toggle_handler.py --test all
python compare_toggle_implementations.py
python test_mode_toggle_visual.py  # For manual visual verification
```

## Troubleshooting

If any tests fail:

1. Check the differences reported by `compare_toggle_implementations.py`
2. Verify that both versions use the same shortcut key for toggle_mode
3. Check for any differences in method implementations that might cause inconsistent behavior
4. Make sure both versions properly update the UI to reflect the current mode

## Requirements

These test scripts require:
- Python 3.6+
- OpenCV (cv2)
- NumPy
- Access to both the `src` and `sam_annotator` package code 