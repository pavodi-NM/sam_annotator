# SAM Annotator Keyboard Shortcuts

This document provides a comprehensive guide to the keyboard shortcuts available in the SAM Annotator tool. These shortcuts help users navigate the interface and perform operations efficiently.

## Basic Navigation

| Action | Shortcut | Description |
|--------|----------|-------------|
| Quit | <kbd>Q</kbd> | Exit the application |
| Next Image | <kbd>N</kbd> | Navigate to the next image in the dataset |
| Previous Image | <kbd>P</kbd> | Navigate to the previous image in the dataset |
| Save | <kbd>S</kbd> | Save current annotations |
| Clear Selection | <kbd>X</kbd> | Clear the current selection |
| Add Annotation | <kbd>A</kbd> | Add the current selection as an annotation |
| Undo | <kbd>Z</kbd> | Undo the last action |
| Redo | <kbd>Y</kbd> | Redo the previously undone action |
| Clear All | <kbd>C</kbd> | Clear all annotations on the current image |

## View Controls

| Action | Shortcut | Description |
|--------|----------|-------------|
| Toggle Masks | <kbd>M</kbd> | Show/hide segmentation masks |
| Toggle Boxes | <kbd>B</kbd> | Show/hide bounding boxes |
| Toggle Labels | <kbd>L</kbd> | Show/hide annotation labels |
| Toggle Points | <kbd>T</kbd> | Show/hide prompt points |
| Toggle View Controls | <kbd>V</kbd> | Show/hide view control panel |
| Toggle Review Mode | <kbd>R</kbd> | Enter/exit annotation review mode |

## Export Operations

SAM Annotator supports exporting annotations to various formats using a two-key sequence:

1. Press <kbd>E</kbd> to enter Export Mode
2. Press the format key:
   - <kbd>C</kbd> for COCO format
   - <kbd>Y</kbd> for YOLO format
   - <kbd>P</kbd> for Pascal VOC format

## Zoom Controls

| Action | Shortcut | Description |
|--------|----------|-------------|
| Zoom In | <kbd>=</kbd> | Increase zoom level |
| Zoom Out | <kbd>-</kbd> | Decrease zoom level |
| Reset Zoom | <kbd>0</kbd> | Reset zoom to 100% |

## Annotation Opacity Controls

| Action | Shortcut | Description |
|--------|----------|-------------|
| Increase Opacity | <kbd>]</kbd> | Make annotations more opaque |
| Decrease Opacity | <kbd>[</kbd> | Make annotations more transparent |

## Function Keys

| Action | Shortcut | Description |
|--------|----------|-------------|
| Help | <kbd>F1</kbd> | Show help information |
| Save View Settings | <kbd>F2</kbd> | Save current view configuration |
| Load View Settings | <kbd>F3</kbd> | Load saved view configuration |

## Using Shortcuts Effectively

### Tips for Efficient Annotation

1. **Learn the basics first:** Focus on mastering <kbd>A</kbd> (add annotation), <kbd>X</kbd> (clear selection), and <kbd>S</kbd> (save) for the most common workflow.

2. **Navigation efficiency:** Use <kbd>N</kbd> and <kbd>P</kbd> to quickly move through images without using the mouse.

3. **Customize view:** Toggle various view elements (<kbd>M</kbd>, <kbd>B</kbd>, <kbd>L</kbd>) based on your current task to reduce visual clutter.

4. **Review workflow:** Press <kbd>R</kbd> to enter review mode when you need to check your annotations without making accidental changes.

### Example Workflows

#### Basic Annotation Workflow
1. Navigate to an image (<kbd>N</kbd>/<kbd>P</kbd>)
2. Create a selection (using mouse)
3. Add the annotation (<kbd>A</kbd>)
4. Repeat steps 2-3 for all objects
5. Save work (<kbd>S</kbd>)
6. Move to next image (<kbd>N</kbd>)

#### Review and Correction Workflow
1. Enter review mode (<kbd>R</kbd>)
2. Navigate through images (<kbd>N</kbd>/<kbd>P</kbd>)
3. Exit review mode to make corrections (<kbd>R</kbd> again)
4. Use undo/redo (<kbd>Z</kbd>/<kbd>Y</kbd>) to fix errors
5. Save changes (<kbd>S</kbd>)

## Customizing Shortcuts

Shortcuts can be customized by modifying the `shortcuts.py` file in the configuration directory. The default location is:

```
sam_annotator/config/shortcuts.py
```

To customize shortcuts, edit the `SHORTCUTS` and `FUNCTION_SHORTCUTS` dictionaries in this file.

Example of customization:
```python
# Changing 'next_image' from 'n' to 'right arrow'
SHORTCUTS = {
    # ... other shortcuts ...
    'next_image': 'Right',  # Changed from 'n'
    # ... other shortcuts ...
}
```

> **Note:** After customizing shortcuts, restart the application for changes to take effect. 