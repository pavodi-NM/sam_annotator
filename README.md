# SAM Annotator

A powerful tool for semi-automatic image annotation based on Meta AI's Segment Anything Model (SAM).

> **⚠️ DEVELOPMENT VERSION NOTICE**
>
> This package is currently in early development (alpha stage). The API and functionality may change 
> significantly between versions. Please, contact the author for any feedback or suggestions.

## Features

- **Semi-Automatic Segmentation**: Use SAM to generate high-quality masks with minimal interaction
- **Multiple Annotation Methods**: Box-based and point-based prompts
- **Class Management**: Assign classes to segmented objects
- **Export Options**: Export annotations to COCO, YOLO, or Pascal VOC formats
- **Undo/Redo**: Robust command system for undoing and redoing annotations
- **Visualization**: Real-time visualization of annotations with adjustable opacity

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- OpenCV
- NumPy
- CUDA-compatible GPU (recommended for optimal performance)

### Installation via pip

```bash
# Install the latest stable version (when available)
pip install sam_annotator
```

For the latest development version:

```bash
# For the latest development version from TestPyPI (specify exact version)
pip install -i https://test.pypi.org/simple/ sam-annotator==0.1.0.dev19

# If you experience dependency issues, add PyPI as an extra index
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sam-annotator==0.1.0.dev19
```

### Setup from source

1. Clone the repository:
   ```bash
   git clone https://github.com/pavodi-nm/sam_annotator.git
   cd sam_annotator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install directly:
   ```bash
   pip install -e .
   ```

3. Download SAM model weights (optional):
   - For SAM version 1: [Download SAM ViT-H Model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
   - For SAM version 2: [Download SAM 2 Model](https://dl.fbaipublicfiles.com/segment_anything/sam2_b.pth)

## Usage

### Running the Application

```bash
python main.py --sam_version [sam1|sam2] --model_type [vit_h|vit_l|vit_b|tiny|small|base|large|tiny_v2|small_v2|base_v2|large_v2] --category_path <path_to_category> --classes_csv <path_to_classes_csv> [--checkpoint <path_to_model_checkpoint>]
```

Example:
```bash
python main.py --sam_version sam1 --model_type vit_h --category_path test_data/s2 --classes_csv class_map_sam.csv
```

Or if installed via pip:
```
sam_annotator --sam_version sam1 --model_type vit_h --category_path /path/to/categories --classes_csv /path/to/classes.csv
```

For a complete list of command-line arguments and configuration options, see the [Configuration Options](https://pavodi-nm.github.io/sam_annotator/configuration/) documentation.

### Directory Structure

Create a directory structure for your dataset:
```
category_path/
├── images/     # Place your images here (mandatory)
├── labels/     # Annotation files will be saved here (optional)
├── masks/      # Visualization of masks will be saved here (optional)
├── metadata/   # Metadata about annotations (optional)
└── exports/    # Exported annotations in various formats (optional)
```

For more details on how annotations are stored and loaded, see the [Loading and Saving Annotations](https://pavodi-nm.github.io/sam_annotator/loading_saving/) documentation.

### Class Definition

Create a CSV file with class definitions:
```csv
class_name
background
person
car
...
```

## Annotation Methods

### Box-Based Annotation (default)

1. Left-click and drag to draw a bounding box around an object
2. Release to generate a segmentation mask
3. Press 'a' to add the annotation with the current class
4. Press 's' to save all the annotations
5. Press 'c' to clear all the annotations 

### Point-Based Annotation

1. Press 'w' to switch to point mode
2. Left-click to add foreground points (green)
3. Press 'a' to add the annotation with the current class
4. Press 's' to save all the annotations
5. Press 'c' to clear all the annotations 

## Keyboard Shortcuts

SAM Annotator provides a comprehensive set of keyboard shortcuts to improve your workflow efficiency:

| Key           | Action                          |
|---------------|----------------------------------|
| a             | Add current mask as annotation   |
| s             | Save annotations                 |
| n             | Next image                       |
| p             | Previous image                   |
| w             | Toggle between annotation modes  |
| x             | Clear current selection          |
| c             | Clear all annotations            |
| z             | Undo last action                 |
| y             | Redo last undone action          |
| m             | Toggle mask visibility           |
| b             | Toggle box visibility            |
| l             | Toggle label visibility          |
| r             | Toggle annotation review panel   |
| e             | Enter export mode                |
| q             | Quit application                 |

For a complete list of keyboard shortcuts and detailed usage examples, see the [Keyboard Shortcuts Documentation](https://pavodi-nm.github.io/sam_annotator/shortcuts/)

## Documentation

- [Getting Started](https://pavodi-nm.github.io/sam_annotator/)
- [Keyboard Shortcuts](https://pavodi-nm.github.io/sam_annotator/shortcuts/)
- [Loading and Saving Annotations](https://pavodi-nm.github.io/sam_annotator/loading_saving/)
- [Annotation Formats](https://pavodi-nm.github.io/sam_annotator/annotation_formats/)
- [Configuration Options](https://pavodi-nm.github.io/sam_annotator/configuration/)
- [Memory Management](https://pavodi-nm.github.io/sam_annotator/memory_management)
- [Implementation Details](https://pavodi-nm.github.io/sam_annotator/implementation/)
- [API Reference](https://pavodi-nm.github.io/sam_annotator/api_reference/)

## Windows and Controls

### Main Window

- Displays the current image with annotations
- Shows status information and current class
- Includes indicators for current mode and annotation count

### Class Window

- Displays available classes for selection
- Click on a class to select it for the next annotation

### Review Panel

- Shows list of current annotations
- Allows selection, deletion, and class change of annotations

### View Controls

- Adjust mask opacity
- Toggle visibility of masks, boxes, and labels

## Exporting Annotations

Press the corresponding key to export annotations in your preferred format:
- 'e' - Export to COCO format
- 'y' - Export to YOLO format
- 'v' - Export to Pascal VOC format

Exports will be saved in the `exports/` directory within your category path.

For more details on supported export formats and their structure, see the [Annotation Formats](https://pavodi-nm.github.io/sam_annotator/annotation_formats/) documentation.

## Advanced Usage

For more details on the implementation and advanced usage, see the [Implementation Details](https://pavodi-nm.github.io/sam_annotator/implementation/) documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/pavodi-nm/sam_annotator/blob/main/LICENSE) file for details.

When using this software, please cite or acknowledge:

```
SAM Annotator by Pavodi NDOYI MANIAMFU (FingerVision and University of Tsukuba - Adaptive Information Processing Lab - Japan)
https://github.com/pavodi-nm/sam_annotator
```

## Acknowledgments

- [Segment Anything Model (SAM)](https://segment-anything.com/) by Meta AI Research
- OpenCV for image processing
- PyTorch for deep learning