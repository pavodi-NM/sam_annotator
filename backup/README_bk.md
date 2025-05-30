# SAM Annotator

A tool for annotating images using the Segment Anything Model (SAM). This application provides an interface for creating and managing image annotations with the power of AI-assisted segmentation.

> **⚠️ DEVELOPMENT VERSION NOTICE**
>
> This package is currently in early development (alpha stage). The API and functionality may change 
> significantly between versions. Use in production environments is not recommended at this time.

## Description

SAM Annotator leverages Meta AI's Segment Anything Model to help users quickly annotate images. The tool allows for both automatic segmentation and manual refinement, making the annotation process more efficient and accurate.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch
- CUDA-compatible GPU (recommended for optimal performance)

### Installation via pip

```bash
# Install the latest stable version (when available)
pip install sam_annotator
```

For the latest development version:

```bash
# For the latest development version from TestPyPI (specify exact version)
pip install -i https://test.pypi.org/simple/ sam-annotator==0.1.0.dev8

# If you experience dependency issues, add PyPI as an extra index
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sam-annotator==0.1.0.dev8
```

After installation, you can run the tool using:

```bash
sam_annotator --category_path /path/to/categories --classes_csv /path/to/classes.csv
```

### Setup from source

1. Clone the repository:
   ```
   git clone https://github.com/pavodi-nm/sam_annotator.git
   cd sam_annotator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   
   Or install directly:
   ```
   pip install -e .
   ```

3. Download the SAM model weights:
   ```
   python download_models.py
   ```

## Usage

1. Start the application:
   ```
   python main.py
   ```
   
   Or if installed via pip:
   ```
   sam_annotator --category_path /path/to/categories --classes_csv /path/to/classes.csv
   ```

2. Load an image using the "Open Image" button

3. Use the available tools to annotate:
   - Click points for automatic segmentation
   - Draw boxes for region-based segmentation
   - Adjust masks with manual tools

4. Save annotations in various formats (COCO, YOLO, etc.)

## Keyboard Shortcuts

SAM Annotator provides numerous keyboard shortcuts to make your annotation workflow more efficient.

### Basic Navigation

| Action | Shortcut | Description |
|--------|----------|-------------|
| Quit | <kbd>Q</kbd> | Exit the application |
| Next Image | <kbd>N</kbd> | Navigate to the next image |
| Previous Image | <kbd>P</kbd> | Navigate to the previous image |
| Save | <kbd>S</kbd> | Save current annotations |
| Add Annotation | <kbd>A</kbd> | Add the current selection |
| Undo | <kbd>Z</kbd> | Undo the last action |
| Redo | <kbd>Y</kbd> | Redo the previously undone action |

For a complete list of keyboard shortcuts, see the [Keyboard Shortcuts Documentation](https://pavodi-nm.github.io/sam_annotator/shortcuts/).

## Documentation

- [Getting Started](https://pavodi-nm.github.io/sam_annotator/)
- [Keyboard Shortcuts](https://pavodi-nm.github.io/sam_annotator/shortcuts/)
- [API Reference](https://pavodi-nm.github.io/sam_annotator/placeholder/) (coming soon)

## Features

- AI-assisted image segmentation using SAM
- Multiple annotation modes (point, box, manual)
- Support for various annotation formats
- Batch processing capabilities
- Annotation history and undo/redo functionality
- Custom label management
- Extensive keyboard shortcuts for efficient workflows

## Dependencies

- PyTorch
- OpenCV
- NumPy
- Segment Anything Model (SAM)

## License

[MIT License](LICENSE)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

When using this software, please cite or acknowledge:

```
SAM Annotator by Pavodi NDOYI MANIAMFU
https://github.com/pavodi-nm/sam_annotator
```

## Acknowledgements

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI Research