# SAM Annotator

A tool for annotating images using the Segment Anything Model (SAM). This application provides an interface for creating and managing image annotations with the power of AI-assisted segmentation.

## Description

SAM Annotator leverages Meta AI's Segment Anything Model to help users quickly annotate images. The tool allows for both automatic segmentation and manual refinement, making the annotation process more efficient and accurate.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch
- CUDA-compatible GPU (recommended for optimal performance)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sam_annotator.git
   cd sam_annotator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
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

2. Load an image using the "Open Image" button

3. Use the available tools to annotate:
   - Click points for automatic segmentation
   - Draw boxes for region-based segmentation
   - Adjust masks with manual tools

4. Save annotations in various formats (COCO, YOLO, etc.)

## Features

- AI-assisted image segmentation using SAM
- Multiple annotation modes (point, box, manual)
- Support for various annotation formats
- Batch processing capabilities
- Annotation history and undo/redo functionality
- Custom label management

## Dependencies

- PyTorch
- OpenCV
- NumPy
- Segment Anything Model (SAM)
- PyQt5/PySide6 (for the GUI)

## License

[MIT License](LICENSE)

## Acknowledgements

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI Research 