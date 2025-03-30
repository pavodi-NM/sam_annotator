import argparse
import logging
import sys
from sam_annotator.core import SAMAnnotator
from sam_annotator import __version__

def main():
    parser = argparse.ArgumentParser(description='SAM Annotator - A tool for annotating images using the Segment Anything Model')
    
    # Add version argument
    parser.add_argument('--version', action='store_true',
                       help='Display the version of SAM Annotator')
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--sam_version', 
                       type=str,
                       choices=['sam1', 'sam2'],
                       default='sam1',
                       help='SAM version to use (sam1 or sam2)')
                       
    model_group.add_argument('--model_type',
                       type=str,
                       help='Model type to use. For SAM1: vit_h, vit_l, vit_b. '
                            'For SAM2: tiny, small, base, large, tiny_v2, small_v2, base_v2, large_v2')
    
    model_group.add_argument('--checkpoint', type=str, 
                       default=None,
                       help='Path to SAM checkpoint. If not provided, will use default for selected model')
    
    # Data paths
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--category_path', type=str,
                       help='Path to category folder')
    data_group.add_argument('--classes_csv', type=str,
                       help='Path to CSV file containing class names')
    
    args = parser.parse_args()
    
    # Handle version request
    if args.version:
        print(f"SAM Annotator version: {__version__}")
        return 0
    
    # Check required arguments when not showing version
    if args.category_path is None or args.classes_csv is None:
        parser.print_help()
        print("\nError: --category_path and --classes_csv are required to run SAM Annotator")
        return 1
    
    # If model_type not specified, set default based on sam_version
    if args.model_type is None:
        args.model_type = 'vit_h' if args.sam_version == 'sam1' else 'small_v2'
        
    if args.checkpoint is None and args.sam_version == 'sam1':
        args.checkpoint = "weights/sam_vit_h_4b8939.pth"
    
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        # Create and run annotator
        annotator = SAMAnnotator(
            checkpoint_path=args.checkpoint,
            category_path=args.category_path,
            classes_csv=args.classes_csv,
            sam_version=args.sam_version,
            model_type=args.model_type  # Pass model_type to annotator
        )
        
        annotator.run()
        return 0
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 