import argparse
import logging
from sam_annotator.core import SAMAnnotator 

def main():
    parser = argparse.ArgumentParser(description='SAM Multi-Object Annotation Tool')
    
    # Model configuration
    parser.add_argument('--sam_version', 
                       type=str,
                       choices=['sam1', 'sam2'],
                       default='sam1',
                       help='SAM version to use (sam1 or sam2)')
                       
    parser.add_argument('--model_type',
                       type=str,
                       help='Model type to use. For SAM1: vit_h, vit_l, vit_b. '
                            'For SAM2: tiny, small, base, large, tiny_v2, small_v2, base_v2, large_v2')
    
    parser.add_argument('--checkpoint', type=str, 
                       default=None,
                       help='Path to SAM checkpoint. If not provided, will use default for selected model')
    
    # Data paths
    parser.add_argument('--category_path', type=str, required=True,
                       help='Path to category folder')
    parser.add_argument('--classes_csv', type=str, required=True,
                       help='Path to CSV file containing class names')
    
    args = parser.parse_args()
    
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
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 