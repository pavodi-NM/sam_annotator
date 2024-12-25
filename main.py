import argparse
import logging
from src.core import SAMAnnotator, SAMWeightManager

def main():
    parser = argparse.ArgumentParser(description='SAM Multi-Object Annotation Tool')
    parser.add_argument('--checkpoint', type=str, 
                       default="weights/sam_vit_h_4b8939.pth",
                       help='Path to SAM checkpoint')
    parser.add_argument('--category_path', type=str, required=True,
                       help='Path to images directory')
    parser.add_argument('--classes_csv', type=str, required=True,
                       help='Path to CSV file containing class names')
    
    args = parser.parse_args()
    
    try:
        annotator = SAMAnnotator(args.checkpoint, 
                               args.category_path, 
                               args.classes_csv)
        annotator.run()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()