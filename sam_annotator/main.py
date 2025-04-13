import argparse
import logging
import os
import sys
import pandas as pd
import time
import json
from sam_annotator.core import SAMAnnotator 
from sam_annotator import __version__
from utils.standalone_viz import MultiMaskViewer, view_masks, find_classes_csv

# Constants for config file
CONFIG_FILE = ".sam_config.json"

def load_config():
    """Load configuration from config file if it exists."""  
    config = {
        "last_category_path": None,
        "last_classes_csv": None,
        "last_sam_version": "sam1",
        "last_model_type": None
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")  
    
    return config

def save_config(config):
    """Save configuration to config file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save config file: {e}")

def create_sample_csv(output_path, logger):
    """Create a sample CSV file with the correct format."""
    try:
        # Default class names
        class_names = [
            "background",
            "object",
            "person",
            "vehicle",
            "animal",
            "plant",
            "furniture",
            "building"
        ]
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame({"class_name": class_names})
        df.to_csv(output_path, index=False)
        
        logger.info(f"Created sample CSV file at: {output_path}")
        logger.info(f"Added {len(class_names)} classes")
        return True
    except Exception as e:
        logger.error(f"Error creating sample CSV file: {str(e)}")
        return False

def validate_csv(csv_path, logger):
    """
    Validate that a CSV file contains the required 'class_name' column.
    Returns True if valid, False otherwise.
    """
    try:
        # Try to read the CSV file
        df = pd.read_csv(csv_path)
        
        # First check if 'class_name' column exists directly
        if 'class_name' in df.columns:
            logger.info(f"CSV validation passed: Found {len(df)} classes in {csv_path}")
            return True
            
        # If not, check for alternative column names
        alternative_columns = ['className', 'name', 'class', 'category', 'label']
        found_column = None
        
        for col in alternative_columns:
            if col in df.columns:
                found_column = col
                logger.warning(f"CSV contains '{found_column}' column instead of 'class_name'")
                logger.info("Would you like to automatically fix this by renaming the column to 'class_name'? (y/n)")
                choice = input("> ").strip().lower()
                if choice == 'y':
                    # Rename the column without changing the data
                    df = df.rename(columns={found_column: 'class_name'})
                    # Save to the same file
                    df.to_csv(csv_path, index=False)
                    logger.info(f"CSV file has been fixed. The column '{found_column}' was renamed to 'class_name'.")
                    return True
                break
        
        # If there's only one column, offer to use it regardless of name
        if len(df.columns) == 1 and not found_column:
            first_col = df.columns[0]
            logger.warning(f"The first column of the CSV file must be 'class_name' but contains '{first_col}'")
            logger.info("Would you like to automatically fix this by introducing 'class_name' as the first column? (Recommended) (y/n)")
            choice = input("> ").strip().lower()
            if choice == 'y':
                # Rename the column without changing the data
                df = df.rename(columns={first_col: 'class_name'})
                # Save to the same file
                df.to_csv(csv_path, index=False)
                logger.info(f"CSV file has been fixed. The column '{first_col}' was renamed to 'class_name'.")
                return True
        
        # If we reach here, validation failed
        available_columns = ', '.join(f"'{col}'" for col in df.columns)
        logger.error(f"CSV validation failed: File does not contain required 'class_name' column")
        logger.error(f"Available columns: {available_columns}")
        
        # Suggest a fix
        logger.error("\nTo fix this issue:")
        logger.error("1. Open your CSV file in a text editor")
        logger.error("2. Make sure the first line is exactly: class_name")
        logger.error("3. Each subsequent line should contain one class name")
        logger.error("\nExample of valid CSV format:")
        logger.error("class_name")
        logger.error("background")
        logger.error("person")
        logger.error("car")
        logger.error("\nOr run with the --use_sample_csv flag to use the included sample file")
        
        # Offer to create a sample file
        logger.info("\nWould you like to create a sample CSV file instead? (y/n)")
        choice = input("> ").strip().lower()
        if choice == 'y':
            logger.info("Where would you like to save the sample file?")
            logger.info(f"1. At the original location: {csv_path}")
            logger.info(f"2. At the default location: sample_classes.csv")
            logger.info("3. Specify a different location")
            option = input("> ").strip()
            
            if option == '1':
                output_path = csv_path
            elif option == '2':
                output_path = "sample_classes.csv"
            elif option == '3':
                output_path = input("Enter the desired file path: ").strip()
            else:
                output_path = csv_path
                
            if create_sample_csv(output_path, logger):
                # Update the return logic - if we successfully created a sample CSV,
                # return True and update args.classes_csv to the new file path
                if output_path == csv_path:
                    # If we overwrote the original file, return True
                    return True
                else:
                    # If we created a new file, inform the user and return the path
                    logger.info(f"Successfully created sample CSV at: {output_path}")
                    logger.info(f"Please use this file with --classes_csv {output_path}")
                    return output_path  # Return the path so main() can update args.classes_csv
        
        return False
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        logger.error("\nMake sure your CSV file:")
        logger.error("1. Exists at the specified path")
        logger.error("2. Is properly formatted CSV")
        logger.error("3. Has the required 'class_name' header")
        
        # Check if the file exists but might not have headers
        if os.path.exists(csv_path):
            try:
                # Try to read file as text to check structure
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    # If it has content but no header or wrong header
                    logger.info("\nThe file exists but may not have the correct header.")
                    logger.info("Would you like to add 'class_name' as the header while preserving all content? (y/n)")
                    choice = input("> ").strip().lower()
                    if choice == 'y':
                        try:
                            # Read existing class names as raw data
                            class_names = []
                            for line in lines:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    class_names.append(line)
                            
                            # Create new dataframe with correct header and all data
                            df = pd.DataFrame({"class_name": class_names})
                            # Save to the same file
                            df.to_csv(csv_path, index=False)
                            logger.info(f"CSV file has been fixed. Added 'class_name' header while preserving {len(class_names)} classes.")
                            return True
                        except Exception as e2:
                            logger.error(f"Error fixing CSV file: {str(e2)}")
            except Exception:
                pass  # If this fails, continue with sample creation offer
        
        # Offer to create a sample file
        logger.info("\nWould you like to create a sample CSV file instead? (y/n)")
        choice = input("> ").strip().lower()
        if choice == 'y':
            logger.info("Where would you like to save the sample file?")
            logger.info(f"1. At the original location: {csv_path}")
            logger.info(f"2. At the default location: sample_classes.csv")
            logger.info("3. Specify a different location")
            option = input("> ").strip()
            
            if option == '1':
                output_path = csv_path
            elif option == '2':
                output_path = "sample_classes.csv"
            elif option == '3':
                output_path = input("Enter the desired file path: ").strip()
            else:
                output_path = csv_path
                
            if create_sample_csv(output_path, logger):
                # Update the return logic - if we successfully created a sample CSV,
                # return True and update args.classes_csv to the new file path
                if output_path == csv_path:
                    # If we overwrote the original file, return True
                    return True
                else:
                    # If we created a new file, inform the user and return the path
                    logger.info(f"Successfully created sample CSV at: {output_path}")
                    logger.info(f"Please use this file with --classes_csv {output_path}")
                    return output_path  # Return the path so main() can update args.classes_csv
        
        return False

# Set up a more detailed debug logging handler
def setup_debug_logging():
    # Create a file handler that logs to a debug.log file
    file_handler = logging.FileHandler('sam_debug.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Also log to console with a higher level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger and add the handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Return the logger for convenience
    return root_logger

def main():
    # Load saved configuration
    config = load_config()
    
    parser = argparse.ArgumentParser(description='SAM Multi-Object Annotation Tool')

    # Version information
    parser.add_argument('--version', action='version', 
                       version=f'%(prog)s {__version__}',
                       help='Show program version and exit')
    
    # Model configuration
    parser.add_argument('--sam_version', 
                       type=str,
                       choices=['sam1', 'sam2'],
                       default=config.get('last_sam_version', 'sam1'),
                       help='SAM version to use (sam1 or sam2)')
                       
    parser.add_argument('--model_type',
                       type=str,
                       default=config.get('last_model_type'),
                       help='Model type to use. For SAM1: vit_h, vit_l, vit_b. '
                            'For SAM2: tiny, small, base, large, tiny_v2, small_v2, base_v2, large_v2')
    
    parser.add_argument('--checkpoint', type=str, 
                       default=None,
                       help='Path to SAM checkpoint. If not provided, will use default for selected model')
    
    # Data paths - make them not required if --visualization is specified
    parser.add_argument('--category_path', type=str,
                       default=config.get('last_category_path'),
                       help='Path to category folder')
    parser.add_argument('--classes_csv', type=str,
                       default=config.get('last_classes_csv'),
                       help='Path to CSV file containing class names (must have a "class_name" column)')
    
    # Visualization option
    parser.add_argument('--visualization', action='store_true',
                       help='Launch visualization tool for reviewing annotations')
    parser.add_argument('--export_stats', action='store_true',
                       help='Export dataset statistics when using visualization tool')
    
    # CSV validation control
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip CSV validation (not recommended)')
    
    # Sample CSV options
    parser.add_argument('--use_sample_csv', action='store_true',
                       help='Use the included sample CSV file')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create a sample CSV file and exit')
    parser.add_argument('--sample_output', type=str,
                       help='Output path for the sample CSV file (used with --create_sample)')
    
    # Add debug flag
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose logging')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup enhanced logging if debug flag is set
    if args.debug:
        logger = setup_debug_logging()
    else:
        # Setup standard logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
    
    logger.info("SAM Annotator starting...")
    logger.info(f"Arguments: {args}")
    
    # If asked to create a sample file, do so and exit
    if args.create_sample:
        output_path = args.sample_output or "sample_classes.csv"
        if create_sample_csv(output_path, logger):
            logger.info("Sample CSV file created successfully. You can now use it with:")
            logger.info(f"python main.py --category_path your_category --classes_csv {output_path}")
        else:
            logger.error("Failed to create sample CSV file.")
        return
    
    # Validate required arguments based on mode
    if not args.visualization:
        if not args.category_path:
            logger.error("Error: --category_path is required for annotation mode")
            parser.print_help()
            sys.exit(1)
        if not args.classes_csv and not args.use_sample_csv:
            logger.error("Error: --classes_csv is required for annotation mode (or use --use_sample_csv)")
            parser.print_help()
            sys.exit(1)
    else:
        # For visualization mode, check if we have necessary paths
        if not args.category_path:
            if config.get('last_category_path'):
                args.category_path = config.get('last_category_path')
                logger.info(f"Using last used category path: {args.category_path}")
            else:
                logger.error("Error: --category_path is required for visualization")
                parser.print_help()
                sys.exit(1)
    
    # If visualization mode is enabled, launch the viewer instead of the annotator
    if args.visualization:
        logger.info(f"Launching visualization tool for {args.category_path}")
        
        # If no classes_csv is provided, try to find one related to the category path
        if not args.classes_csv:
            logger.info("No classes CSV specified, attempting to find one automatically")
            classes_csv = find_classes_csv(args.category_path)
            if classes_csv:
                logger.info(f"Found classes CSV file: {classes_csv}")
                args.classes_csv = classes_csv
                
                # Save this to config for future use
                config["last_classes_csv"] = classes_csv
                save_config(config)
            else:
                logger.warning("No classes CSV file found. Visualization will still work, but class names may not be displayed correctly.")
        
        try:
            view_masks(args.category_path, export_stats=args.export_stats, classes_csv=args.classes_csv)
            return
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}", exc_info=True)
            raise
    
    # If model_type not specified, set default based on sam_version
    if args.model_type is None:
        args.model_type = 'vit_h' if args.sam_version == 'sam1' else 'small_v2'
        
    if args.checkpoint is None and args.sam_version == 'sam1':
        args.checkpoint = "weights/sam_vit_h_4b8939.pth"
    
    # Log setup info
    logger.info(f"Using SAM version: {args.sam_version}")
    logger.info(f"Using model type: {args.model_type}")
    logger.info(f"Checkpoint path: {args.checkpoint}")
    
    # Use sample CSV if requested
    if args.use_sample_csv:
        sample_csv_path = "sample_classes.csv"
        # Create the sample file if it doesn't exist
        if not os.path.exists(sample_csv_path):
            logger.info(f"Sample CSV file not found at {sample_csv_path}. Creating it now.")
            create_sample_csv(sample_csv_path, logger)
        
        args.classes_csv = sample_csv_path
        logger.info(f"Using sample CSV file: {sample_csv_path}")
    
    try:
        # Validate CSV file unless skipped
        if not args.skip_validation:
            logger.info(f"Validating CSV file: {args.classes_csv}")
            validation_result = validate_csv(args.classes_csv, logger)
            
            # Handle different return values from validate_csv
            if isinstance(validation_result, str):
                # If a string is returned, it's the path to a newly created CSV file
                logger.info(f"Using newly created CSV file: {validation_result}")
                args.classes_csv = validation_result
            elif not validation_result:
                # If False is returned, validation failed and no alternative was created
                logger.error("CSV validation failed. Fix issues or use --skip_validation to bypass which is not recommended.")
                sys.exit(1)
            # If True, validation passed, continue with original file
        
        # Log start of SAMAnnotator initialization
        logger.info("Initializing SAMAnnotator...")
        start_time = time.time()
        
        # Save config for future use
        save_config({
            "last_category_path": args.category_path,
            "last_classes_csv": args.classes_csv,
            "last_sam_version": args.sam_version,
            "last_model_type": args.model_type
        })
        
        # Create and run annotator
        annotator = SAMAnnotator(
            checkpoint_path=args.checkpoint,
            category_path=args.category_path,
            classes_csv=args.classes_csv,
            sam_version=args.sam_version,
            model_type=args.model_type,  # Pass model_type to annotator
            #debug=args.debug  # Pass debug flag to annotator
        )
        
        init_time = time.time() - start_time
        logger.info(f"SAMAnnotator initialized in {init_time:.2f} seconds")
        
        logger.info("Starting SAMAnnotator run()")
        annotator.run() 
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__": 
    main()