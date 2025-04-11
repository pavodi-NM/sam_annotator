import argparse
import logging
import os
import sys
import pandas as pd
import time
from src.core import SAMAnnotator 

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
                       help='Path to CSV file containing class names (must have a "class_name" column)')
    
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