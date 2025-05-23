"""CSV utilities for SAM Annotator."""

import os
import pandas as pd

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
    Returns True if valid, False otherwise, or path to new CSV if created.
    """
    # Handle None path
    if csv_path is None:
        logger.error("CSV path is None. Cannot validate.")
        return False
        
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
        return _offer_sample_creation(csv_path, logger)
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        logger.error("\nMake sure your CSV file:")
        logger.error("1. Exists at the specified path")
        logger.error("2. Is properly formatted CSV")
        logger.error("3. Has the required 'class_name' header")
        
        # Check if the file exists but might not have headers
        if csv_path and os.path.exists(csv_path):
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
        return _offer_sample_creation(csv_path, logger)

def _offer_sample_creation(csv_path, logger):
    """Offer to create a sample CSV file as a fallback."""
    logger.info("\nWould you like to create a sample CSV file instead? (y/n)")
    choice = input("> ").strip().lower()
    if choice == 'y':
        logger.info("Where would you like to save the sample file?")
        logger.info(f"1. At the original location: {csv_path or 'Not specified'}")
        logger.info(f"2. At the default location: sample_classes.csv")
        logger.info("3. Specify a different location")
        option = input("> ").strip()
        
        if option == '1' and csv_path:
            output_path = csv_path
        elif option == '1' and not csv_path:
            output_path = "sample_classes.csv"  # Default if no original path
        elif option == '2':
            output_path = "sample_classes.csv"
        elif option == '3':
            output_path = input("Enter the desired file path: ").strip()
        else:
            output_path = csv_path or "sample_classes.csv"
            
        if create_sample_csv(output_path, logger):
            # Update the return logic - if we successfully created a sample CSV,
            # return True or path based on the location
            if output_path == csv_path:
                # If we overwrote the original file, return True
                return True
            else:
                # If we created a new file, inform the user and return the path
                logger.info(f"Successfully created sample CSV at: {output_path}")
                logger.info(f"Please use this file with --classes_csv {output_path}")
                return output_path  # Return the path so main() can update args.classes_csv
    
    return False 