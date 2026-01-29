"""Main entry point for SAM Annotator."""

import logging
import os
import sys
import time

# Add parent directory to path if needed for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from sam_annotator package
from sam_annotator.core import SAMAnnotator
from sam_annotator.utils.standalone_viz import view_masks, find_classes_csv

# Import CLI components
from sam_annotator.cli.config import (
    load_config,
    save_config,
    get_model_type_for_version,
    set_model_type_for_version,
)
from sam_annotator.cli.csv_utils import create_sample_csv, validate_csv
from sam_annotator.cli.logging_utils import setup_standard_logging, setup_debug_logging
from sam_annotator.cli.parser import parse_args
from sam_annotator.cli.preflight import (
    run_preflight_checks,
    check_ultralytics_version,
    get_ultralytics_upgrade_message,
)

# Import model registry for centralized model type handling
from sam_annotator.core.model_registry import (
    resolve_model_type,
    get_default_model_type,
    get_version_requirements,
    supports_auto_download,
)


def main(args=None):
    """Main entry point for SAM Annotator.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])
    """
    # Load saved configuration
    config = load_config()

    # Parse arguments
    args = parse_args(config, args)

    # Setup logging
    logger = setup_debug_logging() if args.debug else setup_standard_logging()

    logger.info("SAM Annotator starting...")
    logger.info(f"Arguments: {args}")

    # If asked to create a sample file, do so and exit
    if args.create_sample:
        output_path = args.sample_output or "sample_classes.csv"
        if create_sample_csv(output_path, logger):
            logger.info("Sample CSV file created successfully. You can now use it with:")
            logger.info(f"sam_annotator --category_path your_category --classes_csv {output_path}")
        else:
            logger.error("Failed to create sample CSV file.")
        return

    # Validate required arguments based on mode
    if not args.visualization:
        if not args.category_path:
            logger.error("Error: --category_path is required for annotation mode")
            sys.exit(1)
        if not args.classes_csv and not args.use_sample_csv:
            logger.error("Error: --classes_csv is required for annotation mode (or use --use_sample_csv)")
            sys.exit(1)
    else:
        # For visualization mode, check if we have necessary paths
        if not args.category_path:
            if config.get('last_category_path'):
                args.category_path = config.get('last_category_path')
                logger.info(f"Using last used category path: {args.category_path}")
            else:
                logger.error("Error: --category_path is required for visualization")
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

    # ========================================================================
    # RESOLVE MODEL TYPE using centralized model registry
    # ========================================================================
    # Get the config-stored model type for this specific SAM version
    config_model_type = get_model_type_for_version(config, args.sam_version)

    # Resolve model type with proper precedence:
    # 1. CLI argument (if provided and valid)
    # 2. Config default for this version (if valid)
    # 3. Hardcoded default
    resolved_model_type, source = resolve_model_type(
        sam_version=args.sam_version,
        cli_model_type=args.model_type,  # May be None if not specified
        config_model_type=config_model_type
    )

    args.model_type = resolved_model_type
    logger.info(f"Resolved model type: {args.model_type} (source: {source})")

    # NOTE: We intentionally do NOT set default checkpoint paths here for auto-download versions.
    # The weight_manager will handle default paths and auto-downloading.
    # We only use args.checkpoint if the user explicitly specified one via --checkpoint.

    # Log setup info
    logger.info(f"Using SAM version: {args.sam_version}")
    logger.info(f"Using model type: {args.model_type}")
    logger.info(f"Checkpoint path: {args.checkpoint}")

    # ========================================================================
    # PRE-FLIGHT CHECKS
    # ========================================================================
    # Check ultralytics version for SAM3
    if args.sam_version == 'sam3':
        requirements = get_version_requirements('sam3')
        if requirements:
            min_version = requirements.get('min_ultralytics_version', '8.3.237')
            version_ok, current_version = check_ultralytics_version(min_version)
            if not version_ok:
                print(get_ultralytics_upgrade_message(min_version, current_version), file=sys.stderr)
                sys.exit(1)

    # Run pre-flight checks (weight availability, etc.)
    preflight_passed, resolved_checkpoint = run_preflight_checks(
        sam_version=args.sam_version,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        logger=logger
    )

    if not preflight_passed:
        logger.error("Pre-flight checks failed. Please resolve the issues and try again.")
        sys.exit(1)

    # Use resolved checkpoint if one was found
    if resolved_checkpoint:
        args.checkpoint = resolved_checkpoint
        logger.info(f"Using resolved checkpoint: {args.checkpoint}")

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
        updated_config = {
            "last_category_path": args.category_path,
            "last_classes_csv": args.classes_csv,
            "last_sam_version": args.sam_version,
        }
        # Only save model_type if user explicitly specified it via --model_type
        # This prevents defaults from becoming "sticky" in the config
        if source == "cli":
            updated_config = set_model_type_for_version(
                updated_config,
                args.sam_version,
                args.model_type
            )
        save_config(updated_config)

        # Create and run annotator
        annotator = SAMAnnotator(
            checkpoint_path=args.checkpoint,
            category_path=args.category_path,
            classes_csv=args.classes_csv,
            sam_version=args.sam_version,
            model_type=args.model_type,
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
