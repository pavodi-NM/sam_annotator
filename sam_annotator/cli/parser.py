"""Command-line argument parser for SAM Annotator."""

import argparse
import sys
from sam_annotator import __version__
from sam_annotator.core.model_registry import (
    get_supported_versions,
    get_model_type_help_text,
)


def create_parser(config):
    """Create and return argument parser with all options.

    Args:
        config: Dictionary containing configuration values to use as defaults

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='SAM Multi-Object Annotation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with SAM2 (default)
  sam_annotator --category_path ./data --classes_csv classes.csv

  # Run with SAM1
  sam_annotator --sam_version sam1 --model_type vit_h --category_path ./data --classes_csv classes.csv

  # Run with SAM3 (requires manual weight download)
  sam_annotator --sam_version sam3 --category_path ./data --classes_csv classes.csv

  # Run visualization mode
  sam_annotator --visualization --category_path ./data
"""
    )

    # Version information
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}',
                        help='Show program version and exit')

    # Model configuration
    supported_versions = get_supported_versions()
    parser.add_argument('--sam_version',
                        type=str,
                        choices=supported_versions,
                        default=config.get('last_sam_version', 'sam2'),
                        help='SAM version to use: '
                             'sam1 (Meta original), '
                             'sam2 (Ultralytics SAM2), '
                             'sam3 (Ultralytics SAM3 - requires manual download)')

    # Model type - don't set default here, let main.py resolve it based on sam_version
    model_type_help = get_model_type_help_text()
    parser.add_argument('--model_type',
                        type=str,
                        default=None,  # Will be resolved in main.py based on sam_version
                        help=f'Model type to use. {model_type_help}')

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

    return parser


def parse_args(config, args=None):
    """Parse command line arguments.

    Args:
        config: Dictionary containing configuration values to use as defaults
        args: Command line arguments to parse (defaults to sys.argv[1:])

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = create_parser(config)
    # Explicitly use [] as default rather than None to avoid using sys.argv
    # This helps avoid issues with test frameworks passing their own args
    return parser.parse_args([] if args is None else args)
