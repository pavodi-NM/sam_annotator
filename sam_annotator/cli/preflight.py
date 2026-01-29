"""Pre-flight checks for SAM Annotator.

This module provides early validation before starting the application,
with user-friendly formatted messages for missing requirements.
"""

import os
import sys
import logging
from typing import Optional, Tuple, List

from sam_annotator.core.model_registry import (
    MODEL_REGISTRY,
    supports_auto_download,
    get_version_requirements,
    validate_version_and_model,
)


def check_weights_available(
    sam_version: str,
    model_type: str,
    checkpoint_path: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Check if weights are available for the specified SAM version.

    Args:
        sam_version: The SAM version (e.g., 'sam1', 'sam2', 'sam3')
        model_type: The model type
        checkpoint_path: Optional custom checkpoint path from CLI

    Returns:
        Tuple of (is_available, weight_path)
        If not available and requires manual download, weight_path is None
    """
    # If user specified a custom checkpoint, check if it exists
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            return True, checkpoint_path
        else:
            return False, None

    # For auto-download versions, weights will be downloaded automatically
    if supports_auto_download(sam_version):
        return True, None

    # For manual download versions (like SAM3), check if weights exist
    requirements = get_version_requirements(sam_version)
    if not requirements:
        return True, None

    weight_filename = requirements.get("weight_filename", f"{sam_version}.pt")

    # Check standard locations
    possible_paths = [
        os.path.join("weights", weight_filename),
        os.path.join(os.getcwd(), weight_filename),
        os.path.join(os.getcwd(), "weights", weight_filename),
        os.path.join(os.path.expanduser("~"), ".cache", sam_version, weight_filename),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return True, path

    return False, None


def get_missing_weights_message(sam_version: str) -> str:
    """Generate a formatted message for missing weights.

    Args:
        sam_version: The SAM version with missing weights

    Returns:
        Formatted string with download instructions
    """
    requirements = get_version_requirements(sam_version)

    if not requirements:
        return f"Weights for {sam_version.upper()} are missing."

    download_url = requirements.get("download_url", "N/A")
    weight_filename = requirements.get("weight_filename", f"{sam_version}.pt")
    weight_size = requirements.get("weight_size_gb", "Unknown")
    min_ultralytics = requirements.get("min_ultralytics_version", "N/A")
    min_gpu_memory = requirements.get("min_gpu_memory_gb", "N/A")
    min_ram = requirements.get("min_system_ram_gb", "N/A")

    # Build the formatted message
    border = "═" * 70
    separator = "─" * 40

    message = f"""
╔{border}╗
║{f"{sam_version.upper()} WEIGHTS NOT FOUND":^70}║
╠{border}╣
║{" " * 70}║
║  {sam_version.upper()} requires manual download (automatic download not available).{" " * (70 - 67)}║
║{" " * 70}║
║  {"STEPS TO DOWNLOAD:":<68}║
║  {separator:<68}║
║  {"1. Visit: " + download_url:<68}║
║  {"2. Log in or create a HuggingFace account":<68}║
║  {"3. Request access to the model (may require approval)":<68}║
║  {"4. Download: " + weight_filename + f" (~{weight_size} GB)":<68}║
║  {"5. Place the file in: ./weights/" + weight_filename:<68}║
║{" " * 70}║
║  {"ALTERNATIVE: Specify a custom path":<68}║
║  {separator:<68}║
║  {"sam_annotator --sam_version " + sam_version + " --checkpoint /path/to/" + weight_filename:<68}║
║{" " * 70}║
║  {"REQUIREMENTS:":<68}║
║  {separator:<68}║
║  {"• ultralytics >= " + str(min_ultralytics):<68}║
║  {"• GPU with " + str(min_gpu_memory) + "+ GB VRAM recommended":<68}║
║  {"• System RAM: " + str(min_ram) + "+ GB":<68}║
║{" " * 70}║
╚{border}╝
"""
    return message


def get_invalid_checkpoint_message(checkpoint_path: str) -> str:
    """Generate a formatted message for invalid checkpoint path.

    Args:
        checkpoint_path: The path that was specified but doesn't exist

    Returns:
        Formatted error message
    """
    border = "═" * 70

    message = f"""
╔{border}╗
║{"CHECKPOINT FILE NOT FOUND":^70}║
╠{border}╣
║{" " * 70}║
║  {"The specified checkpoint file does not exist:":<68}║
║  {checkpoint_path:<68}║
║{" " * 70}║
║  {"Please verify the path and try again.":<68}║
║{" " * 70}║
╚{border}╝
"""
    return message


def run_preflight_checks(
    sam_version: str,
    model_type: str,
    checkpoint_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, Optional[str]]:
    """Run all pre-flight checks before starting the application.

    Args:
        sam_version: The SAM version to use
        model_type: The model type to use
        checkpoint_path: Optional custom checkpoint path (only set if user specified --checkpoint)
        logger: Optional logger instance

    Returns:
        Tuple of (passed, resolved_checkpoint_path)
        If checks fail, prints message and returns (False, None)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # 1. Validate version and model type combination
    is_valid, error_msg = validate_version_and_model(sam_version, model_type)
    if not is_valid:
        print(f"\n❌ Configuration Error: {error_msg}\n", file=sys.stderr)
        return False, None

    # 2. If user specified a custom checkpoint path, verify it exists
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            logger.info(f"Using user-specified checkpoint: {checkpoint_path}")
            return True, checkpoint_path
        else:
            # User specified a path but it doesn't exist - this is an error
            print(get_invalid_checkpoint_message(checkpoint_path), file=sys.stderr)
            return False, None

    # 3. No custom checkpoint specified - check if weights are available or can be auto-downloaded
    if supports_auto_download(sam_version):
        # SAM1 and SAM2 support auto-download, let weight_manager handle it
        logger.info(f"{sam_version.upper()} supports auto-download. Weights will be downloaded if needed.")
        return True, None

    # 4. For manual download versions (like SAM3), check if weights exist
    weights_available, resolved_path = check_weights_available(
        sam_version, model_type, checkpoint_path
    )

    if not weights_available:
        # Show detailed instructions for manual download
        print(get_missing_weights_message(sam_version), file=sys.stderr)
        return False, None

    return True, resolved_path


def check_ultralytics_version(min_version: str) -> Tuple[bool, str]:
    """Check if installed ultralytics version meets minimum requirement.

    Args:
        min_version: Minimum required version (e.g., "8.3.237")

    Returns:
        Tuple of (meets_requirement, installed_version)
    """
    try:
        import ultralytics
        installed = ultralytics.__version__

        # Parse versions for comparison
        def parse_version(v: str) -> Tuple[int, ...]:
            return tuple(int(x) for x in v.split(".")[:3])

        installed_tuple = parse_version(installed)
        min_tuple = parse_version(min_version)

        return installed_tuple >= min_tuple, installed
    except ImportError:
        return False, "not installed"
    except Exception:
        return False, "unknown"


def get_ultralytics_upgrade_message(min_version: str, current_version: str) -> str:
    """Generate a message for upgrading ultralytics.

    Args:
        min_version: Required minimum version
        current_version: Currently installed version

    Returns:
        Formatted upgrade instructions
    """
    border = "═" * 70

    message = f"""
╔{border}╗
║{"ULTRALYTICS VERSION TOO OLD":^70}║
╠{border}╣
║{" " * 70}║
║  {"Current version: " + current_version:<68}║
║  {"Required version: " + min_version + "+":<68}║
║{" " * 70}║
║  {"To upgrade, run:":<68}║
║  {"pip install -U ultralytics>=" + min_version:<68}║
║{" " * 70}║
╚{border}╝
"""
    return message
