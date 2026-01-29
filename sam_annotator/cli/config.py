"""Configuration management for SAM Annotator.

This module handles persistent configuration storage, including:
- Last used paths (category_path, classes_csv)
- Last used SAM version
- Model types per SAM version (version-specific storage)
"""

import json
import os
from typing import Dict, Any, Optional

from sam_annotator.core.model_registry import (
    get_default_model_type,
    get_supported_versions,
    is_valid_model_type,
)

# Constants for config file
CONFIG_FILE = ".sam_config.json"

# Default configuration structure
DEFAULT_CONFIG = {
    "last_category_path": None,
    "last_classes_csv": None,
    "last_sam_version": "sam2",
    "model_types": {
        "sam1": "vit_b",
        "sam2": "small_v2",
        "sam3": "sam3"
    }
}


def load_config() -> Dict[str, Any]:
    """Load configuration from config file if it exists.

    Returns:
        Dictionary containing configuration values with defaults applied
    """
    config = DEFAULT_CONFIG.copy()
    config["model_types"] = DEFAULT_CONFIG["model_types"].copy()

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)

                # Migrate old config format if needed
                loaded_config = _migrate_config(loaded_config)

                # Update config with loaded values
                for key, value in loaded_config.items():
                    if key == "model_types" and isinstance(value, dict):
                        config["model_types"].update(value)
                    else:
                        config[key] = value

        except Exception as e:
            print(f"Warning: Could not load config file: {e}")

    return config


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to config file.

    Args:
        config: Dictionary containing configuration values to save
    """
    try:
        # Load existing config to preserve other values
        existing = load_config()
        existing.update(config)

        with open(CONFIG_FILE, 'w') as f:
            json.dump(existing, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save config file: {e}")


def get_model_type_for_version(config: Dict[str, Any], sam_version: str) -> Optional[str]:
    """Get the stored model type for a specific SAM version.

    Args:
        config: Configuration dictionary
        sam_version: The SAM version (e.g., 'sam1', 'sam2', 'sam3')

    Returns:
        The stored model type, or None if not found
    """
    model_types = config.get("model_types", {})
    stored_type = model_types.get(sam_version)

    # Validate that the stored type is still valid for this version
    if stored_type and is_valid_model_type(sam_version, stored_type):
        return stored_type

    # Fall back to default if stored type is invalid
    return get_default_model_type(sam_version)


def set_model_type_for_version(
    config: Dict[str, Any],
    sam_version: str,
    model_type: str
) -> Dict[str, Any]:
    """Set the model type for a specific SAM version.

    Args:
        config: Configuration dictionary
        sam_version: The SAM version (e.g., 'sam1', 'sam2', 'sam3')
        model_type: The model type to store

    Returns:
        Updated configuration dictionary
    """
    if "model_types" not in config:
        config["model_types"] = {}

    # Only store if it's a valid model type for this version
    if is_valid_model_type(sam_version, model_type):
        config["model_types"][sam_version] = model_type

    return config


def _migrate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate old config format to new format.

    Old format had a single 'last_model_type' field.
    New format has 'model_types' dict with per-version storage.

    Args:
        config: Configuration dictionary (possibly old format)

    Returns:
        Configuration dictionary in new format
    """
    # Check if this is old format (has 'last_model_type' but no 'model_types')
    if "last_model_type" in config and "model_types" not in config:
        old_model_type = config.pop("last_model_type")
        old_sam_version = config.get("last_sam_version", "sam2")

        # Initialize model_types with defaults
        config["model_types"] = DEFAULT_CONFIG["model_types"].copy()

        # If the old model type is valid for the old version, preserve it
        if old_model_type and is_valid_model_type(old_sam_version, old_model_type):
            config["model_types"][old_sam_version] = old_model_type

    # Handle case where 'last_model_type' exists alongside 'model_types' (cleanup)
    if "last_model_type" in config:
        del config["last_model_type"]

    return config


def reset_config() -> None:
    """Reset configuration to defaults by deleting the config file."""
    try:
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
    except Exception as e:
        print(f"Warning: Could not reset config file: {e}")


def get_config_path() -> str:
    """Get the path to the configuration file.

    Returns:
        Absolute path to the config file
    """
    return os.path.abspath(CONFIG_FILE)
