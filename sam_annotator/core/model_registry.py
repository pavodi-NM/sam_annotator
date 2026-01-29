"""Centralized model registry for SAM versions and model types.

This module provides a single source of truth for:
- Valid model types per SAM version
- Default model types per SAM version
- Model type validation
- Model type resolution with proper precedence
"""

from typing import Dict, List, Optional, Tuple


# Centralized registry of all SAM versions and their model types
MODEL_REGISTRY: Dict[str, Dict] = {
    "sam1": {
        "valid_types": ["vit_h", "vit_l", "vit_b"],
        "default": "vit_b",
        "auto_download": True,
        "description": "Meta's original SAM model"
    },
    "sam2": {
        "valid_types": [
            "tiny", "small", "base", "large",
            "tiny_v2", "small_v2", "base_v2", "large_v2"
        ],
        "default": "small_v2",
        "auto_download": True,
        "description": "SAM2 via Ultralytics"
    },
    "sam3": {
        "valid_types": ["sam3"],
        "default": "sam3",
        "auto_download": False,
        "description": "SAM3 with concept segmentation (requires manual download)",
        "requirements": {
            "min_ultralytics_version": "8.3.237",
            "min_gpu_memory_gb": 8,
            "min_system_ram_gb": 16,
            "weight_size_gb": 3.4,
            "download_url": "https://huggingface.co/facebook/sam3",
            "weight_filename": "sam3.pt"
        }
    }
}


def get_supported_versions() -> List[str]:
    """Get list of all supported SAM versions.

    Returns:
        List of version strings (e.g., ['sam1', 'sam2', 'sam3'])
    """
    return list(MODEL_REGISTRY.keys())


def get_valid_model_types(sam_version: str) -> List[str]:
    """Get valid model types for a SAM version.

    Args:
        sam_version: The SAM version (e.g., 'sam1', 'sam2', 'sam3')

    Returns:
        List of valid model type strings
    """
    if sam_version not in MODEL_REGISTRY:
        return []
    return MODEL_REGISTRY[sam_version].get("valid_types", [])


def get_default_model_type(sam_version: str) -> Optional[str]:
    """Get default model type for a SAM version.

    Args:
        sam_version: The SAM version (e.g., 'sam1', 'sam2', 'sam3')

    Returns:
        Default model type string, or None if version not found
    """
    if sam_version not in MODEL_REGISTRY:
        return None
    return MODEL_REGISTRY[sam_version].get("default")


def is_valid_model_type(sam_version: str, model_type: str) -> bool:
    """Check if model_type is valid for the given sam_version.

    Args:
        sam_version: The SAM version (e.g., 'sam1', 'sam2', 'sam3')
        model_type: The model type to validate

    Returns:
        True if valid, False otherwise
    """
    valid_types = get_valid_model_types(sam_version)
    return model_type in valid_types


def supports_auto_download(sam_version: str) -> bool:
    """Check if a SAM version supports automatic weight download.

    Args:
        sam_version: The SAM version (e.g., 'sam1', 'sam2', 'sam3')

    Returns:
        True if auto-download is supported, False otherwise
    """
    if sam_version not in MODEL_REGISTRY:
        return False
    return MODEL_REGISTRY[sam_version].get("auto_download", False)


def get_version_requirements(sam_version: str) -> Optional[Dict]:
    """Get special requirements for a SAM version (e.g., SAM3).

    Args:
        sam_version: The SAM version

    Returns:
        Dictionary of requirements, or None if no special requirements
    """
    if sam_version not in MODEL_REGISTRY:
        return None
    return MODEL_REGISTRY[sam_version].get("requirements")


def resolve_model_type(
    sam_version: str,
    cli_model_type: Optional[str],
    config_model_type: Optional[str]
) -> Tuple[str, str]:
    """Resolve which model_type to use with proper precedence.

    Precedence order:
    1. CLI explicit argument (if valid for this version)
    2. Config default for this version (if valid)
    3. Hardcoded default for this version

    Args:
        sam_version: The SAM version being used
        cli_model_type: Model type from command line (may be None)
        config_model_type: Model type from config for this version (may be None)

    Returns:
        Tuple of (resolved_model_type, source) where source is one of:
        'cli', 'config', 'default'
    """
    valid_types = get_valid_model_types(sam_version)
    default_type = get_default_model_type(sam_version)

    # 1. CLI explicit argument (if valid)
    if cli_model_type and cli_model_type in valid_types:
        return cli_model_type, "cli"

    # 2. Config default for this version (if valid)
    if config_model_type and config_model_type in valid_types:
        return config_model_type, "config"

    # 3. Hardcoded default
    return default_type, "default"


def validate_version_and_model(
    sam_version: str,
    model_type: str
) -> Tuple[bool, Optional[str]]:
    """Validate that sam_version and model_type are compatible.

    Args:
        sam_version: The SAM version
        model_type: The model type

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    # Check version is supported
    if sam_version not in MODEL_REGISTRY:
        supported = ", ".join(get_supported_versions())
        return False, f"Unsupported SAM version: '{sam_version}'. Supported versions: {supported}"

    # Check model type is valid for this version
    valid_types = get_valid_model_types(sam_version)
    if model_type not in valid_types:
        valid_str = ", ".join(valid_types)
        return False, f"Invalid model type '{model_type}' for {sam_version.upper()}. Valid types: {valid_str}"

    return True, None


def get_model_type_help_text() -> str:
    """Generate help text for model types across all versions.

    Returns:
        Formatted help string for CLI --model_type argument
    """
    lines = []
    for version, info in MODEL_REGISTRY.items():
        types_str = ", ".join(info["valid_types"])
        default = info["default"]
        lines.append(f"{version.upper()}: {types_str} (default: {default})")
    return " | ".join(lines)
