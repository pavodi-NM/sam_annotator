"""
SAM Annotator - A tool for image annotation using Segment Anything Model (SAM)
"""

import warnings

# Suppress FutureWarning from segment_anything's torch.load
# This is a temporary fix until Meta updates segment-anything library
# Issue: segment_anything uses torch.load without weights_only parameter
# Reference: https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="segment_anything.*",
    message=".*torch.load.*weights_only.*"
)

# Dynamic version reading from package metadata
import importlib.metadata

try:
    # Read version dynamically from package metadata
    # This will use the version from pyproject.toml after installation
    __version__ = importlib.metadata.version("sam_annotator")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development/editable installs
    # Read directly from pyproject.toml
    from pathlib import Path
    import re

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            __version__ = match.group(1)
        else:
            __version__ = "unknown"
    else:
        __version__ = "unknown"
