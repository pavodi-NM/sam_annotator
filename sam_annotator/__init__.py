"""
SAM Annotator - A tool for image annotation using Segment Anything Model (SAM)
"""

from sam_annotator.core.annotator import SAMAnnotator
from sam_annotator.core.weight_manager import SAMWeightManager
from sam_annotator.core.predictor import SAM1Predictor, SAM2Predictor

__version__ = '0.1.0.dev5'  # Development version

__all__ = [
    'SAMAnnotator',
    'SAMWeightManager',
    'SAM1Predictor',
    'SAM2Predictor',
] 