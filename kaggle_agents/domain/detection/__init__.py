"""
Domain Detection module.

Provides domain detection capabilities for Kaggle competitions.
"""

from .constants import (
    AUDIO_EXTS,
    BBOX_COLUMNS,
    CLASSIFICATION_METRICS,
    CLEAN_DIR_PATTERNS,
    DOMAIN_DESCRIPTIONS,
    DOMAINS,
    FORCE_TYPE_TO_DOMAIN,
    IMAGE_EXTS,
    REGRESSION_METRICS,
    SOURCE_WEIGHTS,
    TABULAR_EXTS,
    TEXT_EXTS,
    TIME_COLUMNS,
    TIME_TARGETS,
)
from .dataclasses import DetectionSignal
from .detector import DomainDetector


__all__ = [
    # Main class
    "DomainDetector",
    "DetectionSignal",
    # Constants
    "DOMAINS",
    "DOMAIN_DESCRIPTIONS",
    "IMAGE_EXTS",
    "AUDIO_EXTS",
    "TEXT_EXTS",
    "TABULAR_EXTS",
    "CLEAN_DIR_PATTERNS",
    "TIME_COLUMNS",
    "TIME_TARGETS",
    "BBOX_COLUMNS",
    "REGRESSION_METRICS",
    "CLASSIFICATION_METRICS",
    "SOURCE_WEIGHTS",
    "FORCE_TYPE_TO_DOMAIN",
]
