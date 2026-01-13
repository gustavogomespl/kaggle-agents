"""
Domain Detection Module - Backward Compatibility Shim.

This file provides backward compatibility for imports from the old location.
The implementation has been split into modular files in the `detection` subpackage.

Use:
    from kaggle_agents.domain.detector import DomainDetector
    # or
    from kaggle_agents.domain.detection import DomainDetector
"""

from __future__ import annotations

# Re-export all public interfaces from the new modular package
from .detection import (
    # Constants
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
    # Dataclass
    DetectionSignal,
    # Main class
    DomainDetector,
)


# For backward compatibility, also expose the original constants from the class
# This was how they were accessed before: DomainDetector.DOMAINS, etc.

# ==================== Convenience Functions ====================


def detect_competition_domain(
    competition_info,
    data_directory,
    llm=None,
    state=None,
):
    """
    Convenience function to detect competition domain using multi-signal fusion.

    Args:
        competition_info: Competition metadata
        data_directory: Path to competition data
        llm: Optional LLM client
        state: Optional workflow state for SOTA tags extraction

    Returns:
        Tuple of (domain_type, confidence)
    """
    detector = DomainDetector(llm=llm)
    return detector.detect(competition_info, data_directory, state=state)


def detect_submission_format(
    sample_submission_path,
    test_dir=None,
    competition_info=None,
):
    """
    Convenience function to detect submission format.

    Critical for distinguishing between:
    - Standard format (one row per sample) - most competitions
    - Pixel-level format (one row per pixel) - image-to-image, segmentation

    Args:
        sample_submission_path: Path to sample_submission.csv
        test_dir: Optional path to test data directory
        competition_info: Optional competition metadata

    Returns:
        Tuple of (format_type, metadata)
    """
    detector = DomainDetector()
    return detector.detect_submission_format(sample_submission_path, test_dir, competition_info)


__all__ = [
    # Main class
    "DomainDetector",
    "DetectionSignal",
    # Convenience functions
    "detect_competition_domain",
    "detect_submission_format",
    # Constants (for backward compatibility)
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
