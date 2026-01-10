"""
Domain and submission format type definitions.
"""

from typing import Literal


# ==================== Domain Types ====================

DomainType = Literal[
    # Image-based
    "image_classification",
    "image_regression",
    "image_to_image",
    "image_segmentation",
    "object_detection",
    # Text-based
    "text_classification",
    "seq_to_seq",
    "text_regression",
    # Audio-based
    "audio_classification",
    "audio_regression",
    # Tabular
    "tabular_classification",
    "tabular_regression",
    # Time series
    "time_series_forecasting",
    # Multi-modal
    "multi_modal",
    # Legacy (for backwards compatibility)
    "tabular",
    "computer_vision",
    "nlp",
    "time_series",
]


# ==================== Submission Format Types ====================

SubmissionFormatType = Literal[
    "standard",  # One row per sample (classification/regression)
    "pixel_level",  # One row per pixel (image-to-image, segmentation)
    "multi_label",  # Multiple rows per sample
    "ranking",  # Ranking format
    "rle_encoded",  # Run-length encoded masks (segmentation)
]


# ==================== Audio Submission Format Types ====================

AudioSubmissionFormatType = Literal[
    "wide",  # One row per sample, one column per class (BirdCLEF style)
    "long",  # One row per (sample, class) pair (MLSP style)
    "pixel_level",  # Image-based audio (spectrograms)
    "unknown",  # Could not detect format
]


# ID pattern types for long format submissions
AudioIdPatternType = Literal[
    "multiplier",  # Id = rec_id * N + class_id (e.g., MLSP: N=100)
    "underscore",  # Id = "rec_id_class_id" (e.g., "1_0", "1_1")
    "dash",  # Id = "rec_id-class_id" (e.g., "1-0", "1-1")
    "unknown",  # Could not detect pattern
]
