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
