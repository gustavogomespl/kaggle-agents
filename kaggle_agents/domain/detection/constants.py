"""
Constants for domain detection.
"""

# Image extensions for format detection
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif"}

# Audio extensions
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}

# Text extensions
TEXT_EXTS = {".txt", ".json"}

# Tabular extensions
TABULAR_EXTS = {".csv", ".parquet"}

# Expanded patterns for image-to-image detection
CLEAN_DIR_PATTERNS = [
    # Existing patterns
    "train_cleaned", "train_clean", "clean", "cleaned",
    "gt", "ground_truth", "train_gt", "target", "targets", "train_target",
    # New patterns for denoising, super-resolution, etc.
    "hr", "high_res", "high_resolution", "hq", "high_quality",
    "output", "outputs", "label", "labels",
    "denoised", "restored", "enhanced", "original",
]

# Columns indicating time series data
TIME_COLUMNS = ["date", "datetime", "timestamp", "time", "day", "week", "month", "year"]
TIME_TARGETS = ["sales", "demand", "volume", "price", "count", "visitors", "quantity"]

# Columns indicating object detection
BBOX_COLUMNS = ["x", "y", "width", "height", "x_min", "y_min", "x_max", "y_max", "bbox", "confidence"]

# Regression metrics
REGRESSION_METRICS = ["rmse", "mae", "mse", "mape", "r2", "rmsle", "medae"]

# Classification metrics
CLASSIFICATION_METRICS = ["logloss", "auc", "accuracy", "f1", "precision", "recall", "map@k", "mrr"]

# Source weights for signal fusion
SOURCE_WEIGHTS = {
    "structural": 1.0,    # Structural heuristics - most reliable
    "submission": 0.9,    # Submission format analysis
    "metric": 0.85,       # Evaluation metric signal
    "llm": 0.9,           # LLM inference (increased from 0.7 to prioritize semantic detection for seq_to_seq)
    "sota": 0.6,          # SOTA tags from notebooks
    "fallback": 0.5,      # Fallback generic
}

# Force domain mapping from environment variables
FORCE_TYPE_TO_DOMAIN = {
    "seq2seq": "seq_to_seq",
    "seq_to_seq": "seq_to_seq",
    "text_normalization": "seq_to_seq",
    "text-normalization": "seq_to_seq",
    "image": "image_classification",
    "image_classification": "image_classification",
    "image-classification": "image_classification",
    "image_to_image": "image_to_image",
    "audio": "audio_classification",
    "audio_classification": "audio_classification",
    "audio_tagging": "audio_tagging",
    "text": "text_classification",
    "text_classification": "text_classification",
    "nlp": "text_classification",
    "tabular": "tabular_classification",
    "tabular_classification": "tabular_classification",
    "tabular_regression": "tabular_regression",
    "regression": "tabular_regression",
}

# Valid domain types
DOMAINS = [
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
]

# Domain descriptions
DOMAIN_DESCRIPTIONS = {
    "image_classification": "Classify images into discrete categories",
    "image_regression": "Predict continuous values from images",
    "image_to_image": "Transform images (denoising, super-resolution, style transfer) or pixel-level (one row per pixel)",
    "image_segmentation": "Pixel-wise classification of images",
    "object_detection": "Locate and classify objects in images",
    "text_classification": "Classify text into categories",
    "seq_to_seq": "Sequence to sequence transformation (translation, normalization)",
    "text_regression": "Predict continuous values from text",
    "audio_classification": "Classify audio signals",
    "audio_regression": "Predict continuous values from audio",
    "tabular_classification": "Classify rows in structured tabular data",
    "tabular_regression": "Predict continuous values from tabular data",
    "time_series_forecasting": "Predict future values from temporal data",
    "multi_modal": "Combination of multiple data types",
}
