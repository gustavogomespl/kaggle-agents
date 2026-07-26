"""
MLE-bench Data Adapter Dataclasses.

Contains data structures for MLE-bench competition data information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MLEBenchDataInfo:
    """Information about MLE-bench competition data."""

    competition_id: str
    workspace: Path
    train_path: Path | None = None
    test_path: Path | None = None
    clean_train_path: Path | None = None
    sample_submission_path: Path | None = None
    train_csv_path: Path | None = None  # For image competitions with labels CSV
    test_csv_path: Path | None = None
    ground_truth_path: Path | None = None  # Private test labels
    description_path: Path | None = None
    data_type: str = "tabular"  # tabular, image, audio, text
    target_column: str = "target"
    target_columns: list[str] = field(default_factory=lambda: ["target"])
    id_column: str = "id"
    extra_files: list[Path] = field(default_factory=list)
    # Label and split-metadata files discovered from the public data.
    label_files: list[Path] = field(default_factory=list)
    # Audio source directory inferred from local file extensions.
    audio_source_path: Path | None = None
