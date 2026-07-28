"""
Data type detection for MLE-bench competitions.

Contains methods for detecting data type, target column, and ID column.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ...core.config import get_run_seed
from ...utils.label_parser import infer_filename_label_table


class DetectionMixin:
    """Mixin providing data type detection methods."""

    def detect_data_type(self, public_dir: Path) -> str:
        """
        Detect the type of data in the competition.

        Returns:
            'tabular', 'image', 'audio', or 'text'
        """
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}

        def _dir_contains_ext(dir_path: Path, exts: set[str], limit: int = 200) -> bool:
            seen = 0
            for p in dir_path.rglob("*"):
                if not p.is_file():
                    continue
                seen += 1
                if p.suffix.lower() in exts:
                    return True
                if seen >= limit:
                    break
            return False

        # 1) Check common directories, then every supplied top-level directory.
        checked_dirs = set()
        patterns = [
            "train",
            "test",
            "images",
            "train_images",
            "test_images",
            "*",
        ]
        for pattern in patterns:
            for dir_path in public_dir.glob(pattern):
                if not dir_path.is_dir() or dir_path in checked_dirs:
                    continue
                checked_dirs.add(dir_path)
                if _dir_contains_ext(dir_path, image_exts):
                    return "image"
                if _dir_contains_ext(dir_path, audio_exts):
                    return "audio"

        # 2) Check root for obvious media files (some zips extract flat)
        for p in list(public_dir.glob("*"))[:500]:
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext in image_exts:
                return "image"
            if ext in audio_exts:
                return "audio"

        # 3) Peek inside zips as a fallback (fast, no extraction assumptions)
        for zip_file in public_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(zip_file, "r") as z:
                    # Only inspect a prefix to keep this cheap
                    for name in z.namelist()[:5000]:
                        lower = name.lower()
                        if any(lower.endswith(ext) for ext in image_exts):
                            return "image"
                        if any(lower.endswith(ext) for ext in audio_exts):
                            return "audio"
            except Exception:
                continue

        # Check for text-heavy CSVs
        for csv_file in public_dir.glob("*.csv"):
            if "train" in csv_file.name.lower():
                try:
                    df = pd.read_csv(csv_file, nrows=5)
                    # Check for text columns (long strings)
                    for col in df.columns:
                        if df[col].dtype == "object":
                            avg_len = df[col].astype(str).str.len().mean()
                            if avg_len > 100:  # Long text
                                return "text"
                except Exception:
                    pass

        return "tabular"

    def _detect_target_column(self, sample_sub_path: Path) -> str:
        """Detect target column from sample submission."""
        try:
            df = pd.read_csv(sample_sub_path, nrows=1)
            if len(df.columns) >= 2:
                return df.columns[1]
        except Exception:
            pass
        return "target"

    def _detect_target_columns(
        self,
        sample_sub_path: Path,
        test_csv_path: Path | None = None,
    ) -> list[str]:
        """Read ordered prediction columns without inferring their semantics."""
        from kaggle_agents.utils.target_inference import (
            _read_schema_columns,
            split_submission_schema,
        )

        try:
            columns = [
                str(column)
                for column in pd.read_csv(sample_sub_path, nrows=0).columns
            ]
            if len(columns) >= 2:
                _, predicted = split_submission_schema(
                    columns,
                    _read_schema_columns(test_csv_path),
                )
                return predicted
        except Exception:
            pass
        return ["target"]

    def _detect_id_column(
        self,
        sample_sub_path: Path,
        test_csv_path: Path | None = None,
    ) -> str:
        """Detect ID column from sample submission."""
        from kaggle_agents.utils.target_inference import (
            _read_schema_columns,
            split_submission_schema,
        )

        try:
            columns = [
                str(column)
                for column in pd.read_csv(sample_sub_path, nrows=0).columns
            ]
            if len(columns) >= 2:
                echoed, _ = split_submission_schema(
                    columns,
                    _read_schema_columns(test_csv_path),
                )
                if echoed:
                    return echoed[0]
            if columns:
                return columns[0]
        except Exception:
            pass
        return "id"

    def _detect_audio_labels_from_filenames(
        self,
        audio_dir: Path,
        explicit_pattern: str | None = None,
    ) -> tuple[list[str], list[str], list[Path]]:
        """Extract labels only from explicit or uniquely inferred structure.

        Args:
            audio_dir: Directory containing audio files
            explicit_pattern: Dataset-derived regex with one target capture group

        Returns:
            Tuple of (ids, labels, paths) where:
            - ids: List of file stems
            - labels: Target values extracted from filenames
            - paths: List of file paths
        """
        audio_exts = {
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
            ".aac",
            ".wma",
            ".aiff",
            ".aif",
        }
        audio_files = [
            path
            for path in audio_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in audio_exts
        ]
        label_table = infer_filename_label_table(
            audio_files,
            explicit_pattern=explicit_pattern,
        )
        return (
            label_table["record_id"].tolist(),
            label_table["target"].tolist(),
            [Path(path) for path in label_table["file_path"]],
        )

    def create_canonical_from_audio_filenames(
        self,
        audio_dir: Path,
        canonical_dir: Path,
        n_folds: int = 5,
        explicit_pattern: str | None = None,
    ) -> dict:
        """Create canonical artifacts from evidence-backed filename targets.

        This is a fallback when no train.csv exists.

        Args:
            audio_dir: Directory containing audio files with labels in filenames
            canonical_dir: Directory to save canonical artifacts
            n_folds: Number of CV folds to create
            explicit_pattern: Dataset-derived regex with one target capture group

        Returns:
            Dictionary with canonical data info
        """
        from sklearn.model_selection import StratifiedKFold

        try:
            ids, labels, _paths = self._detect_audio_labels_from_filenames(
                audio_dir,
                explicit_pattern=explicit_pattern,
            )
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        if not ids:
            return {
                "success": False,
                "error": "No evidence-backed filename targets were found",
            }

        # Create canonical directory
        canonical_dir.mkdir(parents=True, exist_ok=True)

        # Convert to numpy arrays
        train_ids = np.array(ids)
        y = np.array(labels)

        # Create as many stratified folds as every observed class supports.
        class_counts = pd.Series(y).value_counts()
        effective_n_folds = min(n_folds, int(class_counts.min()))
        if effective_n_folds < 2:
            return {
                "success": False,
                "error": "At least two samples per inferred target are required for CV",
            }
        run_seed = get_run_seed()
        skf = StratifiedKFold(
            n_splits=effective_n_folds,
            shuffle=True,
            random_state=run_seed,
        )
        folds = np.zeros(len(train_ids), dtype=int)
        for fold_idx, (_, val_idx) in enumerate(skf.split(train_ids, y)):
            folds[val_idx] = fold_idx

        # Save artifacts. IDs stay str dtype (not object) so candidate code
        # can re-save them with allow_pickle=False.
        if train_ids.dtype == object:
            train_ids = np.asarray([str(v) for v in train_ids])
        np.save(canonical_dir / "train_ids.npy", train_ids, allow_pickle=False)
        np.save(canonical_dir / "y.npy", y)
        np.save(canonical_dir / "folds.npy", folds)

        # Save metadata
        import json

        metadata = {
            "canonical_rows": len(train_ids),
            "n_folds": effective_n_folds,
            "requested_n_folds": n_folds,
            "id_col": "record_id",
            "target_col": "target",
            "is_classification": True,
            "num_classes": len(np.unique(y)),
            "random_seed": run_seed,
            "target_source": (
                "explicit_filename_pattern"
                if explicit_pattern
                else "unique_filename_structure"
            ),
            "source": "audio_filenames",
        }
        with open(canonical_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"   [FALLBACK] Created canonical data from {len(ids)} audio files")
        print(f"   Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

        return {
            "success": True,
            "canonical_dir": str(canonical_dir),
            "train_ids_path": str(canonical_dir / "train_ids.npy"),
            "y_path": str(canonical_dir / "y.npy"),
            "folds_path": str(canonical_dir / "folds.npy"),
            "metadata": metadata,
        }
