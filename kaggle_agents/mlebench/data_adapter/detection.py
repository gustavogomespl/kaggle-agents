"""
Data type detection for MLE-bench competitions.

Contains methods for detecting data type, target column, and ID column.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np


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

        # 1) Check common directories (recursively, to handle nested zips)
        # Use exact matches first, then numbered patterns (train2, test2) to avoid duplicates
        checked_dirs = set()
        # Exact matches first, then numbered patterns for non-standard naming
        patterns = [
            "train", "test", "images", "train_images", "test_images",  # Exact matches
            "train[0-9]*", "test[0-9]*",  # Numbered variants (train2, test2, etc.)
            "essential_data", "supplemental_data", "src_wavs",  # MLSP-style competitions
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

    def _detect_id_column(self, sample_sub_path: Path) -> str:
        """Detect ID column from sample submission."""
        try:
            df = pd.read_csv(sample_sub_path, nrows=1)
            if len(df.columns) >= 1:
                return df.columns[0]
        except Exception:
            pass
        return "id"

    def _detect_audio_labels_from_filenames(
        self, audio_dir: Path
    ) -> tuple[list[str], list[int], list[Path]]:
        """Extract labels from audio filenames when no train.csv exists.

        Supports patterns like:
        - train12345_1.aiff (label=1)
        - whale_001_0.wav (label=0)
        - file_42.mp3 (label=42)

        Args:
            audio_dir: Directory containing audio files

        Returns:
            Tuple of (ids, labels, paths) where:
            - ids: List of file stems
            - labels: List of integer labels extracted from filenames
            - paths: List of file paths
        """
        AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
        LABEL_PATTERNS = [
            r"_(\d+)\.[a-zA-Z0-9]+$",  # train123_1.aiff (most common)
            r"_(\d+)$",  # train123_1 (no extension in stem - rare)
        ]

        audio_files = []
        for f in audio_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                audio_files.append(f)

        ids: list[str] = []
        labels: list[int] = []
        paths: list[Path] = []

        for fp in audio_files:
            for pattern in LABEL_PATTERNS:
                match = re.search(pattern, fp.name)
                if match:
                    ids.append(fp.stem)
                    labels.append(int(match.group(1)))
                    paths.append(fp)
                    break

        return ids, labels, paths

    def create_canonical_from_audio_filenames(
        self, audio_dir: Path, canonical_dir: Path, n_folds: int = 5
    ) -> dict:
        """Create canonical data artifacts from audio files with labels in filenames.

        This is a fallback when no train.csv exists.

        Args:
            audio_dir: Directory containing audio files with labels in filenames
            canonical_dir: Directory to save canonical artifacts
            n_folds: Number of CV folds to create

        Returns:
            Dictionary with canonical data info
        """
        from sklearn.model_selection import StratifiedKFold

        ids, labels, paths = self._detect_audio_labels_from_filenames(audio_dir)

        if not ids:
            return {"success": False, "error": "No audio files with labels found"}

        # Create canonical directory
        canonical_dir.mkdir(parents=True, exist_ok=True)

        # Convert to numpy arrays
        train_ids = np.array(ids)
        y = np.array(labels)

        # Create stratified folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = np.zeros(len(train_ids), dtype=int)
        for fold_idx, (_, val_idx) in enumerate(skf.split(train_ids, y)):
            folds[val_idx] = fold_idx

        # Save artifacts
        np.save(canonical_dir / "train_ids.npy", train_ids)
        np.save(canonical_dir / "y.npy", y)
        np.save(canonical_dir / "folds.npy", folds)

        # Save metadata
        import json

        metadata = {
            "canonical_rows": len(train_ids),
            "n_folds": n_folds,
            "id_col": "id",
            "target_col": "target",
            "is_classification": True,
            "num_classes": len(np.unique(y)),
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
