"""
Data type detection for MLE-bench competitions.

Contains methods for detecting data type, target column, and ID column.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd


class DetectionMixin:
    """Mixin providing data type detection methods."""

    def detect_data_type(self, public_dir: Path) -> str:
        """
        Detect the type of data in the competition.

        Returns:
            'tabular', 'image', 'audio', or 'text'
        """
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
        audio_exts = {".wav", ".mp3", ".flac", ".ogg"}

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
        for dir_name in ["train", "test", "images", "train_images", "test_images"]:
            dir_path = public_dir / dir_name
            if not dir_path.is_dir():
                continue
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
