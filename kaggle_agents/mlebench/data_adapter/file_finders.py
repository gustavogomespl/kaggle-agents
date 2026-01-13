"""
File finding utilities for MLE-bench data adapter.

Contains methods for finding CSV files, label files, audio sources, etc.
"""

from __future__ import annotations

from pathlib import Path


class FileFinderMixin:
    """Mixin providing file finding methods."""

    def _find_csv_file(self, directory: Path, patterns: list[str]) -> Path | None:
        """Find a CSV file matching any of the patterns.

        Handles edge case where a match is actually a directory containing the CSV file.
        """
        for pattern in patterns:
            matches = list(directory.glob(pattern))
            for match in matches:
                if match.is_file():
                    return match
                # Handle directory case: look for CSV file inside
                if match.is_dir():
                    inner_csvs = sorted(match.glob("*.csv"))
                    if inner_csvs:
                        print(
                            f"      📂 Resolved directory '{match.name}' to file: {inner_csvs[0].name}",
                            flush=True,
                        )
                        return inner_csvs[0]
        return None

    def _find_label_files(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> list[Path]:
        """
        Find label files in directory (CSV and TXT formats).

        This handles non-standard formats like MLSP 2013 Birds which uses:
        - rec_labels_test_hidden.txt (multi-label training labels)
        - rec_id2filename.txt (maps rec_id -> audio filename)
        - CVfolds_2.txt (cross-validation fold assignments)

        Args:
            directory: Directory to search in
            recursive: Whether to search recursively

        Returns:
            List of label file paths found
        """
        label_patterns = [
            # Standard CSV patterns
            "**/train_labels.csv",
            "**/labels.csv",
            "**/train.csv",
            # Non-standard TXT patterns (MLSP 2013 Birds, etc.)
            "**/rec_labels*.txt",
            "**/*_labels*.txt",
            "**/labels*.txt",
            "**/CVfolds*.txt",
            "**/rec_id2filename*.txt",
            "**/*2filename*.txt",
            # Additional patterns for other non-standard formats
            "**/train_metadata*.txt",
            "**/metadata*.txt",
        ]

        if not recursive:
            # Convert to non-recursive patterns
            label_patterns = [p.replace("**/", "") for p in label_patterns]

        found_files = []
        for pattern in label_patterns:
            try:
                matches = list(directory.glob(pattern))
                for match in matches:
                    if match.is_file() and match not in found_files:
                        found_files.append(match)
            except Exception:
                continue

        return found_files

    def _find_audio_source_dir(self, directory: Path) -> Path | None:
        """
        Find the directory containing source audio files.

        Handles non-standard structures like MLSP 2013 Birds where audio is in:
        - essential_data/src_wavs/

        Args:
            directory: Parent directory to search in

        Returns:
            Path to audio source directory, or None
        """
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

        # Common audio source directory patterns
        audio_dir_patterns = [
            "src_wavs",
            "wavs",
            "audio",
            "audio_files",
            "raw_audio",
            "train_audio",
        ]

        # First check direct subdirectories
        for subdir_name in audio_dir_patterns:
            subdir = directory / subdir_name
            if subdir.is_dir():
                # Verify it contains audio files
                sample_files = list(subdir.glob("*"))[:20]
                if any(f.suffix.lower() in audio_exts for f in sample_files if f.is_file()):
                    return subdir

        # Then recursively search for directories with audio files
        for subdir in directory.rglob("*"):
            if not subdir.is_dir():
                continue
            # Check if this directory contains audio files
            sample_files = list(subdir.glob("*"))[:20]
            if any(f.suffix.lower() in audio_exts for f in sample_files if f.is_file()):
                return subdir

        return None

    def _find_first_zip(self, directory: Path, kind: str) -> Path | None:
        """Find a likely train/test ZIP in a directory."""
        kind_norm = kind.strip().lower()
        if kind_norm not in {"train", "test"}:
            raise ValueError(f"kind must be 'train' or 'test', got: {kind}")

        patterns = [
            f"{kind_norm}.zip",
            f"{kind_norm}_images.zip",
            f"{kind_norm}_imgs.zip",
            f"{kind_norm}*.zip",
            f"*{kind_norm}*.zip",
        ]
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _find_data_in_subdirs(
        self,
        parent_dir: Path,
        patterns: list[str],
        exclude_dirs: set[str] | None = None,
    ) -> Path | None:
        """
        Search for data files/dirs in subdirectories (generic fallback).

        Args:
            parent_dir: Directory to search in
            patterns: List of file/dir names to look for (e.g., ["train.csv", "train"])
            exclude_dirs: Directory names to skip

        Returns:
            First matching Path found, or None
        """
        if exclude_dirs is None:
            exclude_dirs = {"models", "__pycache__", ".git", ".ipynb_checkpoints"}

        for subdir in sorted(parent_dir.iterdir()):
            if not subdir.is_dir() or subdir.name in exclude_dirs:
                continue

            # Check each pattern in this subdirectory
            for pattern in patterns:
                candidate = subdir / pattern
                if candidate.exists():
                    return candidate

            # If subdir itself contains data files (wav, csv, txt, png, etc.), return it
            # Added .txt for non-standard label formats (MLSP 2013 Birds)
            data_extensions = {
                ".csv",
                ".txt",  # Label files
                ".wav",
                ".mp3",
                ".flac",
                ".ogg",  # Audio
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",  # Images
                ".npy",  # Arrays
            }
            try:
                sample_files = list(subdir.glob("*"))[:20]
                if any(f.suffix.lower() in data_extensions for f in sample_files if f.is_file()):
                    return subdir
            except PermissionError:
                continue

        return None
