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
        Find likely label and split-metadata files from semantic filename hints.

        Args:
            directory: Directory to search in
            recursive: Whether to search recursively

        Returns:
            List of label file paths found
        """
        semantic_hints = (
            "train",
            "label",
            "target",
            "annotation",
            "fold",
            "mapping",
            "filename",
            "metadata",
        )
        iterator = directory.rglob("*") if recursive else directory.glob("*")
        found_files: list[Path] = []
        try:
            for match in iterator:
                if not match.is_file() or match.suffix.lower() not in {".csv", ".txt", ".tsv"}:
                    continue
                normalized_name = match.stem.lower().replace("-", "_")
                if normalized_name.startswith("sample"):
                    continue
                if any(hint in normalized_name for hint in semantic_hints):
                    found_files.append(match)
        except (OSError, PermissionError):
            return []

        return sorted(set(found_files))

    def _find_audio_source_dir(self, directory: Path) -> Path | None:
        """
        Find the directory containing the strongest local concentration of audio.

        Args:
            directory: Parent directory to search in

        Returns:
            Path to audio source directory, or None
        """
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}

        best_dir: Path | None = None
        best_count = 0
        candidate_dirs = [directory]
        try:
            candidate_dirs.extend(path for path in directory.rglob("*") if path.is_dir())
        except (OSError, PermissionError):
            pass

        for candidate in candidate_dirs:
            try:
                audio_count = sum(
                    1
                    for index, path in enumerate(candidate.iterdir())
                    if index < 500 and path.is_file() and path.suffix.lower() in audio_exts
                )
            except (OSError, PermissionError):
                continue
            if audio_count > best_count:
                best_count = audio_count
                best_dir = candidate

        return best_dir

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

            # If the subdirectory itself contains data files, return it.
            data_extensions = {
                ".csv",
                ".txt",  # Label files
                ".wav",
                ".mp3",
                ".flac",
                ".ogg",
                ".aiff",
                ".aif",  # Audio
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
