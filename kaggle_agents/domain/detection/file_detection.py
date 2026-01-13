"""
File and structure-based detection methods.

Contains methods for detecting domain based on file structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .constants import (
    AUDIO_EXTS,
    CLEAN_DIR_PATTERNS,
    IMAGE_EXTS,
    TABULAR_EXTS,
    TEXT_EXTS,
)


if TYPE_CHECKING:
    from ...core.state import CompetitionInfo, DomainType


class FileDetectionMixin:
    """Mixin providing file-based detection methods."""

    def _detect_from_structure(
        self, competition_info: CompetitionInfo, data_dir: Path
    ) -> tuple[DomainType, float]:
        """Heuristic detection from local files when no LLM is available."""
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        text_exts = {".txt", ".json"}
        tabular_exts = {".csv", ".parquet"}

        # =================================================================
        # CRITICAL: Check for audio files GLOBALLY first
        # Audio competitions often have spectrograms in train/test dirs
        # but actual audio files in essential_data/, supplemental_data/, etc.
        # If audio files exist ANYWHERE, this is likely an audio competition.
        # =================================================================
        has_audio, audio_count = self._has_audio_assets(data_dir, return_count=True)
        if has_audio and audio_count >= 10:
            # Strong audio signal - prioritize audio domain
            print(f"   🎵 Found {audio_count} audio files - audio competition detected")
            is_regression = "regression" in (competition_info.problem_type or "").lower()
            return ("audio_regression", 0.92) if is_regression else ("audio_classification", 0.92)

        def classify(counts: dict[str, int], total: int) -> tuple[DomainType, float] | None:
            if total == 0:
                return None

            image_ratio = sum(counts.get(ext, 0) for ext in image_exts) / total
            audio_ratio = sum(counts.get(ext, 0) for ext in audio_exts) / total
            text_ratio = sum(counts.get(ext, 0) for ext in text_exts) / total
            tabular_ratio = sum(counts.get(ext, 0) for ext in tabular_exts) / total

            # Check audio FIRST - audio competitions often contain spectrogram images
            # Also compare ratios to pick the dominant type when both exist
            if audio_ratio >= 0.3:
                # Audio takes priority if it's the dominant type or images are spectrograms
                if audio_ratio >= image_ratio:
                    return ("audio_classification", 0.90)
                # Even if images dominate, if audio is significant, prefer audio
                # (spectrograms are derived from audio, so the source data is audio)
                return ("audio_classification", 0.85)
            if image_ratio >= 0.3:
                # If we detected audio files globally but images dominate this dir,
                # still prefer audio (images might be spectrograms)
                if has_audio and audio_count >= 5:
                    return ("audio_classification", 0.85)
                return ("image_classification", 0.90)
            if text_ratio >= 0.3:
                # Use regression hint if problem_type mentions it
                if "regression" in (competition_info.problem_type or "").lower():
                    return ("text_regression", 0.80)
                return ("text_classification", 0.80)
            if tabular_ratio >= 0.5:
                if "regression" in (competition_info.problem_type or "").lower():
                    return ("tabular_regression", 0.80)
                return ("tabular_classification", 0.80)

            return None

        def analyze_dir(dir_path: Path) -> tuple[dict[str, int], int]:
            counts: dict[str, int] = {}
            total = 0
            for i, file_path in enumerate(dir_path.rglob("*")):
                if i >= 600:
                    break
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext:
                        counts[ext] = counts.get(ext, 0) + 1
                        total += 1
            return counts, total

        # Prefer train/test folders if present, but also check other data directories
        exclude_dirs = {"models", "__pycache__", ".git", "logs", ".ipynb_checkpoints"}

        if data_dir.exists():
            # First priority: train/test prefixed directories
            train_test_dirs = [
                p
                for p in data_dir.iterdir()
                if p.is_dir() and p.name.lower().startswith(("train", "test"))
            ]

            # Second priority: other data directories (essential_data, supplemental_data, data, etc.)
            other_data_dirs = [
                p
                for p in data_dir.iterdir()
                if p.is_dir()
                and p.name.lower() not in exclude_dirs
                and not p.name.lower().startswith(("train", "test"))
            ]

            candidate_dirs = train_test_dirs + other_data_dirs
        else:
            candidate_dirs = []

        # Image-to-image heuristic: look for paired train + clean/target directories
        # Skip this check if we detected audio files (spectrograms shouldn't trigger this)
        if data_dir.exists() and not has_audio:
            dir_map = {p.name.lower(): p for p in data_dir.iterdir() if p.is_dir()}
            # Use expanded CLEAN_DIR_PATTERNS constant
            train_dir = None
            for name in ("train", "training", "train_images"):
                if name in dir_map:
                    train_dir = dir_map[name]
                    break
            clean_dir = None
            for name in CLEAN_DIR_PATTERNS:
                if name in dir_map:
                    clean_dir = dir_map[name]
                    break

            if train_dir and clean_dir:
                train_counts, train_total = analyze_dir(train_dir)
                clean_counts, clean_total = analyze_dir(clean_dir)
                train_result = classify(train_counts, train_total)
                clean_result = classify(clean_counts, clean_total)
                if train_result and clean_result:
                    if train_result[0].startswith("image_") and clean_result[0].startswith(
                        "image_"
                    ):
                        return ("image_to_image", 0.92)

        for dir_path in candidate_dirs:
            counts, total = analyze_dir(dir_path)
            result = classify(counts, total)
            if result:
                return result

        # Fall back to scanning the whole directory tree
        if data_dir.exists():
            counts, total = analyze_dir(data_dir)
            result = classify(counts, total)
            if result:
                return result

        # Default tabular guess
        if "regression" in (competition_info.problem_type or "").lower():
            return ("tabular_regression", 0.50)
        return ("tabular_classification", 0.50)

    def _detect_data_type(self, data_dir: Path) -> str:
        """
        Detect the primary data type from file structure.

        Returns one of: "image", "audio", "text", "tabular"

        Note: Audio takes priority over images because audio competitions often
        contain spectrogram images, but the source data is audio.
        """
        if not data_dir.exists():
            return "tabular"

        counts: dict[str, int] = {"image": 0, "audio": 0, "text": 0, "tabular": 0}

        # Check direct files and subdirectories
        exclude_dirs = {"models", "__pycache__", ".git", "logs", ".ipynb_checkpoints"}

        for path in data_dir.iterdir():
            if path.is_file():
                ext = path.suffix.lower()
                if ext in IMAGE_EXTS:
                    counts["image"] += 1
                elif ext in AUDIO_EXTS:
                    counts["audio"] += 1
                elif ext in TEXT_EXTS:
                    counts["text"] += 1
                elif ext in TABULAR_EXTS:
                    counts["tabular"] += 1
            elif path.is_dir() and path.name.lower() not in exclude_dirs:
                # Sample first 200 files in subdirectory (increased for better coverage)
                for i, subfile in enumerate(path.rglob("*")):
                    if i >= 200:
                        break
                    if subfile.is_file():
                        ext = subfile.suffix.lower()
                        if ext in IMAGE_EXTS:
                            counts["image"] += 10  # Weight directories higher
                        elif ext in AUDIO_EXTS:
                            counts["audio"] += 10
                        elif ext in TEXT_EXTS:
                            counts["text"] += 10

        # CRITICAL: Audio takes priority over images
        # Audio competitions often have spectrograms in train/test but actual
        # audio files in essential_data/, supplemental_data/, etc.
        # If ANY audio files found, prioritize audio detection
        if counts["audio"] > 0:
            return "audio"
        if counts["image"] > max(counts["text"], counts["tabular"]):
            return "image"
        if counts["text"] > counts["tabular"]:
            return "text"
        return "tabular"

    def _find_train_file(self, data_dir: Path) -> Path | None:
        """Find a train.csv/train.parquet file in the data directory or one level below."""
        candidates = [data_dir / "train.csv", data_dir / "train.parquet"]
        for path in candidates:
            if path.exists():
                return path

        if data_dir.exists():
            for child in data_dir.iterdir():
                if not child.is_dir():
                    continue
                for name in ("train.csv", "train.parquet"):
                    path = child / name
                    if path.exists():
                        return path
        return None

    def _has_image_assets(self, data_dir: Path) -> bool:
        """Return True if the data directory contains image assets."""
        if not data_dir.exists():
            return False

        candidate_dirs = [
            p
            for p in data_dir.iterdir()
            if p.is_dir() and p.name.lower().startswith(("train", "test", "images"))
        ]
        if not candidate_dirs:
            candidate_dirs = [data_dir]

        for dir_path in candidate_dirs:
            for i, file_path in enumerate(dir_path.rglob("*")):
                if i >= 200:
                    break
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS:
                    return True

        return False

    def _has_audio_assets(self, data_dir: Path, return_count: bool = False) -> bool | tuple[bool, int]:
        """Return True if the data directory contains audio assets.

        Args:
            data_dir: Path to data directory
            return_count: If True, return (has_audio, count) tuple

        Returns:
            bool or (bool, int) tuple with audio file count
        """
        if not data_dir.exists():
            return (False, 0) if return_count else False

        audio_count = 0

        # Check ALL subdirectories (not just train/test prefixed)
        # This handles non-standard structures like essential_data/, supplemental_data/
        exclude_dirs = {"models", "__pycache__", ".git", "logs", ".ipynb_checkpoints"}

        for item in data_dir.iterdir():
            if item.is_file() and item.suffix.lower() in AUDIO_EXTS:
                audio_count += 1
                if not return_count:
                    return True
            if item.is_dir() and item.name.lower() not in exclude_dirs:
                # Sample up to 500 files to find audio in nested directories
                for i, file_path in enumerate(item.rglob("*")):
                    if i >= 500:
                        break
                    if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTS:
                        audio_count += 1
                        if not return_count and audio_count >= 1:
                            return True

        if return_count:
            return (audio_count > 0, audio_count)
        return audio_count > 0
