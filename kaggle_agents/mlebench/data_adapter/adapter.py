"""
MLE-bench Data Adapter - Main adapter class.

Combines all mixins to provide the full data adapter functionality.
"""

from __future__ import annotations

import os
from pathlib import Path

from .dataclasses import MLEBenchDataInfo
from .detection import DetectionMixin
from .file_finders import FileFinderMixin
from .workspace import WorkspaceMixin
from .zip_handlers import ZipHandlerMixin


class MLEBenchDataAdapter(
    DetectionMixin,
    FileFinderMixin,
    ZipHandlerMixin,
    WorkspaceMixin,
):
    """
    Adapter to prepare MLE-bench data for kaggle-agents workflow.

    MLE-bench structure:
        ~/.cache/mle-bench/data/{competition}/prepared/
            public/
                train.csv or train/ (directory with images)
                test.csv or test/ (directory with images)
                sample_submission.csv
                description.md
            private/
                test.csv (ground truth labels)

    kaggle-agents expected structure:
        /workspace/{competition}/
            train.csv
            test.csv
            sample_submission.csv
            models/
    """

    @staticmethod
    def _detect_mle_cache() -> Path:
        """Detect MLE-bench cache path based on environment."""
        print("[MLEBenchDataAdapter] Detecting cache path...", flush=True)
        print(f"[MLEBenchDataAdapter]   Path.home() = {Path.home()}", flush=True)

        # 1. Check environment variable first
        env_path = os.environ.get("MLEBENCH_DATA_DIR")
        if env_path:
            env_path_obj = Path(env_path)
            print(
                f"[MLEBenchDataAdapter]   MLEBENCH_DATA_DIR = {env_path}, exists = {env_path_obj.exists()}",
                flush=True,
            )
            if env_path_obj.exists():
                return env_path_obj

        # 2. Check common locations in order of priority
        candidates = [
            # User home (works in Colab: /root/.cache)
            Path.home() / ".cache" / "mle-bench" / "data",
            # Explicit /root for containers
            Path("/root/.cache/mle-bench/data"),
            # Colab content directory (alternative)
            Path("/content/.cache/mle-bench/data"),
        ]

        for path in candidates:
            exists = path.exists()
            print(f"[MLEBenchDataAdapter]   Checking {path}, exists = {exists}", flush=True)
            if exists:
                return path

        # Default fallback (will be created if needed)
        default = Path.home() / ".cache" / "mle-bench" / "data"
        print(
            f"[MLEBenchDataAdapter] Warning: No cache found, using default: {default}", flush=True
        )
        return default

    def __init__(self, mle_cache_path: Path | None = None):
        """
        Initialize the adapter.

        Args:
            mle_cache_path: Path to MLE-bench cache directory (auto-detected if None)
        """
        if mle_cache_path:
            self.mle_cache = Path(mle_cache_path)
        else:
            self.mle_cache = self._detect_mle_cache()

        print(f"[MLEBenchDataAdapter] Using cache path: {self.mle_cache}", flush=True)

    def get_competition_path(self, competition_id: str) -> Path:
        """Get the prepared data path for a competition."""
        return self.mle_cache / competition_id / "prepared"

    def is_competition_prepared(self, competition_id: str) -> bool:
        """Check if a competition is already prepared by MLE-bench."""
        comp_path = self.get_competition_path(competition_id)
        public_dir = comp_path / "public"

        # Debug: show what we're looking for
        print("[MLEBenchDataAdapter] Checking if prepared:", flush=True)
        print(f"[MLEBenchDataAdapter]   Competition path: {comp_path}", flush=True)
        print(f"[MLEBenchDataAdapter]   Competition path exists: {comp_path.exists()}", flush=True)
        print(f"[MLEBenchDataAdapter]   Public dir: {public_dir}", flush=True)
        print(f"[MLEBenchDataAdapter]   Public dir exists: {public_dir.exists()}", flush=True)

        # Also check the base competition directory structure
        base_comp_dir = self.mle_cache / competition_id
        if base_comp_dir.exists():
            try:
                contents = list(base_comp_dir.iterdir())
                print(
                    f"[MLEBenchDataAdapter]   Base dir contents: {[p.name for p in contents]}",
                    flush=True,
                )
            except Exception as e:
                print(f"[MLEBenchDataAdapter]   Error listing base dir: {e}", flush=True)

        # Return True if public_dir exists (even if empty).
        # Empty directories will be handled by fallback logic in prepare_workspace().
        # We allow prepare_workspace() to run so it can attempt recovery from raw/ or ZIP.
        if public_dir.exists():
            try:
                has_contents = any(public_dir.iterdir())
                if not has_contents:
                    print(
                        "[MLEBenchDataAdapter]   ⚠️ public/ exists but is EMPTY - "
                        "fallback will be attempted in prepare_workspace()",
                        flush=True,
                    )
            except PermissionError:
                pass  # Will be handled in prepare_workspace()
            return True

        return False

    def prepare_workspace(
        self,
        competition_id: str,
        workspace_path: Path | None = None,
    ) -> MLEBenchDataInfo:
        """
        Prepare workspace with MLE-bench data for kaggle-agents.

        This method:
        1. Locates MLE-bench prepared data
        2. Extracts any ZIP files
        3. Identifies train/test/sample_submission files
        4. Sets up workspace directory structure

        Args:
            competition_id: MLE-bench competition ID
            workspace_path: Optional custom workspace path

        Returns:
            MLEBenchDataInfo with all paths and metadata
        """
        comp_path = self.get_competition_path(competition_id)
        public_dir = comp_path / "public"
        private_dir = comp_path / "private"

        if not public_dir.exists():
            raise FileNotFoundError(
                f"MLE-bench data not found for '{competition_id}'. "
                f"Run: mlebench prepare -c {competition_id}"
            )

        # Create workspace
        if workspace_path is None:
            workspace_path = Path("/content/kaggle_competitions/competitions") / competition_id
        workspace_path.mkdir(parents=True, exist_ok=True)

        print(f"\n[MLE-BENCH] Preparing data for: {competition_id}")
        print(f"   Source: {public_dir}")
        print(f"   Workspace: {workspace_path}")

        # Extract ZIPs
        self._extract_zips(public_dir)

        # Check if public_dir is still empty after extraction - attempt fallback
        public_contents = list(public_dir.glob("*"))
        if not public_contents:
            print("   ⚠️ public/ is empty after extraction, attempting fallback...")
            fallback_success = self._populate_from_fallback(competition_id, public_dir)
            if fallback_success:
                public_contents = list(public_dir.glob("*"))

        # If still empty after fallback, raise a clear error
        if not public_contents:
            raise FileNotFoundError(
                f"Competition data not found for '{competition_id}'.\n"
                f"The public/ directory at {public_dir} is empty.\n"
                f"Please run: mlebench prepare -c {competition_id}\n"
                f"Or manually extract data to: {public_dir}"
            )

        # Detect data type
        data_type = self.detect_data_type(public_dir)
        print(f"   Data type: {data_type}")

        # Initialize result
        info = MLEBenchDataInfo(
            competition_id=competition_id,
            workspace=workspace_path,
            data_type=data_type,
        )

        # Find sample submission (critical for format)
        sample_sub = self._find_csv_file(
            public_dir,
            [
                "sample_submission*.csv",
                "sampleSubmission*.csv",
                "*sample*.csv",
            ],
        )
        if sample_sub:
            info.sample_submission_path = sample_sub
            info.target_column = self._detect_target_column(sample_sub)
            info.id_column = self._detect_id_column(sample_sub)
            print(f"   Sample submission: {sample_sub.name}")
            print(f"   Target column: {info.target_column}")

        # Find train data - check both directories and CSVs regardless of data_type
        info = self._find_train_data(info, public_dir, data_type)

        # Find test data
        info = self._find_test_data(info, public_dir, data_type)

        # Debug: list all files found
        all_files = list(public_dir.glob("*"))
        print(f"   All files in public_dir: {[f.name for f in all_files]}")

        # Find ground truth (for validation after submission)
        if private_dir.exists():
            gt_file = self._find_csv_file(
                private_dir, ["test.csv", "answers.csv", "solution.csv", "test_labels.csv"]
            )
            if gt_file:
                info.ground_truth_path = gt_file

        # Find description
        desc_file = public_dir / "description.md"
        if desc_file.exists():
            info.description_path = desc_file

        # Create models directory in workspace
        (workspace_path / "models").mkdir(exist_ok=True)

        # Create symlinks in workspace pointing to MLE-bench data
        # This allows the developer agent to find files in working_directory
        self._create_workspace_links(info, workspace_path, public_dir)

        return info

    def _find_train_data(
        self, info: MLEBenchDataInfo, public_dir: Path, data_type: str
    ) -> MLEBenchDataInfo:
        """Find train data paths and update info."""
        # Standard train directory patterns
        standard_train_dirs = [
            "train",
            "train_images",
            "train_imgs",
            "training",
            "images/train",
        ]
        # Non-standard patterns used by some MLE-bench competitions
        nonstandard_data_dirs = [
            "essential_data",
            "supplemental_data",
            "data",
            "raw_data",
            "audio",
            "audio_data",
        ]

        # Check standard directories first
        for dir_name in standard_train_dirs:
            train_dir = public_dir / dir_name
            if train_dir.is_dir():
                info.train_path = train_dir
                print(f"   Train dir: {train_dir.name}/")
                break

        # If no standard train dir found, check non-standard directories for audio/image data
        if info.train_path is None:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
            image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            for dir_name in nonstandard_data_dirs:
                data_dir = public_dir / dir_name
                if data_dir.is_dir():
                    # Check if this directory contains audio/image files
                    sample_files = list(data_dir.glob("*"))[:50]
                    has_audio = any(
                        f.suffix.lower() in audio_exts for f in sample_files if f.is_file()
                    )
                    has_images = any(
                        f.suffix.lower() in image_exts for f in sample_files if f.is_file()
                    )
                    if has_audio or has_images:
                        info.train_path = data_dir
                        dtype = "audio" if has_audio else "image"
                        print(f"   Train dir (non-standard): {dir_name}/ [{dtype}]")
                        break

        # Train CSV (labels for image competitions, or data for tabular)
        train_csv = self._find_csv_file(
            public_dir, ["train.csv", "train_labels.csv", "labels.csv", "train*.csv"]
        )
        if train_csv:
            info.train_csv_path = train_csv
            if data_type == "tabular" and (info.train_path is None or info.train_path.is_file()):
                info.train_path = train_csv
            print(f"   Train CSV: {train_csv.name}")

        # Non-standard label file search
        if info.train_csv_path is None or data_type == "audio":
            for data_subdir in nonstandard_data_dirs:
                subdir_path = public_dir / data_subdir
                if subdir_path.is_dir():
                    # Search for label files in this subdirectory
                    label_files = self._find_label_files(subdir_path, recursive=True)
                    if label_files:
                        info.label_files.extend(label_files)
                        for lf in label_files:
                            rel_path = lf.relative_to(public_dir)
                            print(f"   Label file found: {rel_path}")

                    # Search for audio source directory
                    if info.audio_source_path is None:
                        audio_src = self._find_audio_source_dir(subdir_path)
                        if audio_src:
                            info.audio_source_path = audio_src
                            rel_path = audio_src.relative_to(public_dir)
                            print(f"   Audio source dir: {rel_path}/")
                            # Fallback: set train_path to audio source if not already set
                            if info.train_path is None:
                                info.train_path = audio_src
                                print(f"   Train dir (from audio source): {rel_path}/")

        # Image-to-image: look for "clean"/target image directories
        clean_dir_candidates = [
            "train_cleaned",
            "train_clean",
            "clean",
            "cleaned",
            "gt",
            "ground_truth",
            "train_gt",
            "target",
            "targets",
            "train_target",
        ]
        for dir_name in clean_dir_candidates:
            clean_dir = public_dir / dir_name
            if clean_dir.is_dir():
                info.clean_train_path = clean_dir
                print(f"   Clean/target dir: {clean_dir.name}/")
                break

        # Train ZIP fallback (common in CV competitions)
        if info.train_path is None:
            train_zip = self._find_first_zip(public_dir, kind="train")
            if train_zip:
                info.train_path = train_zip
                print(f"   Train ZIP: {train_zip.name}")

        return info

    def _find_test_data(
        self, info: MLEBenchDataInfo, public_dir: Path, data_type: str
    ) -> MLEBenchDataInfo:
        """Find test data paths and update info."""
        # Standard test directory patterns
        standard_test_dirs = [
            "test",
            "test_images",
            "test_imgs",
            "testing",
            "images/test",
        ]

        for dir_name in standard_test_dirs:
            test_dir = public_dir / dir_name
            if test_dir.is_dir():
                info.test_path = test_dir
                print(f"   Test dir: {test_dir.name}/")
                break

        # If no standard test dir found but train was found in non-standard dir,
        # check if the same directory contains test data
        if info.test_path is None and info.train_path and info.train_path.is_dir():
            parent_name = info.train_path.name.lower()
            if parent_name in ["essential_data", "supplemental_data", "data", "audio", "audio_data"]:
                info.test_path = info.train_path
                print(f"   Test dir (shared with train): {info.train_path.name}/")

        # Test CSV
        test_csv = self._find_csv_file(public_dir, ["test.csv", "test*.csv"])
        if test_csv:
            info.test_csv_path = test_csv
            if data_type == "tabular" and (info.test_path is None or info.test_path.is_file()):
                info.test_path = test_csv
            print(f"   Test CSV: {test_csv.name}")

        # Test ZIP fallback (common in CV competitions)
        if info.test_path is None:
            test_zip = self._find_first_zip(public_dir, kind="test")
            if test_zip:
                info.test_path = test_zip
                print(f"   Test ZIP: {test_zip.name}")

        # Some image competitions store all images under a single folder (e.g., `images/`)
        if data_type == "image" and (info.train_path is None or info.test_path is None):
            images_dir = public_dir / "images"
            if images_dir.is_dir():
                if info.train_path is None:
                    info.train_path = images_dir
                    print("   Train dir fallback: images/")
                if info.test_path is None:
                    info.test_path = images_dir
                    print("   Test dir fallback: images/")

        # Generic fallback: search ALL subdirectories for train/test data
        if info.train_path is None:
            train_patterns = ["train.csv", "train", "train_images", "training"]
            found = self._find_data_in_subdirs(public_dir, train_patterns)
            if found:
                if found.is_file() and found.suffix == ".csv":
                    info.train_csv_path = found
                info.train_path = found
                print(f"   Train found in subdir: {found.relative_to(public_dir)}")

        if info.test_path is None:
            test_patterns = ["test.csv", "test", "test_images", "testing"]
            found = self._find_data_in_subdirs(public_dir, test_patterns)
            if found:
                if found.is_file() and found.suffix == ".csv":
                    info.test_csv_path = found
                info.test_path = found
                print(f"   Test found in subdir: {found.relative_to(public_dir)}")

        return info
