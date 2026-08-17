"""
MLE-bench Data Adapter - Main adapter class.

Combines all mixins to provide the full data adapter functionality.
"""

from __future__ import annotations

import os
from itertools import islice
from pathlib import Path

from .artifact_roles import (
    build_auxiliary_artifact,
    one_artifact_path,
    resolve_public_artifacts,
)
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
        provenance = self._extract_zips(public_dir)

        # Check if public_dir is still empty after extraction - attempt fallback
        public_contents = list(public_dir.glob("*"))
        if not public_contents:
            print("   ⚠️ public/ is empty after extraction, attempting fallback...")
            fallback_success = self._populate_from_fallback(competition_id, public_dir)
            if fallback_success:
                public_contents = list(public_dir.glob("*"))
                # The fallback copies archives that were never opened above.
                # Extraction is idempotent, so re-running it only adds the
                # provenance of the newly discovered archives.
                provenance = self._extract_zips(public_dir)

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

        # Resolve typed public artifacts before any legacy filename finder:
        # role resolution reads archive/member provenance the globs cannot see.
        info.public_artifacts = resolve_public_artifacts(
            public_dir, provenance, data_type
        )
        self._apply_resolved_roles(info, data_type)

        # Find sample submission (critical for format), only when the bounded
        # resolver left the role unresolved (e.g. a directory-packed template).
        if info.sample_submission_path is None:
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
                print(f"   Sample submission: {sample_sub.name}")

        # Find train data - check both directories and CSVs regardless of data_type
        info = self._find_train_data(info, public_dir, data_type)

        # Find test data
        info = self._find_test_data(info, public_dir, data_type)

        # Recursive auxiliary discovery runs last: every legacy label-named
        # candidate is inspected and typed, so no untyped second label lane
        # survives into state.
        self._append_recursive_auxiliary_artifacts(info, public_dir)

        # Submission roles are resolved only once the public test schema is
        # known: position alone misreads templates whose first column is the
        # prediction and whose remaining columns echo the test input.
        if info.sample_submission_path:
            info.target_columns = self._detect_target_columns(
                info.sample_submission_path, info.test_csv_path
            )
            info.target_column = info.target_columns[0]
            info.id_column = self._detect_id_column(
                info.sample_submission_path, info.test_csv_path
            )
            print(f"   Target column: {info.target_column}")
            if len(info.target_columns) > 1:
                print(
                    "   Ordered submission targets: "
                    f"{info.target_columns}"
                )

        # Debug: list all files found
        all_files = list(public_dir.glob("*"))
        print(f"   All files in public_dir: {[f.name for f in all_files]}")

        # Private labels belong exclusively to the external MLE-bench grader.
        # Do not discover or expose their path in agent state.

        # Find description
        desc_file = public_dir / "description.md"
        if desc_file.exists():
            info.description_path = desc_file

        # Create models directory in workspace
        (workspace_path / "models").mkdir(exist_ok=True)

        # Stage only public inputs inside the isolated run workspace.
        self._create_workspace_links(info, workspace_path, public_dir)

        return info

    @staticmethod
    def _apply_resolved_roles(info: MLEBenchDataInfo, data_type: str) -> None:
        """Fill the compatibility fields the typed resolver already decided.

        Tabular runs consume the resolved tables directly, so both the CSV and
        the primary data path come from them. Media runs only take the CSV
        metadata fields: their train/test directories are detected
        independently and must survive this step.
        """
        resolved_train = one_artifact_path(info.public_artifacts, "train")
        resolved_test = one_artifact_path(info.public_artifacts, "test")
        resolved_submission = one_artifact_path(info.public_artifacts, "submission")

        if resolved_train is not None:
            info.train_csv_path = resolved_train
            if data_type == "tabular":
                info.train_path = resolved_train
        if resolved_test is not None:
            info.test_csv_path = resolved_test
            if data_type == "tabular":
                info.test_path = resolved_test
        if resolved_submission is not None:
            info.sample_submission_path = resolved_submission

    def _append_recursive_auxiliary_artifacts(
        self, info: MLEBenchDataInfo, public_dir: Path
    ) -> None:
        """Type every recursively discovered label candidate.

        The legacy finder only matches filename hints, which is a guess about
        a file's *name*, never about its content. Each candidate it produces
        is inspected here and appended as a typed auxiliary record (including
        an ``unknown`` one carrying its rejection evidence); records already
        resolved from the bounded pool are excluded by normalized path and by
        archive/member identity.
        """
        known_paths = {artifact.path.resolve() for artifact in info.public_artifacts}
        known_members = {
            (artifact.source_archive.resolve(), artifact.path.resolve())
            for artifact in info.public_artifacts
            if artifact.source_archive is not None
        }
        discovered = list(self._find_label_files(public_dir, recursive=True))
        discovered.extend(info.label_files)

        for path in sorted({Path(candidate) for candidate in discovered}, key=str):
            resolved = path.resolve()
            if resolved in known_paths:
                continue
            if any(resolved == member for _, member in known_members):
                continue
            artifact = build_auxiliary_artifact(public_dir, path)
            if artifact is None:
                continue
            known_paths.add(resolved)
            info.public_artifacts.append(artifact)
            try:
                label = path.relative_to(public_dir)
            except ValueError:  # pragma: no cover - defensive
                label = path.name
            print(f"   Auxiliary artifact: {label} [{artifact.layout}]", flush=True)

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
        excluded_dirs = {
            "models",
            "__pycache__",
            ".git",
            ".ipynb_checkpoints",
        }
        candidate_data_dirs = [
            path
            for path in sorted(public_dir.iterdir())
            if path.is_dir()
            and path.name.lower() not in excluded_dirs
            and not path.name.lower().startswith(("test", "clean", "target", "ground_truth"))
        ]

        # Check standard directories first
        for dir_name in standard_train_dirs:
            train_dir = public_dir / dir_name
            if train_dir.is_dir():
                info.train_path = train_dir
                print(f"   Train dir: {train_dir.name}/")
                break

        # If no exact match, try pattern matching for numbered train directories (e.g., train2)
        if info.train_path is None:
            for train_dir in sorted(public_dir.glob("train[0-9]*")):
                if train_dir.is_dir():  # Excludes train2.zip (only directories)
                    info.train_path = train_dir
                    print(f"   Train dir (pattern match): {train_dir.name}/")
                    break

        # If no conventional train directory exists, inspect every supplied
        # directory and select one only from its observed media extensions.
        if info.train_path is None:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
            image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            for data_dir in candidate_data_dirs:
                # islice stops the recursive walk after 500 files; a filtering
                # generator would keep walking the entire tree of large image
                # datasets just to discard the remainder.
                sample_files = islice(
                    (path for path in data_dir.rglob("*") if path.is_file()),
                    500,
                )
                observed_exts = {path.suffix.lower() for path in sample_files}
                has_audio = bool(observed_exts & audio_exts)
                has_images = bool(observed_exts & image_exts)
                if has_audio or has_images:
                    info.train_path = data_dir
                    dtype = "audio" if has_audio else "image"
                    print(f"   Train dir (content detected): {data_dir.name}/ [{dtype}]")
                    break

        # Train CSV (labels for image competitions, or data for tabular).
        # The legacy globs only fill a role the typed resolver left open.
        train_csv = (
            self._find_csv_file(
                public_dir,
                ["train.csv", "train_labels.csv", "labels.csv", "train*.csv"],
            )
            if info.train_csv_path is None
            else None
        )
        if train_csv:
            info.train_csv_path = train_csv
            if data_type == "tabular" and (info.train_path is None or info.train_path.is_file()):
                info.train_path = train_csv
            print(f"   Train CSV: {train_csv.name}")

        # Discover label/split metadata recursively from semantic filename hints.
        if info.train_csv_path is None or data_type == "audio":
            label_files = self._find_label_files(public_dir, recursive=True)
            info.label_files.extend(
                label_file for label_file in label_files if label_file not in info.label_files
            )
            for label_file in label_files:
                print(f"   Label file found: {label_file.relative_to(public_dir)}")

            if info.audio_source_path is None:
                audio_src = self._find_audio_source_dir(public_dir)
                if audio_src:
                    info.audio_source_path = audio_src
                    rel_path = audio_src.relative_to(public_dir)
                    print(f"   Audio source dir: {rel_path}/")
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

        # If no exact match, try pattern matching for numbered test directories (e.g., test2)
        if info.test_path is None:
            for test_dir in sorted(public_dir.glob("test[0-9]*")):
                if test_dir.is_dir():  # Excludes test2.zip (only directories)
                    info.test_path = test_dir
                    print(f"   Test dir (pattern match): {test_dir.name}/")
                    break

        # Some audio datasets identify train/test records through public split
        # metadata while storing all source media together.
        if (
            info.test_path is None
            and data_type == "audio"
            and info.audio_source_path
            and info.audio_source_path.is_dir()
        ):
            info.test_path = info.audio_source_path
            print(f"   Test dir (shared media source): {info.audio_source_path.name}/")

        # Test CSV - only when the typed resolver left the role unresolved.
        test_csv = (
            self._find_csv_file(public_dir, ["test.csv", "test*.csv"])
            if info.test_csv_path is None
            else None
        )
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
                if found.is_file() and found.suffix == ".csv" and info.train_csv_path is None:
                    info.train_csv_path = found
                info.train_path = found
                print(f"   Train found in subdir: {found.relative_to(public_dir)}")

        if info.test_path is None:
            test_patterns = ["test.csv", "test", "test_images", "testing"]
            found = self._find_data_in_subdirs(public_dir, test_patterns)
            if found:
                if found.is_file() and found.suffix == ".csv" and info.test_csv_path is None:
                    info.test_csv_path = found
                info.test_path = found
                print(f"   Test found in subdir: {found.relative_to(public_dir)}")

        return info
