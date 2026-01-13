"""
Workspace management for MLE-bench data adapter.

Contains methods for creating workspace directories and symlinks.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .dataclasses import MLEBenchDataInfo


class WorkspaceMixin:
    """Mixin providing workspace management methods."""

    def _create_workspace_links(
        self,
        info: MLEBenchDataInfo,
        workspace: Path,
        public_dir: Path,
    ) -> None:
        """
        Create symlinks in workspace pointing to MLE-bench data files.

        This ensures the developer agent can find data files in the working directory.
        """
        print("   Creating workspace links...", flush=True)

        # Files/dirs to link
        items_to_link = []

        # Link canonical train/test assets for non-tabular domains.
        # We always expose them as `train/` and `test/` inside the workspace, even if the
        # underlying directory is named differently (e.g., `train_images/`).
        if info.train_path and info.train_path.is_dir():
            items_to_link.append(("train", info.train_path))
            print(f"      Found train dir: {info.train_path}", flush=True)
        if info.test_path and info.test_path.is_dir():
            items_to_link.append(("test", info.test_path))
            print(f"      Found test dir: {info.test_path}", flush=True)

        # Add train CSV
        if info.train_csv_path and info.train_csv_path.exists():
            items_to_link.append(("train.csv", info.train_csv_path))

        # Add clean/target train directory for image-to-image tasks
        if info.clean_train_path and info.clean_train_path.is_dir():
            items_to_link.append((info.clean_train_path.name, info.clean_train_path))
            print(f"      Found clean/target dir: {info.clean_train_path}", flush=True)

        # Add test CSV (if exists - many image competitions don't have this)
        if info.test_csv_path and info.test_csv_path.exists():
            items_to_link.append(("test.csv", info.test_csv_path))

        # Add sample submission
        # Handle case where "sample_submission.csv" is actually a directory containing the real CSV
        if info.sample_submission_path and info.sample_submission_path.exists():
            target = info.sample_submission_path
            if target.is_dir():
                inner_csvs = sorted(target.glob("*.csv"))
                if inner_csvs:
                    target = inner_csvs[0]
                    print(
                        f"      📂 Resolved sample_submission directory to file: {target.name}",
                        flush=True,
                    )
                    # Update info so downstream code uses the correct path
                    info.sample_submission_path = target
            if target.is_file():
                items_to_link.append(("sample_submission.csv", target))

        # Add audio source directory if found (for MLSP-like competitions)
        if info.audio_source_path and info.audio_source_path.is_dir():
            # Link audio source as 'audio' for easy access
            if not any(item[0] == "audio" for item in items_to_link):
                items_to_link.append(("audio", info.audio_source_path))
                print(f"      Found audio source: {info.audio_source_path}", flush=True)

        # Add non-standard label files (.txt files from essential_data/, etc.)
        # This is critical for MLSP 2013 Birds and similar competitions
        if info.label_files:
            for label_file in info.label_files:
                if label_file.exists():
                    # Keep original name to avoid confusion
                    name = label_file.name
                    if not any(name == item[0] for item in items_to_link):
                        items_to_link.append((name, label_file))
                        print(f"      Found label file: {name}", flush=True)

        # Also link ZIPs (common in CV competitions); keep original names for transparency.
        for zip_file in public_dir.glob("*.zip"):
            items_to_link.append((zip_file.name, zip_file))

        # Also link any other CSVs in public_dir
        for csv_file in public_dir.glob("*.csv"):
            name = csv_file.name
            if not any(name == item[0] for item in items_to_link):
                items_to_link.append((name, csv_file))

        # Also link any subdirectories from public_dir (for competitions with nested data)
        # This handles competitions like mlsp-2013-birds with essential_data/, supplemental_data/
        for item in public_dir.iterdir():
            if item.is_dir():
                # Skip if already linked (e.g., train/, test/, clean dirs)
                if not any(item.name == link[0] for link in items_to_link):
                    items_to_link.append((item.name, item))
                    print(f"      Linking subdirectory: {item.name}", flush=True)

        # Create symlinks
        for link_name, target in items_to_link:
            link_path = workspace / link_name
            if link_path.exists() or link_path.is_symlink():
                # Remove existing link/file
                if link_path.is_symlink() or link_path.is_file():
                    link_path.unlink()
                elif link_path.is_dir():
                    shutil.rmtree(link_path)

            try:
                # Create symlink
                link_path.symlink_to(target)
                print(f"      Linked: {link_name} -> {target}", flush=True)
            except OSError as e:
                # Symlinks may fail on some systems, fall back to copy
                print(f"      Symlink failed for {link_name}, copying instead: {e}", flush=True)
                if target.is_dir():
                    shutil.copytree(target, link_path)
                else:
                    shutil.copy2(target, link_path)
                print(f"      Copied: {link_name}", flush=True)

        # Update info paths to point to workspace
        if (workspace / "train.csv").exists():
            info.train_csv_path = workspace / "train.csv"
            if info.data_type == "tabular":
                info.train_path = workspace / "train.csv"
        if (workspace / "train").exists():
            info.train_path = workspace / "train"
        if info.train_path and info.train_path.is_file():
            linked_train_file = workspace / info.train_path.name
            if linked_train_file.exists():
                info.train_path = linked_train_file

        if (workspace / "test.csv").exists():
            info.test_csv_path = workspace / "test.csv"
            if info.data_type == "tabular":
                info.test_path = workspace / "test.csv"
        if (workspace / "test").exists():
            info.test_path = workspace / "test"
        if info.test_path and info.test_path.is_file():
            linked_test_file = workspace / info.test_path.name
            if linked_test_file.exists():
                info.test_path = linked_test_file

        if (workspace / "sample_submission.csv").exists():
            info.sample_submission_path = workspace / "sample_submission.csv"

        print("   Workspace setup complete!", flush=True)

    def get_state_paths(self, info: MLEBenchDataInfo) -> dict[str, Any]:
        """
        Convert MLEBenchDataInfo to paths dict for KaggleState.

        Args:
            info: MLEBenchDataInfo from prepare_workspace

        Returns:
            Dictionary with paths for state initialization
        """
        # Prefer the main data asset (dir/zip) for non-tabular domains; keep CSVs in `data_files`.
        train_data_path = info.train_path or info.train_csv_path
        test_data_path = info.test_path or info.test_csv_path

        # Validate paths exist - if not, search workspace for actual data
        workspace = info.workspace
        if train_data_path and not Path(train_data_path).exists():
            print(f"   ⚠️ Train path does not exist: {train_data_path}")
            # Search workspace subdirectories for actual train data
            found = self._find_data_in_subdirs(
                workspace,
                ["train.csv", "train", "essential_data", "supplemental_data", "data"],
            )
            if found:
                train_data_path = found
                print(f"   ✓ Found train data: {found}")

        if test_data_path and not Path(test_data_path).exists():
            print(f"   ⚠️ Test path does not exist: {test_data_path}")
            # Search workspace subdirectories for actual test data
            found = self._find_data_in_subdirs(
                workspace,
                ["test.csv", "test", "essential_data", "supplemental_data", "data"],
            )
            if found:
                test_data_path = found
                print(f"   ✓ Found test data: {found}")

        # Final validation - warn if still missing
        if train_data_path and not Path(train_data_path).exists():
            print(f"   ⚠️ WARNING: Train data still not found! Path: {train_data_path}")
        if test_data_path and not Path(test_data_path).exists():
            print(f"   ⚠️ WARNING: Test data still not found! Path: {test_data_path}")

        # Build label files list (both CSV and TXT formats)
        label_file_paths = [str(lf) for lf in info.label_files if lf.exists()]

        return {
            "working_directory": str(info.workspace),
            "train_data_path": str(train_data_path or ""),
            "test_data_path": str(test_data_path or ""),
            "sample_submission_path": str(info.sample_submission_path or ""),
            "target_col": info.target_column,
            "data_files": {
                "train": str(info.train_path) if info.train_path else "",
                "test": str(info.test_path) if info.test_path else "",
                "clean_train": str(info.clean_train_path) if info.clean_train_path else "",
                "train_csv": str(info.train_csv_path) if info.train_csv_path else "",
                "test_csv": str(info.test_csv_path) if info.test_csv_path else "",
                "sample_submission": str(info.sample_submission_path)
                if info.sample_submission_path
                else "",
                "data_type": info.data_type,
                # Non-standard label files (.txt for MLSP 2013 Birds, etc.)
                "label_files": label_file_paths,
                # Audio source directory (e.g., essential_data/src_wavs)
                "audio_source": str(info.audio_source_path) if info.audio_source_path else "",
            },
        }

    def read_description(self, info: MLEBenchDataInfo) -> str:
        """Read competition description if available."""
        if info.description_path and info.description_path.exists():
            return info.description_path.read_text()
        return ""
