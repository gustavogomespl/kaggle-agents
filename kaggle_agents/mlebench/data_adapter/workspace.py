"""
Workspace management for MLE-bench data adapter.

Contains methods for staging public benchmark data in an isolated workspace.
"""

from __future__ import annotations

import os
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
        """Copy public inputs into the run workspace and create local aliases.

        Absolute symlinks into MLE-bench's ``prepared/public`` directory reveal
        the adjacent private-label location through ``Path.resolve()``. Staging
        the public tree matches the baseline protocol and keeps every path seen
        by generated code inside the run workspace. Aliases such as ``train/``
        are relative links between already-staged public paths.
        """
        print("   Staging public benchmark data...", flush=True)
        public_root = public_dir.resolve()

        # Reject a prepared public tree that itself points outside the public
        # boundary. Following such a link could import private data.
        for source in public_dir.rglob("*"):
            if source.is_symlink() and not source.resolve().is_relative_to(public_root):
                raise ValueError(f"Public input escapes its boundary: {source}")

        for source in public_dir.iterdir():
            destination = workspace / source.name
            if destination.exists() or destination.is_symlink():
                continue
            if source.is_dir():
                shutil.copytree(source, destination, symlinks=False)
            else:
                shutil.copy2(source, destination)

        def staged(source: Path | None) -> Path | None:
            if source is None:
                return None
            resolved = source.resolve()
            if not resolved.is_relative_to(public_root):
                raise ValueError(f"MLE-bench public asset is outside public/: {source}")
            return workspace / resolved.relative_to(public_root)

        def local_alias(name: str, source: Path | None) -> Path | None:
            staged_source = staged(source)
            if staged_source is None or not staged_source.exists():
                return None
            alias = workspace / name
            if alias.exists() or alias.is_symlink():
                try:
                    if alias.resolve() == staged_source.resolve():
                        return alias
                except OSError:
                    pass
                # The alias name is occupied by a different staged artifact
                # (e.g. a sample_submission.csv that is a directory in the
                # public tree). Keep pointing at the detected source instead
                # of inheriting the collision.
                return staged_source
            try:
                relative_target = os.path.relpath(staged_source, alias.parent)
                alias.symlink_to(relative_target, target_is_directory=staged_source.is_dir())
            except OSError:
                if staged_source.is_dir():
                    shutil.copytree(staged_source, alias)
                else:
                    shutil.copy2(staged_source, alias)
            return alias

        # Resolve a sample-submission directory to its actual CSV before
        # constructing the canonical local alias.
        if info.sample_submission_path and info.sample_submission_path.is_dir():
            inner_csvs = sorted(info.sample_submission_path.glob("*.csv"))
            if inner_csvs:
                info.sample_submission_path = inner_csvs[0]

        info.train_path = local_alias(
            "train" if info.train_path and info.train_path.is_dir() else info.train_path.name,
            info.train_path,
        ) if info.train_path else None
        info.test_path = local_alias(
            "test" if info.test_path and info.test_path.is_dir() else info.test_path.name,
            info.test_path,
        ) if info.test_path else None
        info.clean_train_path = (
            local_alias(info.clean_train_path.name, info.clean_train_path)
            if info.clean_train_path
            else None
        )
        info.train_csv_path = local_alias("train.csv", info.train_csv_path)
        info.test_csv_path = local_alias("test.csv", info.test_csv_path)
        info.sample_submission_path = local_alias(
            "sample_submission.csv", info.sample_submission_path
        )
        info.audio_source_path = local_alias("audio", info.audio_source_path)
        info.label_files = [
            alias
            for label_file in info.label_files
            if (alias := local_alias(label_file.name, label_file)) is not None
        ]
        info.description_path = staged(info.description_path)
        info.extra_files = [
            path
            for extra_file in info.extra_files
            if (path := staged(extra_file)) is not None
        ]

        # Normalize conventional paths after staging. Type checks (is_file/
        # is_dir) matter: a same-named artifact of the wrong kind (e.g. a
        # sample_submission.csv directory) must not clobber a resolved path.
        if (workspace / "train.csv").is_file():
            info.train_csv_path = workspace / "train.csv"
            if info.data_type == "tabular":
                info.train_path = workspace / "train.csv"
        if (workspace / "train").is_dir():
            info.train_path = workspace / "train"
        if info.train_path and info.train_path.is_file():
            linked_train_file = workspace / info.train_path.name
            if linked_train_file.is_file():
                info.train_path = linked_train_file

        if (workspace / "test.csv").is_file():
            info.test_csv_path = workspace / "test.csv"
            if info.data_type == "tabular":
                info.test_path = workspace / "test.csv"
        if (workspace / "test").is_dir():
            info.test_path = workspace / "test"
        if info.test_path and info.test_path.is_file():
            linked_test_file = workspace / info.test_path.name
            if linked_test_file.is_file():
                info.test_path = linked_test_file

        if (workspace / "sample_submission.csv").is_file():
            info.sample_submission_path = workspace / "sample_submission.csv"

        print("   Public data staged inside run workspace!", flush=True)

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
                ["train.csv", "train"],
            )
            if found:
                train_data_path = found
                print(f"   ✓ Found train data: {found}")

        if test_data_path and not Path(test_data_path).exists():
            print(f"   ⚠️ Test path does not exist: {test_data_path}")
            # Search workspace subdirectories for actual test data
            found = self._find_data_in_subdirs(
                workspace,
                ["test.csv", "test"],
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
            "target_cols": list(info.target_columns),
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
                # Label and split-metadata files discovered from public data.
                "label_files": label_file_paths,
                # Audio source directory inferred from local extensions.
                "audio_source": str(info.audio_source_path) if info.audio_source_path else "",
            },
        }

    def read_description(self, info: MLEBenchDataInfo) -> str:
        """Read competition description if available."""
        if info.description_path and info.description_path.exists():
            return info.description_path.read_text()
        return ""
