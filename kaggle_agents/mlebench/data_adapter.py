"""
MLE-bench Data Adapter.

This module provides utilities to adapt MLE-bench prepared data
to the kaggle-agents expected format.
"""

import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass
class MLEBenchDataInfo:
    """Information about MLE-bench competition data."""

    competition_id: str
    workspace: Path
    train_path: Optional[Path] = None
    test_path: Optional[Path] = None
    sample_submission_path: Optional[Path] = None
    train_csv_path: Optional[Path] = None  # For image competitions with labels CSV
    test_csv_path: Optional[Path] = None
    ground_truth_path: Optional[Path] = None  # Private test labels
    description_path: Optional[Path] = None
    data_type: str = "tabular"  # tabular, image, audio, text
    target_column: str = "target"
    id_column: str = "id"
    extra_files: list[Path] = field(default_factory=list)


class MLEBenchDataAdapter:
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
        print(f"[MLEBenchDataAdapter] Detecting cache path...", flush=True)
        print(f"[MLEBenchDataAdapter]   Path.home() = {Path.home()}", flush=True)

        # 1. Check environment variable first
        env_path = os.environ.get("MLEBENCH_DATA_DIR")
        if env_path:
            env_path_obj = Path(env_path)
            print(f"[MLEBenchDataAdapter]   MLEBENCH_DATA_DIR = {env_path}, exists = {env_path_obj.exists()}", flush=True)
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
        print(f"[MLEBenchDataAdapter] Warning: No cache found, using default: {default}", flush=True)
        return default

    def __init__(self, mle_cache_path: Optional[Path] = None):
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
        print(f"[MLEBenchDataAdapter] Checking if prepared:", flush=True)
        print(f"[MLEBenchDataAdapter]   Competition path: {comp_path}", flush=True)
        print(f"[MLEBenchDataAdapter]   Competition path exists: {comp_path.exists()}", flush=True)
        print(f"[MLEBenchDataAdapter]   Public dir: {public_dir}", flush=True)
        print(f"[MLEBenchDataAdapter]   Public dir exists: {public_dir.exists()}", flush=True)

        # Also check the base competition directory structure
        base_comp_dir = self.mle_cache / competition_id
        if base_comp_dir.exists():
            try:
                contents = list(base_comp_dir.iterdir())
                print(f"[MLEBenchDataAdapter]   Base dir contents: {[p.name for p in contents]}", flush=True)
            except Exception as e:
                print(f"[MLEBenchDataAdapter]   Error listing base dir: {e}", flush=True)

        return public_dir.exists()

    def detect_data_type(self, public_dir: Path) -> str:
        """
        Detect the type of data in the competition.

        Returns:
            'tabular', 'image', 'audio', or 'text'
        """
        # Check for image directories
        for dir_name in ['train', 'test', 'images', 'train_images', 'test_images']:
            dir_path = public_dir / dir_name
            if dir_path.is_dir():
                # Sample files in directory
                sample_files = list(dir_path.glob('*'))[:10]
                extensions = [f.suffix.lower() for f in sample_files if f.is_file()]

                if any(ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'] for ext in extensions):
                    return 'image'
                elif any(ext in ['.wav', '.mp3', '.flac', '.ogg'] for ext in extensions):
                    return 'audio'

        # Check for text-heavy CSVs
        for csv_file in public_dir.glob('*.csv'):
            if 'train' in csv_file.name.lower():
                try:
                    df = pd.read_csv(csv_file, nrows=5)
                    # Check for text columns (long strings)
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            avg_len = df[col].astype(str).str.len().mean()
                            if avg_len > 100:  # Long text
                                return 'text'
                except Exception:
                    pass

        return 'tabular'

    def _extract_zips(self, directory: Path) -> None:
        """Extract all ZIP files in directory."""
        for zip_file in directory.glob('*.zip'):
            extract_dir = directory / zip_file.stem
            if not extract_dir.exists():
                print(f"   Extracting: {zip_file.name}")
                try:
                    with zipfile.ZipFile(zip_file, 'r') as z:
                        z.extractall(directory)
                except zipfile.BadZipFile:
                    print(f"   Warning: {zip_file.name} is not a valid zip")

    def _find_csv_file(self, directory: Path, patterns: list[str]) -> Optional[Path]:
        """Find a CSV file matching any of the patterns."""
        for pattern in patterns:
            matches = list(directory.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _detect_target_column(self, sample_sub_path: Path) -> str:
        """Detect target column from sample submission."""
        try:
            df = pd.read_csv(sample_sub_path, nrows=1)
            if len(df.columns) >= 2:
                return df.columns[1]
        except Exception:
            pass
        return 'target'

    def _detect_id_column(self, sample_sub_path: Path) -> str:
        """Detect ID column from sample submission."""
        try:
            df = pd.read_csv(sample_sub_path, nrows=1)
            if len(df.columns) >= 1:
                return df.columns[0]
        except Exception:
            pass
        return 'id'

    def prepare_workspace(
        self,
        competition_id: str,
        workspace_path: Optional[Path] = None,
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
        sample_sub = self._find_csv_file(public_dir, [
            'sample_submission*.csv',
            'sampleSubmission*.csv',
            '*sample*.csv',
        ])
        if sample_sub:
            info.sample_submission_path = sample_sub
            info.target_column = self._detect_target_column(sample_sub)
            info.id_column = self._detect_id_column(sample_sub)
            print(f"   Sample submission: {sample_sub.name}")
            print(f"   Target column: {info.target_column}")

        # Find train data
        if data_type == 'image':
            # Image competition: look for train directory + labels CSV
            for dir_name in ['train', 'train_images', 'images/train']:
                train_dir = public_dir / dir_name
                if train_dir.is_dir():
                    info.train_path = train_dir
                    break

            # Look for labels CSV
            train_csv = self._find_csv_file(public_dir, [
                'train.csv', 'train_labels.csv', 'labels.csv'
            ])
            if train_csv:
                info.train_csv_path = train_csv
                print(f"   Train images: {info.train_path}")
                print(f"   Train labels: {train_csv.name}")
        else:
            # Tabular/text: direct CSV
            train_csv = self._find_csv_file(public_dir, ['train.csv', 'train*.csv'])
            if train_csv:
                info.train_path = train_csv
                info.train_csv_path = train_csv
                print(f"   Train: {train_csv.name}")

        # Find test data
        if data_type == 'image':
            for dir_name in ['test', 'test_images', 'images/test']:
                test_dir = public_dir / dir_name
                if test_dir.is_dir():
                    info.test_path = test_dir
                    break

            test_csv = self._find_csv_file(public_dir, ['test.csv', 'test*.csv'])
            if test_csv:
                info.test_csv_path = test_csv
                print(f"   Test images: {info.test_path}")
        else:
            test_csv = self._find_csv_file(public_dir, ['test.csv', 'test*.csv'])
            if test_csv:
                info.test_path = test_csv
                info.test_csv_path = test_csv
                print(f"   Test: {test_csv.name}")

        # Find ground truth (for validation after submission)
        if private_dir.exists():
            gt_file = self._find_csv_file(private_dir, [
                'test.csv', 'answers.csv', 'solution.csv', 'test_labels.csv'
            ])
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
        import shutil

        print(f"   Creating workspace links...", flush=True)

        # Files/dirs to link
        items_to_link = []

        # Add train data
        if info.train_path and info.train_path.exists():
            items_to_link.append(("train", info.train_path))
        if info.train_csv_path and info.train_csv_path.exists():
            items_to_link.append(("train.csv", info.train_csv_path))

        # Add test data
        if info.test_path and info.test_path.exists():
            items_to_link.append(("test", info.test_path))
        if info.test_csv_path and info.test_csv_path.exists():
            items_to_link.append(("test.csv", info.test_csv_path))

        # Add sample submission
        if info.sample_submission_path and info.sample_submission_path.exists():
            items_to_link.append(("sample_submission.csv", info.sample_submission_path))

        # Also link any other CSVs in public_dir
        for csv_file in public_dir.glob("*.csv"):
            name = csv_file.name
            if not any(name == item[0] for item in items_to_link):
                items_to_link.append((name, csv_file))

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

        if (workspace / "test.csv").exists():
            info.test_csv_path = workspace / "test.csv"
            if info.data_type == "tabular":
                info.test_path = workspace / "test.csv"
        if (workspace / "test").exists():
            info.test_path = workspace / "test"

        if (workspace / "sample_submission.csv").exists():
            info.sample_submission_path = workspace / "sample_submission.csv"

        print(f"   Workspace setup complete!", flush=True)

    def get_state_paths(self, info: MLEBenchDataInfo) -> dict[str, Any]:
        """
        Convert MLEBenchDataInfo to paths dict for KaggleState.

        Args:
            info: MLEBenchDataInfo from prepare_workspace

        Returns:
            Dictionary with paths for state initialization
        """
        return {
            "working_directory": str(info.workspace),
            "train_data_path": str(info.train_csv_path or info.train_path or ""),
            "test_data_path": str(info.test_csv_path or info.test_path or ""),
            "sample_submission_path": str(info.sample_submission_path or ""),
            "target_col": info.target_column,
            "data_files": {
                "train": str(info.train_path) if info.train_path else "",
                "test": str(info.test_path) if info.test_path else "",
                "train_csv": str(info.train_csv_path) if info.train_csv_path else "",
                "test_csv": str(info.test_csv_path) if info.test_csv_path else "",
                "sample_submission": str(info.sample_submission_path) if info.sample_submission_path else "",
                "data_type": info.data_type,
            },
        }

    def read_description(self, info: MLEBenchDataInfo) -> str:
        """Read competition description if available."""
        if info.description_path and info.description_path.exists():
            return info.description_path.read_text()
        return ""
