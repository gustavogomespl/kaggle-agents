"""
Data type detection for MLE-bench competitions.

Contains methods for detecting data type, target column, and ID column.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ...core.config import get_run_seed
from ...utils.label_parser import infer_filename_label_table


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

        # 1) Check common directories, then every supplied top-level directory.
        checked_dirs = set()
        patterns = [
            "train",
            "test",
            "images",
            "train_images",
            "test_images",
            "*",
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

    def _detect_target_columns(
        self,
        sample_sub_path: Path,
        test_csv_path: Path | None = None,
    ) -> list[str]:
        """Read ordered prediction columns without inferring their semantics."""
        from kaggle_agents.utils.target_inference import (
            _read_schema_columns,
            split_submission_schema,
        )

        try:
            columns = [
                str(column)
                for column in pd.read_csv(sample_sub_path, nrows=0).columns
            ]
            if len(columns) >= 2:
                _, predicted = split_submission_schema(
                    columns,
                    _read_schema_columns(test_csv_path),
                )
                return predicted
        except Exception:
            pass
        return ["target"]

    def _detect_id_column(
        self,
        sample_sub_path: Path,
        test_csv_path: Path | None = None,
    ) -> str:
        """Detect ID column from sample submission."""
        from kaggle_agents.utils.target_inference import (
            _read_schema_columns,
            split_submission_schema,
        )

        try:
            columns = [
                str(column)
                for column in pd.read_csv(sample_sub_path, nrows=0).columns
            ]
            if len(columns) >= 2:
                echoed, _ = split_submission_schema(
                    columns,
                    _read_schema_columns(test_csv_path),
                )
                if echoed:
                    return echoed[0]
            if columns:
                return columns[0]
        except Exception:
            pass
        return "id"

    AUDIO_LABEL_EXTENSIONS = frozenset(
        {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif"}
    )
    IMAGE_LABEL_EXTENSIONS = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    )

    def _detect_media_labels_from_filenames(
        self,
        media_dir: Path,
        extensions: frozenset[str],
        explicit_pattern: str | None = None,
    ) -> tuple[list[str], list[str], list[Path]]:
        """Extract labels only from explicit or uniquely inferred structure.

        Args:
            media_dir: Directory containing media files with labels in filenames
            extensions: Lowercase suffixes that count as media files
            explicit_pattern: Dataset-derived regex with one target capture group

        Returns:
            Tuple of (ids, labels, paths) where:
            - ids: List of file stems
            - labels: Target values extracted from filenames
            - paths: List of file paths
        """
        media_files = [
            path
            for path in media_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ]
        label_table = infer_filename_label_table(
            media_files,
            explicit_pattern=explicit_pattern,
        )
        return (
            label_table["record_id"].tolist(),
            label_table["target"].tolist(),
            [Path(path) for path in label_table["file_path"]],
        )

    def _detect_audio_labels_from_filenames(
        self,
        audio_dir: Path,
        explicit_pattern: str | None = None,
    ) -> tuple[list[str], list[str], list[Path]]:
        """Audio entry point for filename-derived labels."""
        return self._detect_media_labels_from_filenames(
            audio_dir,
            self.AUDIO_LABEL_EXTENSIONS,
            explicit_pattern=explicit_pattern,
        )

    def create_canonical_from_audio_filenames(
        self,
        audio_dir: Path,
        canonical_dir: Path,
        n_folds: int = 5,
        explicit_pattern: str | None = None,
        test_ids: list[str] | None = None,
    ) -> dict:
        """Audio entry point for filename-derived canonical artifacts."""
        return self.create_canonical_from_media_filenames(
            audio_dir,
            canonical_dir,
            extensions=self.AUDIO_LABEL_EXTENSIONS,
            source="audio_filenames",
            n_folds=n_folds,
            explicit_pattern=explicit_pattern,
            test_ids=test_ids,
        )

    def create_canonical_from_image_filenames(
        self,
        image_dir: Path,
        canonical_dir: Path,
        n_folds: int = 5,
        explicit_pattern: str | None = None,
        test_ids: list[str] | None = None,
    ) -> dict:
        """Image entry point for filename-derived canonical artifacts.

        Image competitions whose labels live in filenames or class directories
        have no train.csv, and without this the whole canonical contract was
        skipped: no injected CANONICAL_* constants, no y.npy, and therefore no
        independently recomputed OOF score for any component.
        """
        return self.create_canonical_from_media_filenames(
            image_dir,
            canonical_dir,
            extensions=self.IMAGE_LABEL_EXTENSIONS,
            source="image_filenames",
            n_folds=n_folds,
            explicit_pattern=explicit_pattern,
            test_ids=test_ids,
        )

    def create_canonical_from_media_filenames(  # noqa: PLR0913
        self,
        media_dir: Path,
        canonical_dir: Path,
        *,
        extensions: frozenset[str],
        source: str,
        n_folds: int = 5,
        explicit_pattern: str | None = None,
        test_ids: list[str] | None = None,
    ) -> dict:
        """Own the canonical directory it writes, from a clean slate.

        This producer is only invoked when no usable contract exists (no
        train.csv, or a prep that resolved zero rows), so anything already in
        ``canonical/`` is unusable by definition. Rebuilding from scratch
        prevents two poisoned outcomes: stale files from an earlier failed
        prep (a temporal mask, old test IDs) surviving next to fresh arrays
        and failing every component's shape checks, and a refusal or crash
        mid-write leaving a partial tree that makes the executor's integrity
        gate refuse ALL generated-code execution.
        """
        canonical_dir = Path(canonical_dir)
        if canonical_dir.exists():
            shutil.rmtree(canonical_dir)
        try:
            result = self._create_canonical_from_media_filenames_impl(
                media_dir,
                canonical_dir,
                extensions=extensions,
                source=source,
                n_folds=n_folds,
                explicit_pattern=explicit_pattern,
                test_ids=test_ids,
            )
        except BaseException:
            shutil.rmtree(canonical_dir, ignore_errors=True)
            raise
        if not result.get("success"):
            shutil.rmtree(canonical_dir, ignore_errors=True)
        return result

    def _create_canonical_from_media_filenames_impl(
        self,
        media_dir: Path,
        canonical_dir: Path,
        *,
        extensions: frozenset[str],
        source: str,
        n_folds: int = 5,
        explicit_pattern: str | None = None,
        test_ids: list[str] | None = None,
    ) -> dict:
        """Create canonical artifacts from evidence-backed filename targets.

        This is a fallback when no train.csv exists. The artifacts written here
        must satisfy the same contract the generated-code header loads, or the
        header is not injected at all and every candidate program raises
        NameError on the CANONICAL_* constants the prompts tell it to use.

        Args:
            media_dir: Directory containing files with labels in their names
            canonical_dir: Directory to save canonical artifacts
            extensions: Lowercase suffixes that count as media files
            source: Provenance label recorded in the metadata
            n_folds: Number of CV folds to create
            explicit_pattern: Dataset-derived regex with one target capture group
            test_ids: Graded test row IDs, in submission-template order

        Returns:
            Dictionary with canonical data info
        """
        from sklearn.model_selection import StratifiedKFold

        try:
            ids, labels, _paths = self._detect_media_labels_from_filenames(
                media_dir,
                extensions,
                explicit_pattern=explicit_pattern,
            )
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        if not ids:
            return {
                "success": False,
                "error": "No evidence-backed filename targets were found",
            }

        # Create canonical directory
        canonical_dir.mkdir(parents=True, exist_ok=True)

        # Convert to numpy arrays
        train_ids = np.array(ids)
        y = np.array(labels)

        # Create as many stratified folds as every observed class supports.
        class_counts = pd.Series(y).value_counts()
        effective_n_folds = min(n_folds, int(class_counts.min()))
        if effective_n_folds < 2:
            return {
                "success": False,
                "error": "At least two samples per inferred target are required for CV",
            }
        run_seed = get_run_seed()
        skf = StratifiedKFold(
            n_splits=effective_n_folds,
            shuffle=True,
            random_state=run_seed,
        )
        folds = np.zeros(len(train_ids), dtype=int)
        for fold_idx, (_, val_idx) in enumerate(skf.split(train_ids, y)):
            folds[val_idx] = fold_idx

        # Save artifacts. IDs stay str dtype (not object) so candidate code
        # can re-save them with allow_pickle=False.
        if train_ids.dtype == object:
            train_ids = np.asarray([str(v) for v in train_ids])
        np.save(canonical_dir / "train_ids.npy", train_ids, allow_pickle=False)
        np.save(canonical_dir / "y.npy", y)
        np.save(canonical_dir / "folds.npy", folds)

        # The media itself is the feature, so the tabular feature list is
        # empty - but the file must exist, because the generated-code header is
        # only injected when the whole canonical file set is present.
        import json

        feature_cols_path = canonical_dir / "feature_cols.json"
        feature_cols_path.write_text(json.dumps([]), encoding="utf-8")

        normalized_test_ids = [str(value) for value in (test_ids or [])]
        test_ids_path: Path | None = None
        if normalized_test_ids:
            test_ids_path = canonical_dir / "test_ids.npy"
            np.save(
                test_ids_path,
                np.asarray(normalized_test_ids, dtype=str),
                allow_pickle=False,
            )

        class_order = [str(value) for value in np.unique(y).tolist()]
        metadata = {
            "canonical_rows": len(train_ids),
            "n_folds": effective_n_folds,
            "requested_n_folds": n_folds,
            "id_col": "record_id",
            "id_is_synthetic": False,
            "target_col": "target",
            # Fields the injected header validates as required. Omitting them
            # made the header raise instead of loading the contract.
            "target_cols": ["target"],
            "target_type": "single",
            "n_targets": 1,
            "n_features": 0,
            "is_classification": True,
            "num_classes": len(class_order),
            "n_classes": len(class_order),
            "class_order": class_order,
            "cv_strategy": "stratified_kfold",
            "n_test": len(normalized_test_ids),
            "random_seed": run_seed,
            "target_source": (
                "explicit_filename_pattern"
                if explicit_pattern
                else "unique_filename_structure"
            ),
            "source": source,
        }
        with open(canonical_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"   [FALLBACK] Created canonical data from {len(ids)} media files")
        print(f"   Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

        return {
            "success": True,
            "canonical_dir": str(canonical_dir),
            "train_ids_path": str(canonical_dir / "train_ids.npy"),
            "y_path": str(canonical_dir / "y.npy"),
            "folds_path": str(canonical_dir / "folds.npy"),
            "feature_cols_path": str(feature_cols_path),
            "metadata_path": str(canonical_dir / "metadata.json"),
            "test_ids_path": str(test_ids_path) if test_ids_path else None,
            "metadata": metadata,
        }
