"""
Precomputed features detection for audio competitions.

Detects available precomputed features in competition data directories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class PrecomputedFeaturesInfo:
    """Information about detected precomputed features."""

    features_found: dict[str, Path] = field(default_factory=dict)
    feature_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    feature_columns: dict[str, list[str]] = field(default_factory=dict)
    total_features: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "features_found": {k: str(v) for k, v in self.features_found.items()},
            "feature_shapes": self.feature_shapes,
            "feature_columns": self.feature_columns,
            "total_features": self.total_features,
            "warnings": self.warnings,
        }

    def has_features(self) -> bool:
        """Check if any precomputed features were found."""
        return len(self.features_found) > 0


def detect_precomputed_features(
    data_dir: Path | str,
    extensions: list[str] | None = None,
    recursive: bool = True,
) -> PrecomputedFeaturesInfo:
    """
    Detect available precomputed features in competition data directory.

    Searches recursively for numeric feature matrices and public split/mapping
    metadata. Metadata roles are also inferred from their columns so support
    does not depend on a task-specific filename.

    Args:
        data_dir: Directory to search for feature files
        extensions: File extensions to consider (default: ['.txt', '.csv', '.npy', '.npz'])
        recursive: Whether to search subdirectories

    Returns:
        PrecomputedFeaturesInfo with detected features

    Example:
        A supplied data bundle may contain numeric feature matrices, a fold
        assignment table, and an ID-to-file mapping.
    """
    data_dir = Path(data_dir)
    if extensions is None:
        extensions = [".txt", ".tsv", ".csv", ".npy", ".npz", ".parquet", ".feather"]

    features_found: dict[str, Path] = {}
    feature_shapes: dict[str, tuple[int, ...]] = {}
    feature_columns: dict[str, list[str]] = {}
    warnings: list[str] = []

    if not data_dir.exists():
        warnings.append(f"Data directory not found: {data_dir}")
        return PrecomputedFeaturesInfo(warnings=warnings)

    # Search for feature files
    if recursive:
        all_files = list(data_dir.rglob("*"))
    else:
        all_files = list(data_dir.glob("*"))

    # Filter by extension
    candidate_files = [
        f for f in all_files
        if f.is_file() and f.suffix.lower() in extensions
    ]

    def infer_metadata_role(file_path: Path) -> str | None:
        if file_path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
            return None
        try:
            preview = pd.read_csv(file_path, sep=None, engine="python", nrows=20)
        except Exception:
            return None
        normalized_columns = [
            str(column).strip().lower().replace("-", "_") for column in preview.columns
        ]
        if any("fold" in column or "split" in column for column in normalized_columns):
            return "cv_folds"
        has_id = any(
            column == "id" or column.endswith("_id") or column.startswith("id_")
            for column in normalized_columns
        )
        has_file = any(
            "file" in column or "path" in column for column in normalized_columns
        )
        if has_id and has_file:
            return "id_mapping"
        return None

    def infer_feature_key(file_path: Path) -> str | None:
        normalized_stem = "".join(
            character if character.isalnum() else "_"
            for character in file_path.stem.lower()
        ).strip("_")
        excluded_tokens = {
            "train",
            "test",
            "sample",
            "label",
            "labels",
            "target",
            "targets",
            "annotation",
            "annotations",
        }
        stem_tokens = set(normalized_stem.split("_"))
        if stem_tokens & excluded_tokens:
            return None

        shape, columns = _get_feature_info(file_path)
        if not shape or len(shape) < 2 or shape[1] < 2:
            return None

        if file_path.suffix.lower() in {".csv", ".txt", ".tsv", ".parquet", ".feather"}:
            try:
                if file_path.suffix.lower() == ".parquet":
                    preview = pd.read_parquet(file_path).head(20)
                elif file_path.suffix.lower() == ".feather":
                    preview = pd.read_feather(file_path).head(20)
                else:
                    preview = pd.read_csv(file_path, sep=None, engine="python", nrows=20)
            except Exception:
                return None
            numeric_columns = sum(
                pd.api.types.is_numeric_dtype(preview[column])
                for column in preview.columns
            )
            has_feature_marker = "feature" in normalized_stem or "embedding" in normalized_stem
            if not has_feature_marker and (
                len(preview.columns) < 3
                or numeric_columns < max(2, len(preview.columns) - 1)
            ):
                return None

        key = normalized_stem
        for suffix in ("_features", "_feature", "_embeddings", "_embedding"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
                break
        if not key:
            key = "numeric_features"

        candidate_key = key
        counter = 2
        while candidate_key in features_found:
            candidate_key = f"{key}_{counter}"
            counter += 1
        return candidate_key

    # Infer metadata and feature matrices from the public file schemas.
    for file_path in candidate_files:
        metadata_role = infer_metadata_role(file_path)
        if metadata_role and metadata_role not in features_found:
            features_found[metadata_role] = file_path
            shape, columns = _get_feature_info(file_path)
            if shape:
                feature_shapes[metadata_role] = shape
            if columns:
                feature_columns[metadata_role] = columns
            continue

        feature_type = infer_feature_key(file_path)
        if feature_type is None:
            continue
        features_found[feature_type] = file_path
        shape, columns = _get_feature_info(file_path)
        if shape:
            feature_shapes[feature_type] = shape
        if columns:
            feature_columns[feature_type] = columns

    # Calculate total feature dimensions
    total_features = 0
    for feature_type, shape in feature_shapes.items():
        if len(shape) >= 2 and feature_type not in ("cv_folds", "id_mapping"):
            total_features += shape[1]

    return PrecomputedFeaturesInfo(
        features_found=features_found,
        feature_shapes=feature_shapes,
        feature_columns=feature_columns,
        total_features=total_features,
        warnings=warnings,
    )


def _get_feature_info(file_path: Path) -> tuple[tuple[int, ...] | None, list[str] | None]:
    """
    Get shape and column info for a feature file.

    Args:
        file_path: Path to feature file

    Returns:
        (shape, columns) tuple
    """
    shape = None
    columns = None

    try:
        if file_path.suffix.lower() in (".csv", ".txt", ".tsv"):
            # Try to read first few rows to get shape
            df = pd.read_csv(file_path, sep=None, engine="python", nrows=5)
            # Get full row count by counting lines
            with open(file_path) as f:
                num_rows = sum(1 for _ in f) - 1  # Subtract header
            shape = (num_rows, len(df.columns))
            columns = list(df.columns)

        elif file_path.suffix.lower() == ".npy":
            import numpy as np
            arr = np.load(file_path, mmap_mode='r')
            shape = arr.shape

        elif file_path.suffix.lower() == ".npz":
            import numpy as np
            with np.load(file_path) as data:
                # Get first array's shape
                for key in data.files:
                    shape = data[key].shape
                    break

        elif file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
            shape = df.shape
            columns = list(df.columns)

        elif file_path.suffix.lower() == ".feather":
            df = pd.read_feather(file_path)
            shape = df.shape
            columns = list(df.columns)

    except Exception:
        pass

    return shape, columns


def load_precomputed_features(
    features_info: PrecomputedFeaturesInfo,
    feature_types: list[str] | None = None,
    id_column: str | None = None,
) -> pd.DataFrame | None:
    """
    Load precomputed features into a DataFrame.

    Args:
        features_info: PrecomputedFeaturesInfo from detect_precomputed_features()
        feature_types: List of feature types to load (default: all except cv_folds, id_mapping)
        id_column: Name of ID column to use as index

    Returns:
        DataFrame with all requested features, or None if no features found
    """
    if not features_info.has_features():
        return None

    if feature_types is None:
        feature_types = [
            ft for ft in features_info.features_found.keys()
            if ft not in ("cv_folds", "id_mapping")
        ]

    dfs = []
    for ft in feature_types:
        if ft not in features_info.features_found:
            continue

        file_path = features_info.features_found[ft]
        try:
            if file_path.suffix.lower() in (".csv", ".txt", ".tsv"):
                df = pd.read_csv(file_path, sep=None, engine="python")
            elif file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".feather":
                df = pd.read_feather(file_path)
            elif file_path.suffix.lower() == ".npy":
                import numpy as np
                arr = np.load(file_path)
                df = pd.DataFrame(arr, columns=[f"{ft}_{i}" for i in range(arr.shape[1])])
            else:
                continue

            # Prefix columns with feature type
            df.columns = [f"{ft}_{col}" if col != id_column else col for col in df.columns]
            dfs.append(df)

        except Exception as e:
            print(f"Warning: Failed to load {ft} features: {e}")

    if not dfs:
        return None

    # Merge all feature DataFrames
    result = dfs[0]
    for df in dfs[1:]:
        if id_column and id_column in result.columns and id_column in df.columns:
            result = result.merge(df, on=id_column, how='outer')
        else:
            # Assume same row order
            result = pd.concat([result, df], axis=1)

    return result


def generate_feature_loading_code(features_info: PrecomputedFeaturesInfo) -> str:
    """
    Generate code snippet for loading detected precomputed features.

    Generates appropriate loading code based on file extension:
    - .csv, .txt, .tsv: pd.read_csv()
    - .parquet: pd.read_parquet()
    - .feather: pd.read_feather()
    - .npy: np.load()
    - .npz: np.load()

    Args:
        features_info: PrecomputedFeaturesInfo from detect_precomputed_features()

    Returns:
        Python code snippet for loading features
    """
    if not features_info.has_features():
        return "# No precomputed features detected"

    # Check which imports are needed
    needs_numpy = False
    needs_pandas = False

    for ft, path in features_info.features_found.items():
        if ft in ("cv_folds", "id_mapping"):
            continue
        ext = path.suffix.lower()
        if ext in (".npy", ".npz"):
            needs_numpy = True
        else:
            needs_pandas = True

    code_lines = ["# Load precomputed features"]
    if needs_pandas:
        code_lines.append("import pandas as pd")
    if needs_numpy:
        code_lines.append("import numpy as np")
    code_lines.append("from pathlib import Path")
    code_lines.append("")

    for ft, path in features_info.features_found.items():
        if ft in ("cv_folds", "id_mapping"):
            continue

        shape_str = ""
        if ft in features_info.feature_shapes:
            shape_str = f"  # Shape: {features_info.feature_shapes[ft]}"

        ext = path.suffix.lower()
        code_lines.append(f"# {ft.title()} features{shape_str}")

        if ext == ".parquet":
            code_lines.append(f"{ft}_df = pd.read_parquet(Path('{path}'))")
        elif ext == ".npy":
            code_lines.append(f"{ft}_arr = np.load(Path('{path}'))")
            code_lines.append("# Convert to DataFrame if needed:")
            code_lines.append(f"# {ft}_df = pd.DataFrame({ft}_arr, columns=[f'{ft}_{{i}}' for i in range({ft}_arr.shape[1])])")
        elif ext == ".npz":
            code_lines.append(f"{ft}_data = np.load(Path('{path}'))")
            code_lines.append(f"# Access arrays via: {ft}_data.files, {ft}_data['array_name']")
        elif ext == ".feather":
            code_lines.append(f"{ft}_df = pd.read_feather(Path('{path}'))")
        else:
            # Auto-detect delimiters for text tables.
            code_lines.append(
                f"{ft}_df = pd.read_csv(Path('{path}'), sep=None, engine='python')"
            )

        code_lines.append("")

    return "\n".join(code_lines)


def print_features_info(features_info: PrecomputedFeaturesInfo) -> None:
    """Print formatted precomputed features information."""
    print("\n" + "=" * 60)
    print("=== PRECOMPUTED FEATURES DETECTION ===")
    print("=" * 60)

    if not features_info.has_features():
        print("No precomputed features found.")
        if features_info.warnings:
            print("\nWarnings:")
            for w in features_info.warnings:
                print(f"  - {w}")
        print("=" * 60 + "\n")
        return

    print(f"Total feature types found: {len(features_info.features_found)}")
    print(f"Total feature dimensions: {features_info.total_features}")

    print("\nFeatures:")
    for ft, path in features_info.features_found.items():
        shape_str = ""
        if ft in features_info.feature_shapes:
            shape_str = f" ({features_info.feature_shapes[ft]})"
        print(f"  - {ft}: {path.name}{shape_str}")

    if features_info.warnings:
        print("\nWarnings:")
        for w in features_info.warnings:
            print(f"  - {w}")

    print("=" * 60 + "\n")
