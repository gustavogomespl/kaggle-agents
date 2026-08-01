"""Safe packed artifacts for variable-sized image-to-image predictions."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image

from .csv_utils import detect_delimiter
from .target_inference import infer_pixel_submission_schema


_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
_PACKED_KEYS = {"values", "offsets", "shapes", "image_ids"}
_MAX_PACKED_MEMBER_BYTES = 2 * 1024**3
_PACKED_IMAGE_RANKS = {2, 3}


@dataclass(frozen=True)
class PackedImages:
    """Validated variable-sized images stored without Python objects."""

    values: np.ndarray
    offsets: np.ndarray
    shapes: np.ndarray
    image_ids: np.ndarray

    def image(self, index: int) -> np.ndarray:
        """Return one image reshaped from its packed slice."""
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return self.values[start:stop].reshape(tuple(self.shapes[index]))

    def __len__(self) -> int:
        return int(self.image_ids.shape[0])


def validate_packed_images(  # noqa: PLR0912
    values: np.ndarray,
    offsets: np.ndarray,
    shapes: np.ndarray,
    image_ids: np.ndarray,
) -> PackedImages:
    """Validate and return the pickle-free packed image representation."""
    values = np.asarray(values)
    offsets = np.asarray(offsets)
    shapes = np.asarray(shapes)
    image_ids = np.asarray(image_ids)

    if values.dtype != np.dtype(np.float32) or values.ndim != 1:
        raise ValueError("values must be a 1-D float32 array")
    if not np.all(np.isfinite(values)):
        raise ValueError("packed image values contain NaN or Inf")
    if offsets.dtype != np.dtype(np.int64) or offsets.ndim != 1:
        raise ValueError("offsets must be a 1-D int64 array")
    if shapes.dtype != np.dtype(np.int32) or shapes.ndim != 2:
        raise ValueError("shapes must be a 2-D int32 array")
    if shapes.shape[1] not in _PACKED_IMAGE_RANKS:
        raise ValueError("packed images must have rank 2 or 3")
    if image_ids.ndim != 1 or image_ids.dtype.kind != "U":
        raise ValueError("image_ids must be a 1-D unicode array")

    n_images = int(image_ids.shape[0])
    if shapes.shape[0] != n_images:
        raise ValueError(
            f"shape row count {shapes.shape[0]} does not match image ID count {n_images}"
        )
    if offsets.shape[0] != n_images + 1:
        raise ValueError(f"offset count {offsets.shape[0]} must equal image count + 1")
    if offsets.size == 0 or int(offsets[0]) != 0:
        raise ValueError("offsets must start at zero")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("offsets must be monotonically non-decreasing")
    if int(offsets[-1]) != int(values.size):
        raise ValueError(
            f"final offset {int(offsets[-1])} does not match values length {values.size}"
        )
    if shapes.size and np.any(shapes <= 0):
        raise ValueError("all packed image dimensions must be positive")
    # Python integers make overflow impossible. np.prod(..., int64) can wrap a
    # hostile declared shape to zero and make an empty artifact look valid.
    element_counts = np.asarray(
        [
            int(np.prod([int(dimension) for dimension in shape], dtype=object))
            for shape in shapes
        ],
        dtype=object,
    )
    if not np.array_equal(element_counts, np.diff(offsets)):
        raise ValueError("packed offsets do not match declared image shapes")

    normalized_ids = image_ids.tolist()
    if any(not value for value in normalized_ids):
        raise ValueError("image IDs must be non-empty")
    if len(set(normalized_ids)) != n_images:
        raise ValueError("packed artifact contains duplicate image IDs")

    return PackedImages(
        values=values,
        offsets=offsets,
        shapes=shapes,
        image_ids=image_ids,
    )


def save_packed_images(
    path: str | Path,
    images: Iterable[np.ndarray],
    *,
    image_ids: Sequence[object],
) -> Path:
    """Save variable-sized numeric images as one safe ``.npz`` artifact."""
    path = Path(path)
    arrays = [np.asarray(image) for image in images]
    ids = [str(value) for value in image_ids]
    if len(arrays) != len(ids):
        raise ValueError(f"image count {len(arrays)} does not match image ID count {len(ids)}")
    if not arrays:
        raise ValueError("cannot pack an empty image collection")
    rank = arrays[0].ndim
    if rank == 0 or any(array.ndim != rank for array in arrays):
        raise ValueError("all packed images must have the same positive rank")

    float_images = [np.asarray(array, dtype=np.float32) for array in arrays]
    values = np.concatenate([array.reshape(-1) for array in float_images])
    sizes = np.asarray([array.size for array in float_images], dtype=np.int64)
    offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(sizes, dtype=np.int64)])
    shapes = np.asarray([array.shape for array in float_images], dtype=np.int32)
    image_ids_array = np.asarray(ids, dtype=str)
    packed = validate_packed_images(values, offsets, shapes, image_ids_array)

    if path.suffix != ".npz":
        raise ValueError("packed image artifact path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        values=packed.values,
        offsets=packed.offsets,
        shapes=packed.shapes,
        image_ids=packed.image_ids,
    )
    return path


def load_packed_images(path: str | Path) -> PackedImages:
    """Load and validate a packed image artifact with pickle disabled."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as zipped:
            members = zipped.infolist()
            expected_members = {f"{key}.npy" for key in _PACKED_KEYS}
            member_names = [member.filename for member in members]
            if (
                set(member_names) != expected_members
                or len(member_names) != len(expected_members)
            ):
                raise ValueError(
                    "packed image archive members do not match the contract"
                )
            total_uncompressed = sum(member.file_size for member in members)
            if any(
                member.file_size > _MAX_PACKED_MEMBER_BYTES
                for member in members
            ) or total_uncompressed > _MAX_PACKED_MEMBER_BYTES:
                raise ValueError(
                    "packed image archive exceeds the safe uncompressed size limit"
                )
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != _PACKED_KEYS:
                missing = sorted(_PACKED_KEYS - keys)
                extra = sorted(keys - _PACKED_KEYS)
                raise ValueError(f"packed image keys mismatch: missing={missing}, extra={extra}")
            arrays = {name: np.asarray(archive[name]) for name in _PACKED_KEYS}
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"failed to load packed image artifact: {exc}") from exc
    return validate_packed_images(**arrays)


def _packed_image_aliases(image_id: str) -> set[str]:
    """Return only lossless/common template aliases for a canonical image ID."""
    normalized = str(image_id)
    path = PurePosixPath(normalized)
    aliases = {normalized, path.as_posix(), path.name}
    if path.suffix:
        aliases.add(path.with_suffix("").as_posix())
        aliases.add(path.stem)
    return {alias for alias in aliases if alias}


def write_packed_image_submission(
    *,
    packed_predictions_path: str | Path,
    sample_submission_path: str | Path,
    output_path: str | Path,
    target_cols: Sequence[str],
    id_col: str | None = None,
    chunk_rows: int = 100_000,
) -> Path:
    """Map packed 2-D images to observed ``image_row_col`` IDs and stream CSV.

    The mapping is accepted only when the public template proves one
    coordinate-ID column, a unique zero- or one-based coordinate convention,
    and exact one-to-one coverage of every pixel in the packed artifact.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    packed = load_packed_images(packed_predictions_path)
    declared_targets = [str(column) for column in target_cols]
    if len(declared_targets) != 1:
        raise ValueError(
            "Packed image submission currently requires exactly one prediction column"
        )
    if packed.values.size and (
        float(packed.values.min()) < 0.0
        or float(packed.values.max()) > 1.0
    ):
        raise ValueError("Packed image predictions must use the [0, 1] scale")

    image_hw: list[tuple[int, int]] = []
    for shape in packed.shapes:
        dimensions = tuple(int(value) for value in shape)
        if len(dimensions) == 2:
            height, width = dimensions
        elif len(dimensions) == 3 and dimensions[2] == 1:
            height, width = dimensions[:2]
        else:
            raise ValueError(
                "Pixel-ID submissions require HxW or HxWx1 packed predictions"
            )
        image_hw.append((height, width))

    alias_to_index: dict[str, int | None] = {}
    for index, image_id in enumerate(packed.image_ids.astype(str)):
        for alias in _packed_image_aliases(image_id):
            previous = alias_to_index.get(alias)
            if previous is None and alias in alias_to_index:
                continue
            if previous is not None and previous != index:
                alias_to_index[alias] = None
            else:
                alias_to_index[alias] = index

    sample_path = Path(sample_submission_path)
    output = Path(output_path)
    sample_delimiter = detect_delimiter(sample_path)
    preview = pd.read_csv(
        sample_path,
        sep=sample_delimiter,
        nrows=200,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    missing_targets = [
        column for column in declared_targets if column not in preview.columns
    ]
    if missing_targets:
        raise ValueError(
            f"Submission template is missing prediction columns: {missing_targets}"
        )
    resolved_id_col = str(id_col) if id_col in preview.columns else None
    if resolved_id_col is None:
        inferred_roles = infer_pixel_submission_schema(preview)
        if inferred_roles is None:
            raise ValueError(
                "Could not prove a unique pixel-coordinate ID column from the template"
            )
        resolved_id_col = inferred_roles[0][0]
    if resolved_id_col in declared_targets:
        raise ValueError("Pixel ID column cannot also be a prediction column")

    def parse_pixel_id(value: object) -> tuple[int, int, int]:
        parts = str(value).rsplit("_", 2)
        if (
            len(parts) != 3
            or not parts[0]
            or not parts[1].isdigit()
            or not parts[2].isdigit()
        ):
            raise ValueError(
                f"Template ID does not match observed image_row_col form: {value!r}"
            )
        image_index = alias_to_index.get(parts[0])
        if image_index is None:
            if parts[0] in alias_to_index:
                raise ValueError(
                    f"Template image alias is ambiguous: {parts[0]!r}"
                )
            raise ValueError(
                f"Template image ID has no packed prediction: {parts[0]!r}"
            )
        return image_index, int(parts[1]), int(parts[2])

    possible_bases = {0, 1}
    template_rows = 0
    first_pass = pd.read_csv(
        sample_path,
        sep=sample_delimiter,
        chunksize=chunk_rows,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    try:
        for chunk in first_pass:
            template_rows += len(chunk)
            for value in chunk[resolved_id_col]:
                image_index, row, column = parse_pixel_id(value)
                height, width = image_hw[image_index]
                possible_bases = {
                    base
                    for base in possible_bases
                    if 0 <= row - base < height
                    and 0 <= column - base < width
                }
                if not possible_bases:
                    raise ValueError(
                        f"Pixel coordinate is outside its packed image: {value!r}"
                    )
    finally:
        first_pass.close()

    expected_rows = int(
        sum(height * width for height, width in image_hw)
    )
    if template_rows != expected_rows:
        raise ValueError(
            f"Template has {template_rows} pixel rows but packed predictions "
            f"contain {expected_rows} pixels"
        )
    if len(possible_bases) != 1:
        raise ValueError(
            "Could not prove a unique zero- or one-based coordinate convention"
        )
    coordinate_base = possible_bases.pop()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        dir=str(output.parent),
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    seen_pixels = np.zeros(packed.values.size, dtype=bool)
    wrote_header = False
    second_pass = pd.read_csv(
        sample_path,
        sep=sample_delimiter,
        chunksize=chunk_rows,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    try:
        for chunk in second_pass:
            predictions = np.empty(len(chunk), dtype=np.float32)
            for offset, value in enumerate(chunk[resolved_id_col]):
                image_index, row, column = parse_pixel_id(value)
                height, width = image_hw[image_index]
                flat_index = (
                    int(packed.offsets[image_index])
                    + (row - coordinate_base) * width
                    + (column - coordinate_base)
                )
                if seen_pixels[flat_index]:
                    raise ValueError(f"Duplicate pixel ID in template: {value!r}")
                seen_pixels[flat_index] = True
                predictions[offset] = packed.values[flat_index]
            chunk[declared_targets[0]] = predictions
            chunk.to_csv(
                temporary,
                mode="a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
        if not bool(seen_pixels.all()):
            missing = int((~seen_pixels).sum())
            raise ValueError(
                f"Template does not cover {missing} packed prediction pixels"
            )
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        second_pass.close()
    return output


def packed_image_rmse(
    prediction_path: str | Path,
    target_path: str | Path,
) -> float:
    """Compute trusted pixel RMSE after exact ID/shape/offset alignment."""
    predictions = load_packed_images(prediction_path)
    targets = load_packed_images(target_path)
    if not np.array_equal(predictions.image_ids, targets.image_ids):
        raise ValueError("prediction image IDs do not match canonical target order")
    if not np.array_equal(predictions.shapes, targets.shapes):
        raise ValueError("prediction image shapes do not match canonical targets")
    if not np.array_equal(predictions.offsets, targets.offsets):
        raise ValueError("prediction offsets do not match canonical targets")
    difference = predictions.values.astype(np.float64) - targets.values.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def validate_image_fold_assignments(
    folds: np.ndarray,
    image_ids: np.ndarray,
) -> np.ndarray:
    """Validate that there is exactly one scalar fold assignment per image."""
    folds = np.asarray(folds)
    image_ids = np.asarray(image_ids).reshape(-1)
    if folds.ndim != 1:
        raise ValueError(f"Image folds must be a 1-D list, got shape {folds.shape}")
    if folds.shape[0] != image_ids.shape[0]:
        raise ValueError(
            f"Fold assignment count {folds.shape[0]} does not match image ID "
            f"count {image_ids.shape[0]}"
        )
    if not np.issubdtype(folds.dtype, np.integer) or np.any(folds < 0):
        raise ValueError("Image folds must contain non-negative integers")
    return folds


def _relative_image_paths(directory: Path) -> dict[str, Path]:
    return {
        path.relative_to(directory).as_posix(): path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    }


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).copy()


def _normalize_target_image(image: np.ndarray) -> np.ndarray:
    """Normalize canonical target pixels to the competition's unit scale."""
    image = np.asarray(image)
    if image.dtype == np.bool_:
        return image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        if image.size and int(image.min()) < 0:
            raise ValueError(
                "Signed integer clean targets contain negative pixel values"
            )
        maximum = float(np.iinfo(image.dtype).max)
        normalized = image.astype(np.float32) / maximum
    else:
        normalized = image.astype(np.float32)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Clean target image contains NaN or Inf")
    if normalized.size and (
        float(normalized.min()) < 0.0 or float(normalized.max()) > 1.0
    ):
        raise ValueError(
            "Floating-point clean targets must already use the [0, 1] scale"
        )
    return normalized


def prepare_image_to_image_canonical_data(
    *,
    noisy_dir: str | Path,
    clean_dir: str | Path,
    test_dir: str | Path | None = None,
    output_dir: str | Path,
    n_folds: int = 5,
) -> dict[str, object]:
    """Pair noisy/clean images and write canonical targets, atomically enough.

    A contract with zero test rows is refused up front: it satisfies the
    executor's file-presence integrity gate and then kills every component
    with "Packed image evidence cannot be empty" — a poisoned success. And a
    failure part-way through must not leave a partial ``canonical/`` behind,
    or that same gate refuses ALL generated-code execution for the rest of
    the run (the orphan-directory class of bug).
    """
    output_dir = Path(output_dir)
    canonical_dir = output_dir / "canonical"
    contract_was_complete = (canonical_dir / "metadata.json").is_file()
    try:
        return _prepare_image_to_image_canonical_data_impl(
            noisy_dir=noisy_dir,
            clean_dir=clean_dir,
            test_dir=test_dir,
            output_dir=output_dir,
            n_folds=n_folds,
        )
    except BaseException:
        if not contract_was_complete:
            shutil.rmtree(canonical_dir, ignore_errors=True)
        raise


def _prepare_image_to_image_canonical_data_impl(
    *,
    noisy_dir: str | Path,
    clean_dir: str | Path,
    test_dir: str | Path | None,
    output_dir: Path,
    n_folds: int,
) -> dict[str, object]:
    """Pair noisy/clean images by relative path and write canonical targets."""
    noisy_dir = Path(noisy_dir)
    clean_dir = Path(clean_dir)
    if not noisy_dir.is_dir():
        raise ValueError(f"Noisy image directory does not exist: {noisy_dir}")
    if not clean_dir.is_dir():
        raise ValueError(f"Clean target directory does not exist: {clean_dir}")
    if test_dir is None or not Path(test_dir).is_dir():
        raise ValueError(
            "image-to-image canonical contract requires an existing test "
            "image directory (zero test rows cannot produce a submission): "
            f"{test_dir}"
        )
    test_by_id = _relative_image_paths(Path(test_dir))
    if not test_by_id:
        raise ValueError(f"No test images were found in: {test_dir}")

    noisy_by_id = _relative_image_paths(noisy_dir)
    clean_by_id = _relative_image_paths(clean_dir)
    noisy_ids = set(noisy_by_id)
    clean_ids = set(clean_by_id)
    if noisy_ids != clean_ids:
        missing_clean = sorted(noisy_ids - clean_ids)
        missing_noisy = sorted(clean_ids - noisy_ids)
        raise ValueError(
            "Image pair coverage mismatch; "
            f"missing clean targets={missing_clean}; "
            f"missing noisy inputs={missing_noisy}"
        )
    if not noisy_ids:
        raise ValueError("No paired image files were found")

    image_ids = sorted(noisy_ids)
    clean_images: list[np.ndarray] = []
    for image_id in image_ids:
        noisy_image = _load_image(noisy_by_id[image_id])
        clean_image = _load_image(clean_by_id[image_id])
        if noisy_image.shape != clean_image.shape:
            raise ValueError(
                f"Paired image shape mismatch for {image_id}: "
                f"noisy={noisy_image.shape}, clean={clean_image.shape}"
            )
        clean_images.append(_normalize_target_image(clean_image))

    canonical_dir = output_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    train_ids = np.asarray(image_ids, dtype=str)
    train_ids_path = canonical_dir / "train_ids.npy"
    np.save(train_ids_path, train_ids)
    folds_count = min(max(int(n_folds), 1), len(image_ids))
    folds = np.arange(len(image_ids), dtype=np.int64) % folds_count
    validate_image_fold_assignments(folds, train_ids)
    folds_path = canonical_dir / "folds.npy"
    np.save(folds_path, folds)
    y_path = save_packed_images(
        canonical_dir / "image_targets.npz",
        clean_images,
        image_ids=image_ids,
    )
    input_paths_path = canonical_dir / "image_input_paths.npy"
    np.save(
        input_paths_path,
        np.asarray([str(noisy_by_id[image_id]) for image_id in image_ids], dtype=str),
    )
    test_ids = sorted(test_by_id)
    test_ids_path = canonical_dir / "test_ids.npy"
    np.save(test_ids_path, np.asarray(test_ids, dtype=str), allow_pickle=False)
    test_input_paths_path = canonical_dir / "image_test_input_paths.npy"
    np.save(
        test_input_paths_path,
        np.asarray([str(test_by_id[image_id]) for image_id in test_ids], dtype=str),
        allow_pickle=False,
    )
    metadata = {
        "task_type": "image_to_image",
        "canonical_rows": len(image_ids),
        "n_folds": folds_count,
        "id_col": "image_id",
        "target_col": "image_pixels",
        "target_cols": ["image_pixels"],
        "target_type": "multi_target",
        "n_targets": 1,
        "packed_image_contract": True,
        "id_is_synthetic": False,
        "is_classification": False,
        "cv_strategy": "image_kfold",
        "n_test": len(test_ids),
        "target_value_range": [0.0, 1.0],
        "integer_pixel_normalization": "dtype_max",
    }
    metadata_path = canonical_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "canonical_dir": str(canonical_dir),
        "train_ids_path": str(train_ids_path),
        "y_path": str(y_path),
        "folds_path": str(folds_path),
        "image_input_paths_path": str(input_paths_path),
        "test_ids_path": str(test_ids_path),
        "image_test_input_paths_path": str(test_input_paths_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }
