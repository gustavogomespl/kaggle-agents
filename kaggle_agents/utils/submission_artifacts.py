"""Fail-closed lifecycle helpers for accepted submission artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_REBUILD_CHUNK_ROWS = 100_000


def rebuild_submission_from_component_predictions(
    *,
    working_dir: Path,
    component_name: str,
    sample_submission_path: Path,
    target_cols: list[str],
    id_col: str | None = None,
    test_ids_are_positional: bool = False,
) -> Path | None:
    """Rebuild a malformed submission from already validated test predictions.

    This deliberately does not execute generated code.  It only fills the
    template's explicitly resolved prediction roles, preserving every echoed
    column byte-for-value from the public template.  Component test IDs are
    used to restore template row order when both sides provide a real ID role.
    """
    import numpy as np

    if not component_name or not target_cols or not sample_submission_path.is_file():
        return None

    packed_predictions_path = (
        working_dir / "models" / f"test_{component_name}.npz"
    )
    if packed_predictions_path.is_file():
        try:
            from .image_to_image_contract import (
                write_packed_image_submission,
            )

            return write_packed_image_submission(
                packed_predictions_path=packed_predictions_path,
                sample_submission_path=sample_submission_path,
                output_path=working_dir / "submission.csv",
                target_cols=target_cols,
                id_col=id_col,
            )
        except (OSError, ValueError):
            return None

    predictions_path = working_dir / "models" / f"test_{component_name}.npy"
    if not predictions_path.is_file():
        return None
    try:
        from .csv_utils import detect_delimiter, read_csv_auto

        # The template is streamed rather than materialized: this path also
        # serves pixel-level templates with millions of rows. The rebuilt file
        # is written as a comma CSV, matching the injected writer's contract.
        template_sep = detect_delimiter(sample_submission_path)
        read_kwargs: dict[str, Any] = {
            "sep": template_sep,
            "dtype": str,
            "keep_default_na": False,
            "na_filter": False,
        }
        template_columns = [
            str(column)
            for column in read_csv_auto(
                sample_submission_path, nrows=0, **read_kwargs
            ).columns
        ]
        if any(column not in template_columns for column in target_cols):
            return None
        template_rows = sum(
            len(chunk)
            for chunk in read_csv_auto(
                sample_submission_path,
                chunksize=_REBUILD_CHUNK_ROWS,
                **read_kwargs,
            )
        )
        predictions = np.asarray(np.load(predictions_path, allow_pickle=False))
    except (OSError, ValueError):
        return None

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if (
        predictions.ndim == 2
        and len(target_cols) == 1
        and predictions.shape[1] > 1
    ):
        class_order_path = (
            working_dir / "models" / f"class_order_{component_name}.npy"
        )
        try:
            class_order = np.asarray(
                np.load(class_order_path, allow_pickle=False)
            ).reshape(-1)
        except (OSError, ValueError):
            return None
        if len(class_order) != predictions.shape[1]:
            return None
        predictions = np.asarray(
            [str(class_order[index]) for index in np.argmax(predictions, axis=1)]
        ).reshape(-1, 1)
    if predictions.ndim != 2 or predictions.shape != (
        template_rows,
        len(target_cols),
    ):
        return None

    test_ids_path = working_dir / "models" / f"test_ids_{component_name}.npy"
    if test_ids_path.is_file():
        try:
            test_ids = np.asarray(np.load(test_ids_path, allow_pickle=False)).reshape(-1)
        except (OSError, ValueError):
            return None
        prediction_ids = [str(value) for value in test_ids]
        if len(prediction_ids) != len(predictions):
            return None
        positional_ids = [str(index) for index in range(len(predictions))]
        if test_ids_are_positional:
            if prediction_ids != positional_ids:
                return None
        else:
            if not id_col or id_col not in template_columns:
                return None
            template_ids = [
                str(value)
                for chunk in read_csv_auto(
                    sample_submission_path,
                    usecols=[id_col],
                    chunksize=_REBUILD_CHUNK_ROWS,
                    **read_kwargs,
                )
                for value in chunk[id_col].tolist()
            ]
            if (
                len(set(prediction_ids)) != len(prediction_ids)
                or len(set(template_ids)) != len(template_ids)
                or set(prediction_ids) != set(template_ids)
            ):
                return None
            positions = {value: index for index, value in enumerate(prediction_ids)}
            predictions = predictions[[positions[value] for value in template_ids]]

    destination = working_dir / "submission.csv"
    temporary = working_dir / f".submission-rebuild-{uuid.uuid4().hex}.tmp"
    written = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            for index, chunk in enumerate(
                read_csv_auto(
                    sample_submission_path,
                    chunksize=_REBUILD_CHUNK_ROWS,
                    **read_kwargs,
                )
            ):
                block = predictions[written : written + len(chunk)]
                for offset, column in enumerate(target_cols):
                    chunk[column] = block[:, offset]
                chunk.to_csv(handle, index=False, header=index == 0)
                written += len(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def snapshot_accepted_submission(
    working_dir: Path,
    source: Path,
    *,
    run_id: str,
    iteration: int,
) -> tuple[Path, str]:
    """Create a content-addressed, read-only snapshot of an accepted submission.

    The source must be a real file owned by the current run workspace. In
    particular, a symlink to the public sample submission cannot be promoted.
    """
    return _snapshot_submission(
        working_dir,
        source,
        run_id=run_id,
        iteration=iteration,
        store_name=".accepted_submissions",
    )


def snapshot_best_candidate_submission(
    working_dir: Path,
    source: Path,
    *,
    run_id: str,
    iteration: int,
) -> tuple[Path, str]:
    """Snapshot the current CV-selected candidate before robustness validation."""
    return _snapshot_submission(
        working_dir,
        source,
        run_id=run_id,
        iteration=iteration,
        store_name=".best_candidate_submissions",
    )


def _snapshot_submission(
    working_dir: Path,
    source: Path,
    *,
    run_id: str,
    iteration: int,
    store_name: str,
) -> tuple[Path, str]:
    """Create a content-addressed, read-only snapshot in a run-local store."""
    workspace = working_dir.resolve()
    source_path = source.resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError("Accepted submission source must be a regular file")
    if not source_path.is_relative_to(workspace):
        raise ValueError("Accepted submission source is outside the current run workspace")
    if source.name.lower() in {"sample_submission.csv", "sample-submission.csv"}:
        raise ValueError("The sample submission template cannot be accepted as output")
    if not run_id or not _SAFE_RUN_ID_RE.fullmatch(run_id):
        raise ValueError("Invalid run_id for accepted submission snapshot")

    digest = sha256_file(source)
    snapshot_dir = workspace / store_name / run_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / f"iteration-{max(0, int(iteration)):04d}-{digest}.csv"

    if snapshot.exists():
        if snapshot.is_symlink() or not snapshot.is_file():
            raise ValueError("Accepted submission snapshot path is not a regular file")
        if sha256_file(snapshot) != digest:
            raise ValueError("Existing accepted submission snapshot failed hash verification")
    else:
        # Exclusive creation prevents silently replacing an accepted artifact.
        with snapshot.open("xb") as handle:
            with source.open("rb") as source_handle:
                for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256_file(snapshot) != digest:
            snapshot.unlink(missing_ok=True)
            raise ValueError("Accepted submission snapshot failed hash verification")
        snapshot.chmod(0o444)

    return snapshot, digest


def verified_accepted_submission(
    state: Mapping[str, Any],
    working_dir: Path,
) -> Path | None:
    """Resolve the run's accepted snapshot only when path and hash both verify."""
    snapshot_value = state.get("accepted_submission_snapshot_path")
    accepted_value = state.get("accepted_submission_path")
    if not snapshot_value or not accepted_value:
        return None

    try:
        if Path(str(snapshot_value)).resolve() != Path(str(accepted_value)).resolve():
            return None
    except OSError:
        return None

    return _verified_submission(
        state,
        working_dir,
        path_key="accepted_submission_snapshot_path",
        digest_key="accepted_submission_sha256",
        store_name=".accepted_submissions",
    )


def verified_best_candidate_submission(
    state: Mapping[str, Any],
    working_dir: Path,
) -> Path | None:
    """Resolve the current CV-selected candidate without treating it as accepted."""
    return _verified_submission(
        state,
        working_dir,
        path_key="best_candidate_submission_snapshot_path",
        digest_key="best_candidate_submission_sha256",
        store_name=".best_candidate_submissions",
    )


def _verified_submission(  # noqa: PLR0911
    state: Mapping[str, Any],
    working_dir: Path,
    *,
    path_key: str,
    digest_key: str,
    store_name: str,
) -> Path | None:
    """Verify a run-local content-addressed submission snapshot."""
    snapshot_value = state.get(path_key)
    expected_digest = str(state.get(digest_key) or "").lower()
    run_id = str(state.get("run_id") or "")

    if not snapshot_value or not _SHA256_RE.fullmatch(expected_digest):
        return None
    if not run_id or not _SAFE_RUN_ID_RE.fullmatch(run_id):
        return None

    snapshot = Path(str(snapshot_value))
    workspace = working_dir.resolve()
    snapshot_root = workspace / store_name / run_id
    try:
        resolved = snapshot.resolve(strict=True)
    except OSError:
        return None

    if snapshot.is_symlink() or not resolved.is_file():
        return None
    if not resolved.is_relative_to(snapshot_root):
        return None
    if resolved.name.lower() in {"sample_submission.csv", "sample-submission.csv"}:
        return None
    if sha256_file(resolved) != expected_digest:
        return None
    return resolved


def restore_accepted_submission(
    state: Mapping[str, Any],
    working_dir: Path,
) -> Path | None:
    """Atomically restore the last verified snapshot byte-for-byte."""
    snapshot = verified_accepted_submission(state, working_dir)
    return _restore_submission(
        snapshot,
        str(state.get("accepted_submission_sha256") or ""),
        working_dir,
    )


def restore_best_candidate_submission(
    state: Mapping[str, Any],
    working_dir: Path,
) -> Path | None:
    """Restore the CV-selected candidate without marking it robustness-accepted."""
    snapshot = verified_best_candidate_submission(state, working_dir)
    return _restore_submission(
        snapshot,
        str(state.get("best_candidate_submission_sha256") or ""),
        working_dir,
    )


def _restore_submission(
    snapshot: Path | None,
    expected_digest: str,
    working_dir: Path,
) -> Path | None:
    """Atomically restore a verified snapshot byte-for-byte."""
    if snapshot is None:
        return None

    expected_digest = expected_digest.lower()
    destination = working_dir / "submission.csv"
    temporary = working_dir / f".submission-restore-{uuid.uuid4().hex}.tmp"

    try:
        with temporary.open("xb") as target, snapshot.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if sha256_file(temporary) != expected_digest:
            return None
        temporary.replace(destination)
        if sha256_file(destination) != expected_digest:
            return None
        return destination
    finally:
        temporary.unlink(missing_ok=True)
