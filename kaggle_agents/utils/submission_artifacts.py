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
