"""Host-side integrity boundary for canonical evaluation artifacts.

Generated programs execute in the run workspace and therefore need read access
to ``canonical/``.  They must never be able to change the labels, folds, IDs,
or metadata that the host later uses to evaluate their OOF predictions.

This module snapshots the complete canonical directory in controller memory,
verifies it after execution, and restores the original bytes when anything
changed. It is an integrity check, not an OS sandbox; formal benchmark runs
still need a mount namespace without private labels.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


_REQUIRED_TABULAR_CANONICAL_FILES = frozenset(
    {
        "feature_cols.json",
        "folds.npy",
        "metadata.json",
        "train_ids.npy",
        "y.npy",
    }
)
_REQUIRED_PACKED_IMAGE_CANONICAL_FILES = frozenset(
    {
        "feature_cols.json",
        "folds.npy",
        "image_input_paths.npy",
        "image_targets.npz",
        "image_test_input_paths.npy",
        "metadata.json",
        "test_ids.npy",
        "train_ids.npy",
    }
)


class CanonicalIntegrityError(RuntimeError):
    """Raised when the host cannot establish or restore canonical integrity."""


@dataclass(frozen=True)
class CanonicalIntegritySnapshot:
    """Host-memory description of one canonical snapshot.

    The bytes intentionally remain in the controller process instead of a
    named temporary directory. Generated code runs under the same OS user in
    the Colab-compatible executor; a filesystem backup would therefore be
    discoverable and mutable before the host tried to restore from it.
    """

    canonical_dir: Path
    files: dict[str, bytes]
    manifest: dict[str, str]
    modes: dict[str, int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _directory_contents(root: Path) -> dict[str, bytes]:
    """Read every regular file while rejecting link indirection."""
    if root.is_symlink() or not root.is_dir():
        raise CanonicalIntegrityError(
            f"Canonical contract root is not a regular directory: {root}"
        )

    contents: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise CanonicalIntegrityError(
                f"Canonical contract contains a symlink: {relative}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise CanonicalIntegrityError(
                f"Canonical contract contains a non-regular entry: {relative}"
            )
        contents[relative] = path.read_bytes()
    return contents


def _directory_manifest(root: Path) -> dict[str, str]:
    """Hash every regular file below ``root`` and reject link indirection."""
    if root.is_symlink() or not root.is_dir():
        raise CanonicalIntegrityError(
            f"Canonical contract root is not a regular directory: {root}"
        )

    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise CanonicalIntegrityError(
                f"Canonical contract contains a symlink: {relative}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise CanonicalIntegrityError(
                f"Canonical contract contains a non-regular entry: {relative}"
            )
        manifest[relative] = _sha256_file(path)
    return manifest


def _directory_modes(root: Path) -> dict[str, int]:
    """Record permission bits so restoration does not alter host behavior."""
    modes = {".": stat.S_IMODE(root.stat().st_mode)}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CanonicalIntegrityError(
                "Canonical contract contains a symlink while recording modes: "
                f"{path.relative_to(root).as_posix()}"
            )
        modes[path.relative_to(root).as_posix()] = stat.S_IMODE(
            path.stat().st_mode
        )
    return modes


def _apply_directory_modes(root: Path, modes: dict[str, int]) -> None:
    """Restore children before the root so traversal remains available."""
    for relative, mode in sorted(
        modes.items(),
        key=lambda item: item[0].count("/"),
        reverse=True,
    ):
        path = root if relative == "." else root / relative
        path.chmod(mode)


def snapshot_canonical_contract(
    working_dir: str | Path,
) -> CanonicalIntegritySnapshot | None:
    """Snapshot ``canonical/`` in controller-owned memory.

    ``None`` means no canonical directory exists.  The real MLE-bench workflow
    creates the directory before model generation; retaining ``None`` keeps
    direct executor use and regular Kaggle workflows backward compatible.
    """
    workspace = Path(working_dir).resolve()
    canonical_dir = workspace / "canonical"
    if not canonical_dir.exists():
        return None

    files = _directory_contents(canonical_dir)
    manifest = {
        relative: _sha256_bytes(contents)
        for relative, contents in files.items()
    }
    modes = _directory_modes(canonical_dir)
    try:
        metadata = json.loads(files["metadata.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalIntegrityError(
            "Canonical metadata is missing or invalid before generated-code execution"
        ) from exc
    packed_image_contract = bool(
        metadata.get("packed_image_contract")
        or metadata.get("task_type") == "image_to_image"
    )
    required_files = (
        _REQUIRED_PACKED_IMAGE_CANONICAL_FILES
        if packed_image_contract
        else _REQUIRED_TABULAR_CANONICAL_FILES
    )
    missing = sorted(required_files - set(manifest))
    if missing:
        raise CanonicalIntegrityError(
            "Canonical contract is incomplete before generated-code execution; "
            f"missing: {missing}"
        )

    # Detect a concurrent mutation between reading the bytes and completing
    # the snapshot. No generated process has started yet, so any mismatch is a
    # host-side setup failure and must fail closed.
    if _directory_manifest(canonical_dir) != manifest:
        raise CanonicalIntegrityError(
            "Canonical contract changed while the host snapshot was created"
        )

    return CanonicalIntegritySnapshot(
        canonical_dir=canonical_dir,
        files=files,
        manifest=manifest,
        modes=modes,
    )


def _describe_manifest_changes(
    expected: dict[str, str],
    observed: dict[str, str],
) -> list[str]:
    missing = sorted(set(expected) - set(observed))
    added = sorted(set(observed) - set(expected))
    changed = sorted(
        path
        for path in set(expected) & set(observed)
        if expected[path] != observed[path]
    )
    descriptions: list[str] = []
    if changed:
        descriptions.append("modified=" + ",".join(changed))
    if missing:
        descriptions.append("missing=" + ",".join(missing))
    if added:
        descriptions.append("added=" + ",".join(added))
    return descriptions


def _restore_canonical_contract(snapshot: CanonicalIntegritySnapshot) -> None:
    """Replace a mutated canonical tree from controller-owned memory."""
    memory_manifest = {
        relative: _sha256_bytes(contents)
        for relative, contents in snapshot.files.items()
    }
    if memory_manifest != snapshot.manifest:
        raise CanonicalIntegrityError(
            "Host-memory canonical integrity snapshot failed verification"
        )

    canonical_dir = snapshot.canonical_dir
    workspace = canonical_dir.parent
    restore_dir = workspace / f".canonical-restore-{uuid.uuid4().hex}"
    displaced = workspace / f".canonical-mutated-{uuid.uuid4().hex}"

    restore_dir.mkdir(parents=True, exist_ok=False)
    directory_paths = sorted(
        (
            relative
            for relative in snapshot.modes
            if relative != "." and relative not in snapshot.files
        ),
        key=lambda relative: relative.count("/"),
    )
    for relative in directory_paths:
        (restore_dir / relative).mkdir(parents=True, exist_ok=True)
    for relative, contents in snapshot.files.items():
        destination = restore_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
    _apply_directory_modes(restore_dir, snapshot.modes)
    if _directory_manifest(restore_dir) != snapshot.manifest:
        shutil.rmtree(restore_dir, ignore_errors=True)
        raise CanonicalIntegrityError(
            "Restored canonical copy failed verification"
        )

    moved_current = False
    try:
        if canonical_dir.exists() or canonical_dir.is_symlink():
            canonical_dir.replace(displaced)
            moved_current = True
        restore_dir.replace(canonical_dir)
        if _directory_manifest(canonical_dir) != snapshot.manifest:
            raise CanonicalIntegrityError(
                "Canonical contract failed verification after restore"
            )
    except Exception:
        if not canonical_dir.exists() and moved_current and displaced.exists():
            displaced.replace(canonical_dir)
        raise
    finally:
        if restore_dir.exists():
            shutil.rmtree(restore_dir, ignore_errors=True)
        if displaced.exists() or displaced.is_symlink():
            if displaced.is_symlink() or displaced.is_file():
                displaced.unlink(missing_ok=True)
            else:
                shutil.rmtree(displaced, ignore_errors=True)


def verify_and_restore_canonical_contract(
    snapshot: CanonicalIntegritySnapshot,
) -> list[str]:
    """Verify the canonical tree, restoring it when mutated.

    Returns a concise list describing detected changes.  An empty list means
    that generated code left the complete contract byte-for-byte unchanged.
    """
    changes: list[str]
    try:
        observed = _directory_manifest(snapshot.canonical_dir)
        changes = _describe_manifest_changes(snapshot.manifest, observed)
        if _directory_modes(snapshot.canonical_dir) != snapshot.modes:
            changes.append("permissions_changed")
    except CanonicalIntegrityError as exc:
        changes = [str(exc)]

    if changes:
        _restore_canonical_contract(snapshot)
    return changes


# ---------------------------------------------------------------------------
# Protected preamble inputs
# ---------------------------------------------------------------------------
#
# The injected header reads a handful of files eagerly, above the candidate
# body: canonical arrays, a verified sparse-label artifact, an ID mapping. The
# generator fingerprints them when it renders the header, so a failure in that
# region is only attributable to the harness while those bytes are still the
# bytes the header was rendered against.
#
# Hashing them unconditionally per attempt is not an option: canonical arrays
# reach multiple gigabytes and every retry would re-read them. Digests are
# therefore memoized by ``(resolved path, st_size, st_mtime_ns)``; a clean stat
# match reuses the cached digest and a full byte re-hash runs only when the
# stat changed.

_PROTECTED_DIGEST_CACHE: OrderedDict[tuple[str, int, int], str] = OrderedDict()
_PROTECTED_DIGEST_CACHE_SIZE = 512
# Bytes above this size are verified but not held in controller memory. The
# canonical directory has its own snapshot; this cap only affects unusually
# large auxiliary inputs, which are reported rather than restored.
_PROTECTED_SNAPSHOT_MAX_BYTES = 64 * 1024 * 1024

ProtectedInputDeclaration = tuple[str, int, str]


class ProtectedInputIntegrityError(CanonicalIntegrityError):
    """A fingerprinted preamble input is missing, mutated or unsafe."""


class ProtectedInputMutationError(RuntimeError):
    """A previously fingerprinted preamble input changed before execution.

    Defined next to the check that raises it so the executor can name the type
    in an ``except`` clause without importing the developer package;
    ``kaggle_agents.agents.developer.execution_failures`` re-exports this exact
    class as part of the Developer failure taxonomy.
    """


@dataclass(frozen=True)
class ProtectedInputSnapshot:
    """Controller-memory copy of the restorable protected inputs."""

    workspace: Path
    declared: tuple[ProtectedInputDeclaration, ...]
    files: dict[str, bytes]
    modes: dict[str, int]


def reset_protected_input_digest_cache() -> None:
    """Drop the stat-keyed digest memo (tests and long-lived workers)."""
    _PROTECTED_DIGEST_CACHE.clear()


def _cached_file_digest(path: Path) -> str:
    """SHA-256 of ``path``, re-read only when its stat changed."""
    stat_result = path.stat()
    key = (str(path), int(stat_result.st_size), int(stat_result.st_mtime_ns))
    cached = _PROTECTED_DIGEST_CACHE.get(key)
    if cached is not None:
        _PROTECTED_DIGEST_CACHE.move_to_end(key)
        return cached
    digest = _sha256_file(path)
    _PROTECTED_DIGEST_CACHE[key] = digest
    _PROTECTED_DIGEST_CACHE.move_to_end(key)
    while len(_PROTECTED_DIGEST_CACHE) > _PROTECTED_DIGEST_CACHE_SIZE:
        _PROTECTED_DIGEST_CACHE.popitem(last=False)
    return digest


def _resolved_protected_path(workspace: Path, relative: str) -> Path:
    """Confine one declared path to the workspace, rejecting link indirection."""
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ProtectedInputIntegrityError(
            f"Protected input path is not workspace-relative: {relative!r}"
        )
    path = workspace / relative
    if path.is_symlink():
        raise ProtectedInputIntegrityError(
            f"Protected input is a symlink: {relative}"
        )
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ProtectedInputIntegrityError(
            f"Protected input escapes the workspace: {relative}"
        ) from exc
    return path


def describe_protected_input_changes(
    working_dir: str | Path,
    declared: Sequence[ProtectedInputDeclaration],
) -> list[str]:
    """Compare declared size/hash against what is on disk right now."""
    workspace = Path(working_dir)
    changes: list[str] = []
    for relative, size, digest in declared:
        try:
            path = _resolved_protected_path(workspace, relative)
        except ProtectedInputIntegrityError as exc:
            changes.append(str(exc))
            continue
        if not path.is_file():
            changes.append(f"missing={relative}")
            continue
        try:
            observed_size = path.stat().st_size
        except OSError as exc:
            changes.append(f"unreadable={relative}:{exc}")
            continue
        if int(observed_size) != int(size):
            changes.append(
                f"size_changed={relative}:{observed_size}!={size}"
            )
            continue
        try:
            observed_digest = _cached_file_digest(path)
        except OSError as exc:
            changes.append(f"unreadable={relative}:{exc}")
            continue
        if observed_digest != digest:
            changes.append(f"modified={relative}")
    return changes


def snapshot_protected_inputs(
    working_dir: str | Path,
    declared: Sequence[ProtectedInputDeclaration],
    skip_relatives: Iterable[str] = (),
) -> ProtectedInputSnapshot | None:
    """Hold restorable protected inputs in controller memory before launch."""
    if not declared:
        return None
    workspace = Path(working_dir)
    skip = {str(item) for item in skip_relatives}
    files: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    for relative, size, _digest in declared:
        if relative in skip or int(size) > _PROTECTED_SNAPSHOT_MAX_BYTES:
            continue
        path = _resolved_protected_path(workspace, relative)
        if not path.is_file():
            raise ProtectedInputIntegrityError(f"missing={relative}")
        files[relative] = path.read_bytes()
        modes[relative] = stat.S_IMODE(path.stat().st_mode)
    return ProtectedInputSnapshot(
        workspace=workspace,
        declared=tuple(declared),
        files=files,
        modes=modes,
    )


def verify_and_restore_protected_inputs(
    snapshot: ProtectedInputSnapshot,
) -> list[str]:
    """Verify the protected inputs, restoring the ones held in memory."""
    changes = describe_protected_input_changes(snapshot.workspace, snapshot.declared)
    if not changes:
        return changes

    for relative, contents in snapshot.files.items():
        path = snapshot.workspace / relative
        try:
            if path.is_symlink() or (path.exists() and not path.is_file()):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            mode = snapshot.modes.get(relative)
            if mode is not None:
                path.chmod(mode)
        except OSError as exc:
            changes.append(f"restore_failed={relative}:{exc}")
    return changes


__all__ = [
    "CanonicalIntegrityError",
    "CanonicalIntegritySnapshot",
    "ProtectedInputIntegrityError",
    "ProtectedInputMutationError",
    "ProtectedInputSnapshot",
    "describe_protected_input_changes",
    "reset_protected_input_digest_cache",
    "snapshot_canonical_contract",
    "snapshot_protected_inputs",
    "verify_and_restore_canonical_contract",
    "verify_and_restore_protected_inputs",
]
