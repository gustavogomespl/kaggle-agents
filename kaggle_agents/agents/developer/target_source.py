"""One canonical-target decision, shared by every Developer path.

The Developer used to decide "where do targets come from?" three times per
component - once while rendering the executable preamble (a ``canonical/``
directory probe), once while composing the prompt (a second, independent
directory probe) and once more while rewriting audio candidate code. The three
answers could disagree, so a run could inject a canonical contract into the
code, tell the model to parse a stale sparse-label file in the prompt, and
then rewrite the body toward ``_PRELOADED_TARGETS_DF`` - a name the preamble
never defined.

This module resolves the question exactly once per generated component and
hands the same immutable :class:`DeveloperTargetSource` to every consumer:

* a complete canonical claim always wins, whatever ``run_mode`` or
  ``component_type`` says;
* any partial, corrupt or self-contradictory canonical claim raises
  :class:`CanonicalTargetContractError` BEFORE any LLM call, instead of
  silently degrading to a filename-derived target;
* sparse preloading happens only for an inspector-verified sparse-label
  artifact, never for a file whose *name* looks label-ish.

Validation is keyed by the declared representation kind. The four real
producers (dense tabular, media-filename fallback, packed image-to-image and
the audio filename fallback) write deliberately heterogeneous metadata, so a
single universal schema would refuse legitimate contracts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ...core.state.contracts import CanonicalDataContract
from ...utils.canonical_validation import (
    describe_violations as _describe,
)
from ...utils.canonical_validation import (
    json_safe,
    json_safe_mapping,
    load_canonical_array,
    representation_kind_for,
)
from ...utils.canonical_validation import (
    validate_dense_rows as _validate_dense_rows,
)
from ...utils.canonical_validation import (
    validate_feature_manifest as _validate_feature_manifest,
)
from ...utils.canonical_validation import (
    validate_metadata_agreement as _validate_metadata_agreement,
)
from ...utils.canonical_validation import (
    validate_packed_representation as _validate_packed,
)
from ...utils.canonical_validation import (
    violation as _violation,
)
from ...utils.label_parser import inspect_label_layout


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...core.state import KaggleState


TargetSourceMode = Literal["canonical", "sparse_preload", "none"]

MARKER_VERSION = "developer-target-source/1"

_BYTE_DIGEST_CACHE_SIZE = 512
_VALIDATION_CACHE_SIZE = 32
_STREAM_CHUNK_BYTES = 1 << 20

# Byte digests keyed by (absolute path, size, mtime_ns): a component may not
# re-hash a multi-gigabyte canonical array that has not changed.
_BYTE_DIGEST_CACHE: OrderedDict[tuple[str, int, int], str] = OrderedDict()
# Successes and failures are cached separately so a corrupt contract can never
# be served from the "already validated" lane.
_VALIDATION_SUCCESS_CACHE: OrderedDict[str, bool] = OrderedDict()
_VALIDATION_FAILURE_CACHE: OrderedDict[str, list[dict[str, object]]] = OrderedDict()


def reset_target_source_caches() -> None:
    """Drop every process-local cache (tests and long-lived workers)."""
    _BYTE_DIGEST_CACHE.clear()
    _VALIDATION_SUCCESS_CACHE.clear()
    _VALIDATION_FAILURE_CACHE.clear()


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtectedInput:
    """A file the generated preamble reads eagerly, before the body boundary."""

    relative_path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size": self.size,
            "sha256": self.sha256,
        }


class DeveloperTargetSourceError(ValueError):
    """Base class for fail-closed target-source resolution errors."""

    def __init__(
        self,
        message: str,
        violations: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        super().__init__(message)
        self.violations: list[dict[str, object]] = [
            json_safe_mapping(violation) for violation in (violations or ())
        ]

    def to_dict(self) -> dict[str, object]:
        return {"message": str(self), "violations": self.violations}


class CanonicalTargetContractError(DeveloperTargetSourceError):
    """Structured pre-generation corruption of a claimed canonical contract."""


class AmbiguousTargetArtifactError(DeveloperTargetSourceError):
    """More than one candidate target/mapping artifact and no explicit manifest."""


@dataclass(frozen=True)
class DeveloperTargetSource:
    """The single, immutable target decision for one generated component."""

    mode: TargetSourceMode
    canonical_metadata: dict[str, object]
    canonical_target_path: Path | None
    packed_image_contract: bool
    label_files: tuple[str, ...]
    sparse_label_files: tuple[str, ...]
    id_mapping_path: Path | None
    required_canonical_paths: tuple[Path, ...]
    protected_inputs: tuple[ProtectedInput, ...]
    target_source_fingerprint: str
    representation_kind: str = ""
    canonical_dir: Path | None = None
    # The DECLARED test identity, or None when the contract declares none.
    # Renderers must not substitute a path of their own: a leftover
    # canonical/test_ids.npy from an earlier prep is not this run's identity.
    canonical_test_ids_path: Path | None = None

    @property
    def canonical_authoritative(self) -> bool:
        return self.mode == "canonical"

    def execution_metadata(self) -> dict[str, object]:
        """JSON-safe summary for attempt/execution records."""
        return {
            "mode": self.mode,
            "representation_kind": self.representation_kind,
            "packed_image_contract": self.packed_image_contract,
            "canonical_target_path": (
                str(self.canonical_target_path) if self.canonical_target_path else ""
            ),
            "label_files": list(self.label_files),
            "sparse_label_files": list(self.sparse_label_files),
            "id_mapping_path": str(self.id_mapping_path) if self.id_mapping_path else "",
            "canonical_test_ids_path": (
                str(self.canonical_test_ids_path) if self.canonical_test_ids_path else ""
            ),
            "target_source_fingerprint": self.target_source_fingerprint,
            "protected_inputs": [item.to_dict() for item in self.protected_inputs],
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _stable_json(payload: object) -> str:
    return json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"))


def _bounded_put(cache: OrderedDict, key: object, value: object, limit: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def _file_stat(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return int(stat.st_size), int(stat.st_mtime_ns)


def file_byte_digest(path: Path) -> str | None:
    """Stream a SHA-256 over ``path``, memoized by ``(size, mtime_ns)``."""
    stat = _file_stat(path)
    if stat is None:
        return None
    key = (str(path), stat[0], stat[1])
    cached = _BYTE_DIGEST_CACHE.get(key)
    if cached is not None:
        _BYTE_DIGEST_CACHE.move_to_end(key)
        return cached
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    value = digest.hexdigest()
    _bounded_put(_BYTE_DIGEST_CACHE, key, value, _BYTE_DIGEST_CACHE_SIZE)
    return value


def seed_byte_digest(path: Path, size: int, mtime_ns: int, digest: str) -> None:
    """Trust a digest recorded by the preparation node for an unchanged file."""
    _bounded_put(
        _BYTE_DIGEST_CACHE,
        (str(path), int(size), int(mtime_ns)),
        str(digest),
        _BYTE_DIGEST_CACHE_SIZE,
    )


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _protected_manifest(paths: Iterable[Path], root: Path) -> tuple[ProtectedInput, ...]:
    entries: list[ProtectedInput] = []
    for path in paths:
        stat = _file_stat(path)
        digest = file_byte_digest(path)
        if stat is None or digest is None:
            continue
        entries.append(
            ProtectedInput(
                relative_path=_relative_to(path, root),
                size=stat[0],
                sha256=digest,
            )
        )
    return tuple(sorted(entries, key=lambda item: item.relative_path))


def _fingerprint(
    kind: str,
    mode: str,
    manifest: Sequence[ProtectedInput],
    typed_fingerprints: Sequence[str],
) -> str:
    payload = {
        "version": MARKER_VERSION,
        "kind": kind,
        "mode": mode,
        "manifest": [
            [item.relative_path, item.size, item.sha256] for item in manifest
        ],
        "typed": sorted(str(value) for value in typed_fingerprints),
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Typed / legacy public artifact resolution (replaces the old
# ``_resolve_semantic_data_artifacts`` filename split)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ArtifactCandidate:
    path: Path
    layout: str
    evidence: tuple[str, ...]
    fingerprint: str


def _typed_records(data_files: Mapping[str, object]) -> list[Mapping[str, object]] | None:
    """Typed records when the key is present, ``None`` when it is absent."""
    if "public_artifacts" not in data_files:
        return None
    raw = data_files.get("public_artifacts")
    if not isinstance(raw, (list, tuple)):
        return []
    return [record for record in raw if isinstance(record, Mapping)]


def _declared_metadata_paths(precomputed_info: Mapping[str, object]) -> set[Path]:
    features = precomputed_info.get("features_found") if precomputed_info else None
    if not isinstance(features, Mapping):
        return set()
    resolved: set[Path] = set()
    for role, value in features.items():
        if role in {"cv_folds", "id_mapping"} and value:
            try:
                resolved.add(Path(str(value)).expanduser().resolve())
            except OSError:  # pragma: no cover - defensive
                continue
    return resolved


def _explicit_mapping(precomputed_info: Mapping[str, object]) -> Path | None:
    features = precomputed_info.get("features_found") if precomputed_info else None
    if not isinstance(features, Mapping):
        return None
    raw = features.get("id_mapping")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_file() else None


def _candidates_from_typed(
    records: Sequence[Mapping[str, object]],
    claimed_label_paths: set[Path],
) -> tuple[list[_ArtifactCandidate], list[_ArtifactCandidate], list[_ArtifactCandidate]]:
    """Return (verified sparse, id mappings, rejected TARGET candidates).

    Only a genuine target claim can land in ``rejected``, because a non-empty
    ``rejected`` with nothing verified fails the whole component closed. Two
    things count as a claim: a typed record that declared
    ``layout="sparse_labels"`` and then failed re-inspection, and a path some
    other lane explicitly listed in ``data_files["label_files"]``.

    Everything else stays out. The adapter types EVERY unassigned public
    delimited table as ``auxiliary`` with whatever the inspector said, and the
    inspector answers ``unknown``/``ambiguous_layout`` for plenty of ordinary
    tables (a two-column lookup, a single-column index). Treating those as
    rejected target candidates would hard-stop every canonical-less run that
    happens to ship one ordinary metadata CSV - a plain "no target evidence"
    outcome must return ``none``, not raise.
    """
    verified: list[_ArtifactCandidate] = []
    mappings: list[_ArtifactCandidate] = []
    rejected: list[_ArtifactCandidate] = []
    for record in records:
        role = str(record.get("role", ""))
        layout = str(record.get("layout", ""))
        raw_path = record.get("path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        evidence = tuple(str(item) for item in (record.get("evidence") or ()))
        fingerprint = str(record.get("fingerprint") or "")
        candidate = _ArtifactCandidate(path, layout, evidence, fingerprint)
        if layout == "id_mapping":
            mappings.append(candidate)
            continue
        if role != "auxiliary":
            continue
        if layout == "sparse_labels":
            # Re-inspect: a stale checkpoint must not be able to declare a
            # rectangular table "sparse labels" and get it preloaded.
            if not path.is_file():
                rejected.append(
                    _ArtifactCandidate(path, "missing", ("file_not_found",), fingerprint)
                )
                continue
            inspection = inspect_label_layout(path)
            if inspection.layout == "sparse_labels":
                verified.append(candidate)
            else:
                rejected.append(
                    _ArtifactCandidate(
                        path,
                        inspection.layout,
                        tuple(inspection.evidence),
                        fingerprint,
                    )
                )
        elif _is_claimed_label_path(path, claimed_label_paths):
            # Not a typed target claim, but another lane listed this exact file
            # as a label artifact: that IS a claim, so it must fail closed
            # rather than disappear.
            rejected.append(candidate)
    return verified, mappings, rejected


def _is_claimed_label_path(path: Path, claimed_label_paths: set[Path]) -> bool:
    if not claimed_label_paths:
        return False
    try:
        return path.expanduser().resolve() in claimed_label_paths
    except OSError:  # pragma: no cover - defensive
        return False


def _resolved_label_claims(data_files: Mapping[str, object]) -> set[Path]:
    """Paths some lane explicitly declared to be label artifacts."""
    claims: set[Path] = set()
    for raw in data_files.get("label_files") or ():
        if not raw:
            continue
        try:
            claims.add(Path(str(raw)).expanduser().resolve())
        except OSError:  # pragma: no cover - defensive
            continue
    return claims


def _candidates_from_legacy(
    label_files: Sequence[object],
    metadata_paths: set[Path],
) -> tuple[list[_ArtifactCandidate], list[_ArtifactCandidate], list[_ArtifactCandidate]]:
    verified: list[_ArtifactCandidate] = []
    mappings: list[_ArtifactCandidate] = []
    rejected: list[_ArtifactCandidate] = []
    for raw in label_files or ():
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        try:
            if path.resolve() in metadata_paths:
                continue
        except OSError:  # pragma: no cover - defensive
            continue
        if not path.is_file():
            rejected.append(_ArtifactCandidate(path, "missing", ("file_not_found",), ""))
            continue
        inspection = inspect_label_layout(path)
        candidate = _ArtifactCandidate(
            path,
            inspection.layout,
            tuple(inspection.evidence),
            "",
        )
        if inspection.layout == "sparse_labels":
            verified.append(candidate)
        elif inspection.layout == "id_mapping":
            mappings.append(candidate)
        else:
            rejected.append(candidate)
    return verified, mappings, rejected


def _rejection_details(candidates: Sequence[_ArtifactCandidate]) -> list[dict[str, object]]:
    return [
        {
            "path": str(candidate.path),
            "layout": candidate.layout,
            "evidence": list(candidate.evidence),
        }
        for candidate in candidates
    ]


def _resolve_mapping_path(
    mappings: Sequence[_ArtifactCandidate],
    explicit: Path | None,
) -> Path | None:
    if explicit is not None:
        return explicit
    existing = [candidate for candidate in mappings if candidate.path.is_file()]
    if not existing:
        return None
    if len(existing) > 1:
        raise AmbiguousTargetArtifactError(
            "Multiple ID-mapping artifacts and no explicit manifest declares "
            "which one names model inputs: "
            + ", ".join(str(candidate.path) for candidate in existing),
            _rejection_details(existing),
        )
    return existing[0].path


# ---------------------------------------------------------------------------
# Canonical claim evidence
# ---------------------------------------------------------------------------


_CANONICAL_CLAIM_KEYS = (
    "canonical_contract",
    "canonical_metadata",
    "canonical_dir",
    "canonical_train_ids_path",
    "canonical_y_path",
    "canonical_folds_path",
    "canonical_feature_cols_path",
    "canonical_test_ids_path",
)

def _canonical_claim_present(state: Mapping[str, object] | None) -> bool:
    if not state:
        return False
    return any(state.get(key) for key in _CANONICAL_CLAIM_KEYS)


# ---------------------------------------------------------------------------
# Validation marker
# ---------------------------------------------------------------------------


_MARKER_SEMANTIC_FIELDS = (
    "n_train",
    "n_test",
    "n_folds",
    "id_col",
    "id_is_synthetic",
    "target_col",
    "target_cols",
    "target_type",
    "is_classification",
    "cv_strategy",
    "folds_hash",
    "y_hash",
    "train_ids_hash",
    "train_schema_hash",
    "packed_image_contract",
)

_CONTRACT_PATH_FIELDS = (
    "train_ids_path",
    "y_path",
    "folds_path",
    "feature_cols_path",
    "metadata_path",
    "temporal_splits_path",
    "oof_eligible_mask_path",
    "temporal_order_path",
    "test_ids_path",
    "image_input_paths_path",
    "image_test_input_paths_path",
)


def _declared_paths(contract: Mapping[str, object]) -> list[Path]:
    paths: list[Path] = []
    for field in _CONTRACT_PATH_FIELDS:
        raw = contract.get(field)
        if raw:
            paths.append(Path(str(raw)))
    return paths


def _marker_fingerprint(
    contract: Mapping[str, object],
    metadata_sha256: str,
    representation_kind: str,
) -> str:
    payload = {
        "version": MARKER_VERSION,
        "representation_kind": representation_kind,
        "metadata_sha256": metadata_sha256,
        "semantics": {
            field: json_safe(contract.get(field)) for field in _MARKER_SEMANTIC_FIELDS
        },
        "paths": sorted(str(path) for path in _declared_paths(contract)),
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _current_file_map(paths: Iterable[Path]) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for path in paths:
        stat = _file_stat(path)
        if stat is None:
            continue
        mapping[str(path)] = [stat[0], stat[1]]
    return mapping


def build_canonical_validation_marker(
    contract: Mapping[str, object],
    metadata: Mapping[str, object],
) -> dict[str, object] | None:
    """Record the proof ``canonical_data_preparation_node()`` just established.

    The node has already loaded and hashed the semantic arrays; the marker
    reuses those values and adds ONE streaming byte pass per canonical file so
    later components never re-deserialize a multi-million-row contract just to
    convince themselves nothing changed.
    """
    metadata_path = contract.get("metadata_path")
    if not metadata_path:
        return None
    metadata_digest = file_byte_digest(Path(str(metadata_path)))
    if metadata_digest is None:
        return None
    representation_kind = representation_kind_for(metadata, contract)
    paths = _declared_paths(contract)
    protected = _protected_manifest(paths, Path(str(contract.get("canonical_dir") or "")).parent)
    return {
        "version": MARKER_VERSION,
        "representation_kind": representation_kind,
        "metadata_sha256": metadata_digest,
        "fingerprint": _marker_fingerprint(contract, metadata_digest, representation_kind),
        "files": _current_file_map(paths),
        "protected_inputs": [item.to_dict() for item in protected],
    }


# ---------------------------------------------------------------------------
# Canonical validation
# ---------------------------------------------------------------------------


def _validate_contract_shape(
    raw_contract: Mapping[str, object],
    violations: list[dict[str, object]],
) -> CanonicalDataContract | None:
    field_names = {field.name for field in dataclasses.fields(CanonicalDataContract)}
    required = {
        field.name
        for field in dataclasses.fields(CanonicalDataContract)
        if field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING  # type: ignore[misc]
    }
    extra = sorted(set(raw_contract) - field_names)
    if extra:
        violations.append(_violation("unknown_contract_keys", keys=extra))
    missing = sorted(required - set(raw_contract) - {"target_cols", "target_type"})
    if missing:
        violations.append(_violation("missing_contract_keys", keys=missing))
    if violations:
        return None
    try:
        contract = CanonicalDataContract.from_dict(dict(raw_contract))
    except Exception as exc:
        violations.append(_violation("undeserializable_contract", error=str(exc)))
        return None
    for field_name in ("n_train", "n_test", "n_folds"):
        value = getattr(contract, field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            violations.append(
                _violation("non_integer_contract_field", field=field_name, value=value)
            )
    for field_name in ("id_col", "target_col", "target_type", "cv_strategy"):
        if not isinstance(getattr(contract, field_name), str):
            violations.append(_violation("non_string_contract_field", field=field_name))
    if not isinstance(contract.target_cols, list) or not contract.target_cols:
        violations.append(_violation("invalid_target_cols"))
    if violations:
        return None
    return contract


def _validate_paths(
    contract: CanonicalDataContract,
    canonical_root: Path,
    violations: list[dict[str, object]],
) -> None:
    try:
        root = canonical_root.resolve()
    except OSError:  # pragma: no cover - defensive
        violations.append(_violation("unresolvable_canonical_root", path=str(canonical_root)))
        return
    for field in ("canonical_dir", *_CONTRACT_PATH_FIELDS):
        raw = getattr(contract, field, None)
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_symlink():
            violations.append(_violation("symlinked_canonical_path", field=field, path=str(path)))
            continue
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover - defensive
            violations.append(_violation("unresolvable_canonical_path", field=field, path=str(path)))
            continue
        inside = resolved == root or resolved.is_relative_to(root)
        if not inside:
            violations.append(
                _violation(
                    "canonical_path_outside_canonical_directory",
                    field=field,
                    path=str(path),
                    canonical_root=str(root),
                )
            )


def _required_files(
    contract: CanonicalDataContract,
    metadata: Mapping[str, object],
    kind: str,
) -> list[tuple[str, Path | None]]:
    required: list[tuple[str, Path | None]] = [
        ("train_ids_path", Path(contract.train_ids_path) if contract.train_ids_path else None),
        ("y_path", Path(contract.y_path) if contract.y_path else None),
        ("folds_path", Path(contract.folds_path) if contract.folds_path else None),
        (
            "feature_cols_path",
            Path(contract.feature_cols_path) if contract.feature_cols_path else None,
        ),
        ("metadata_path", Path(contract.metadata_path) if contract.metadata_path else None),
    ]
    if kind == "packed_image":
        for field in ("image_input_paths_path", "image_test_input_paths_path"):
            raw = getattr(contract, field)
            required.append((field, Path(raw) if raw else None))
    if contract.cv_strategy == "temporal_forward_chaining":
        for field in ("temporal_splits_path", "oof_eligible_mask_path", "temporal_order_path"):
            raw = getattr(contract, field)
            required.append((field, Path(raw) if raw else None))
    declared_test_rows = _declared_test_rows(contract, metadata)
    if declared_test_rows > 0:
        raw = contract.test_ids_path
        required.append(("test_ids_path", Path(raw) if raw else None))
    return required


def _declared_test_rows(
    contract: CanonicalDataContract,
    metadata: Mapping[str, object],
) -> int:
    for value in (contract.n_test, metadata.get("n_test")):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _validate_test_identity(
    contract: CanonicalDataContract,
    metadata: Mapping[str, object],
    violations: list[dict[str, object]],
) -> None:
    declared_rows = _declared_test_rows(contract, metadata)
    raw = contract.test_ids_path
    if declared_rows <= 0:
        return
    if not raw:
        violations.append(_violation("missing_declared_test_ids_path", n_test=declared_rows))
        return
    test_ids = load_canonical_array(Path(str(raw)), violations, "test_ids")
    if test_ids is None:
        return
    test_ids = np.asarray(test_ids)
    if test_ids.ndim != 1:
        violations.append(_violation("test_ids_are_not_scalar", shape=list(test_ids.shape)))
        return
    keys = [str(value) for value in test_ids.tolist()]
    if len(keys) != declared_rows:
        violations.append(
            _violation("test_id_count_disagreement", declared=declared_rows, observed=len(keys))
        )
    if len(set(keys)) != len(keys):
        violations.append(_violation("duplicate_test_ids"))
    if bool(metadata.get("test_ids_are_positional", False)) and keys != [
        str(index) for index in range(len(keys))
    ]:
        violations.append(_violation("positional_test_ids_are_not_row_positions"))


def _full_canonical_validation(
    contract: CanonicalDataContract,
    metadata: Mapping[str, object],
    kind: str,
) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    is_valid, contract_violations = contract.validate()
    if not is_valid:
        violations.extend(
            _violation("contract_checksum_violation", detail=detail)
            for detail in contract_violations
        )
    if kind == "packed_image":
        _validate_packed(contract, violations)
    else:
        _validate_dense_rows(contract, metadata, violations)
    _validate_test_identity(contract, metadata, violations)
    return violations


# ---------------------------------------------------------------------------
# Canonical resolution
# ---------------------------------------------------------------------------


def _read_metadata_json(
    contract: CanonicalDataContract,
    violations: list[dict[str, object]],
) -> tuple[dict[str, object] | None, str]:
    path = Path(contract.metadata_path)
    digest = file_byte_digest(path)
    if digest is None:
        violations.append(_violation("missing_metadata_json", path=str(path)))
        return None, ""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        violations.append(_violation("invalid_metadata_json", error=str(exc)))
        return None, digest
    if not isinstance(metadata, dict):
        violations.append(_violation("invalid_metadata_json", error="metadata is not an object"))
        return None, digest
    return metadata, digest


def _resolve_canonical(
    working_dir: Path,
    state: Mapping[str, object],
    typed_fingerprints: Sequence[str],
    sparse_label_files: tuple[str, ...],
    id_mapping_path: Path | None,
) -> DeveloperTargetSource:
    violations: list[dict[str, object]] = []
    raw_contract = state.get("canonical_contract")
    if not isinstance(raw_contract, Mapping) or not raw_contract:
        raise CanonicalTargetContractError(
            "Canonical preparation reported success but no canonical_contract "
            "is present in the state; refusing to generate against an "
            "undeclared contract.",
            [_violation("missing_canonical_contract")],
        )

    contract = _validate_contract_shape(raw_contract, violations)
    if contract is None:
        raise CanonicalTargetContractError(
            _describe(violations, "canonical_contract is malformed"), violations
        )

    canonical_root = working_dir / "canonical"
    _validate_paths(contract, canonical_root, violations)
    if violations:
        raise CanonicalTargetContractError(
            _describe(violations, "canonical paths are unsafe"), violations
        )

    metadata, metadata_digest = _read_metadata_json(contract, violations)
    if metadata is None:
        raise CanonicalTargetContractError(
            _describe(violations, "canonical metadata.json is unusable"), violations
        )

    kind = representation_kind_for(metadata, raw_contract)

    for field, path in _required_files(contract, metadata, kind):
        if path is None:
            violations.append(_violation("undeclared_required_artifact", field=field))
        elif not path.is_file():
            violations.append(
                _violation("missing_canonical_artifact", field=field, path=str(path))
            )
    if violations:
        raise CanonicalTargetContractError(
            _describe(violations, "canonical contract is incomplete"), violations
        )

    _validate_metadata_agreement(
        contract,
        metadata,
        state.get("canonical_metadata") if isinstance(state.get("canonical_metadata"), Mapping) else None,
        kind,
        violations,
    )
    if kind != "packed_image":
        _validate_feature_manifest(contract, metadata, violations)
    if violations:
        raise CanonicalTargetContractError(
            _describe(violations, "canonical contract contradicts its metadata"), violations
        )

    _establish_row_and_hash_proof(
        working_dir,
        state,
        raw_contract,
        contract,
        metadata,
        metadata_digest,
        kind,
    )

    required_paths = [
        path for _field, path in _required_files(contract, metadata, kind) if path
    ]
    optional = [
        Path(str(getattr(contract, field)))
        for field in (
            "test_ids_path",
            "temporal_splits_path",
            "oof_eligible_mask_path",
            "temporal_order_path",
        )
        if getattr(contract, field)
    ]
    protected_sources = list(dict.fromkeys([*required_paths, *optional]))
    protected = _protected_manifest(protected_sources, working_dir)
    return DeveloperTargetSource(
        mode="canonical",
        canonical_metadata=dict(metadata),
        canonical_target_path=Path(contract.y_path),
        packed_image_contract=kind == "packed_image",
        label_files=(),
        sparse_label_files=sparse_label_files,
        id_mapping_path=id_mapping_path,
        required_canonical_paths=tuple(protected_sources),
        protected_inputs=protected,
        target_source_fingerprint=_fingerprint(kind, "canonical", protected, typed_fingerprints),
        representation_kind=kind,
        canonical_dir=Path(contract.canonical_dir),
        canonical_test_ids_path=(
            Path(str(contract.test_ids_path)) if contract.test_ids_path else None
        ),
    )


def _establish_row_and_hash_proof(
    working_dir: Path,
    state: Mapping[str, object],
    raw_contract: Mapping[str, object],
    contract: CanonicalDataContract,
    metadata: Mapping[str, object],
    metadata_digest: str,
    kind: str,
) -> None:
    """Prove the rows/shapes/hashes hold, doing the heavy work at most once.

    When the marker recorded by ``canonical_data_preparation_node()`` still
    matches the contract fingerprint AND every canonical file's
    ``(size, mtime_ns)``, that node's proof stands and nothing is
    re-deserialized - which is what keeps an 8.9-million-row contract from
    being re-hashed once per generated component. A legacy checkpoint without
    a marker pays for one full validation, cached by the same key.
    """
    current_files = _current_file_map(_declared_paths(raw_contract))
    fingerprint = _marker_fingerprint(raw_contract, metadata_digest, kind)
    marker = state.get("canonical_contract_validation")
    if (
        isinstance(marker, Mapping)
        and marker.get("version") == MARKER_VERSION
        and marker.get("fingerprint") == fingerprint
        and marker.get("metadata_sha256") == metadata_digest
        and marker.get("files") == current_files
    ):
        _seed_from_marker(marker, working_dir)
        return

    cache_key = _validation_cache_key(fingerprint, current_files)
    cached_failure = _VALIDATION_FAILURE_CACHE.get(cache_key)
    if cached_failure is not None:
        _VALIDATION_FAILURE_CACHE.move_to_end(cache_key)
        raise CanonicalTargetContractError(
            _describe(cached_failure, "canonical contract failed validation"),
            cached_failure,
        )
    if cache_key in _VALIDATION_SUCCESS_CACHE:
        _VALIDATION_SUCCESS_CACHE.move_to_end(cache_key)
        return

    heavy = _full_canonical_validation(contract, metadata, kind)
    if heavy:
        _bounded_put(_VALIDATION_FAILURE_CACHE, cache_key, heavy, _VALIDATION_CACHE_SIZE)
        raise CanonicalTargetContractError(
            _describe(heavy, "canonical contract failed validation"), heavy
        )
    _bounded_put(_VALIDATION_SUCCESS_CACHE, cache_key, True, _VALIDATION_CACHE_SIZE)


def _seed_from_marker(marker: Mapping[str, object], working_dir: Path) -> None:
    files = marker.get("files")
    entries = marker.get("protected_inputs")
    if not isinstance(files, Mapping) or not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        relative = str(entry.get("relative_path", ""))
        digest = str(entry.get("sha256", ""))
        if not relative or not digest:
            continue
        path = working_dir / relative
        stat = _file_stat(path)
        if stat is None:
            continue
        seed_byte_digest(path, stat[0], stat[1], digest)


def _validation_cache_key(fingerprint: str, files: Mapping[str, list[int]]) -> str:
    return hashlib.sha256(
        _stable_json({"fingerprint": fingerprint, "files": files}).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def resolve_developer_target_source(
    working_dir: str | Path,
    state: KaggleState | Mapping[str, object] | None,
    data_files: Mapping[str, object] | None = None,
    precomputed_info: Mapping[str, object] | None = None,
    component_type: str = "model",
) -> DeveloperTargetSource:
    """Decide, exactly once, where this component's targets come from.

    ``component_type`` is kept for call-site compatibility and diagnostics
    only: it may not change canonical precedence, and it never authorizes a
    fallback away from a claimed contract. ``run_mode`` is likewise ignored -
    the caller decides how to REPORT terminality, never which target source
    is selected.
    """
    del component_type  # diagnostics only; must not influence selection
    working_dir = Path(working_dir)
    state = state if isinstance(state, Mapping) else {}
    data_files = data_files if isinstance(data_files, Mapping) else {}
    precomputed_info = precomputed_info if isinstance(precomputed_info, Mapping) else {}

    typed = _typed_records(data_files)
    metadata_paths = _declared_metadata_paths(precomputed_info)
    explicit_mapping = _explicit_mapping(precomputed_info)
    if typed is None:
        verified, mappings, rejected = _candidates_from_legacy(
            data_files.get("label_files") or (), metadata_paths
        )
        typed_fingerprints: list[str] = []
    else:
        verified, mappings, rejected = _candidates_from_typed(
            typed,
            _resolved_label_claims(data_files),
        )
        typed_fingerprints = [
            str(record.get("fingerprint") or "") for record in typed
        ]

    id_mapping_path = _resolve_mapping_path(mappings, explicit_mapping)
    sparse_label_files = tuple(str(candidate.path) for candidate in verified)

    canonical_prepared = bool(state.get("canonical_data_prepared"))
    if canonical_prepared:
        return _resolve_canonical(
            working_dir,
            state,
            typed_fingerprints,
            sparse_label_files,
            id_mapping_path,
        )

    if _canonical_claim_present(state):
        present = sorted(key for key in _CANONICAL_CLAIM_KEYS if state.get(key))
        raise CanonicalTargetContractError(
            "canonical_data_prepared is false while the state still declares a "
            f"canonical contract ({', '.join(present)}); refusing to guess "
            "whether the contract is authoritative.",
            [_violation("contradictory_canonical_claim", declared_keys=present)],
        )

    if len(verified) > 1:
        raise AmbiguousTargetArtifactError(
            "Multiple verified sparse-label artifacts and no manifest declares "
            "which one carries the targets: "
            + ", ".join(sparse_label_files),
            _rejection_details(verified),
        )

    if not verified and rejected:
        raise AmbiguousTargetArtifactError(
            "Every label-named public artifact failed layout inspection: "
            + "; ".join(
                f"{candidate.path} -> {candidate.layout} "
                f"({', '.join(candidate.evidence) or 'no evidence'})"
                for candidate in rejected
            ),
            _rejection_details(rejected),
        )

    if verified:
        protected_sources = [verified[0].path]
        if id_mapping_path is not None:
            protected_sources.append(id_mapping_path)
        protected = _protected_manifest(protected_sources, working_dir)
        return DeveloperTargetSource(
            mode="sparse_preload",
            canonical_metadata={},
            canonical_target_path=None,
            packed_image_contract=False,
            label_files=sparse_label_files,
            sparse_label_files=sparse_label_files,
            id_mapping_path=id_mapping_path,
            required_canonical_paths=(),
            protected_inputs=protected,
            target_source_fingerprint=_fingerprint(
                "sparse_labels", "sparse_preload", protected, typed_fingerprints
            ),
            representation_kind="sparse_labels",
            canonical_dir=None,
        )

    protected = (
        _protected_manifest([id_mapping_path], working_dir)
        if id_mapping_path is not None
        else ()
    )
    return DeveloperTargetSource(
        mode="none",
        canonical_metadata={},
        canonical_target_path=None,
        packed_image_contract=False,
        label_files=(),
        sparse_label_files=(),
        id_mapping_path=id_mapping_path,
        required_canonical_paths=(),
        protected_inputs=protected,
        target_source_fingerprint=_fingerprint("none", "none", protected, typed_fingerprints),
        representation_kind="none",
        canonical_dir=None,
    )


def auxiliary_public_artifacts(
    data_files: Mapping[str, object] | None,
    target_source: DeveloperTargetSource | None,
) -> tuple[dict[str, str], ...]:
    """Neutral, non-target description of the remaining public artifacts.

    In canonical mode a stale sparse-label file is NOT described at all: the
    canonical contract owns the targets, and naming the file in a prompt is
    exactly how a candidate ends up re-parsing it.
    """
    if not isinstance(data_files, Mapping):
        return ()
    records = _typed_records(data_files) or []
    hidden = set(target_source.sparse_label_files) if target_source else set()
    hidden |= set(target_source.label_files) if target_source else set()
    described: list[dict[str, str]] = []
    for record in records:
        if str(record.get("role", "")) != "auxiliary":
            continue
        path = str(record.get("path") or "")
        if not path or path in hidden:
            continue
        described.append(
            {
                "path": path,
                "layout": str(record.get("layout", "")),
                "evidence": ", ".join(
                    str(item) for item in (record.get("evidence") or ())
                ),
            }
        )
    return tuple(described)


__all__ = [
    "MARKER_VERSION",
    "AmbiguousTargetArtifactError",
    "CanonicalTargetContractError",
    "DeveloperTargetSource",
    "DeveloperTargetSourceError",
    "ProtectedInput",
    "TargetSourceMode",
    "auxiliary_public_artifacts",
    "build_canonical_validation_marker",
    "file_byte_digest",
    "representation_kind_for",
    "reset_target_source_caches",
    "resolve_developer_target_source",
]
