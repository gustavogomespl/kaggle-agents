"""Shared canonical-contract validation.

One implementation of the row/shape/fold/semantic rules, used by BOTH the
producer (``canonical_data_preparation_node``, which validates what it just
wrote using the arrays it already holds) and the consumer
(``DeveloperTargetSource``, which validates a claimed contract before any code
is generated against it). They used to keep separate copies with different
error shapes, so a contract could satisfy one and be refused by the other.

Validation is keyed by the declared representation kind. The real producers
write deliberately heterogeneous metadata - a media-filename contract has no
dense feature list, a packed image contract has no dense ``y`` at all - so a
single universal schema would refuse legitimate contracts.

Every function appends JSON-safe violation dicts instead of raising, so each
caller can decide how to report terminality.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from ..core.state.contracts import CanonicalDataContract
from .image_to_image_contract import validate_packed_canonical_contract


RepresentationKind = Literal["dense_tabular", "media_filename", "packed_image"]


def json_safe(value: object) -> object:
    """Coerce a value into something ``json.dumps`` accepts."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def json_safe_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    return {str(key): json_safe(value) for key, value in mapping.items()}


def describe_violations(
    violations: Sequence[Mapping[str, object]],
    summary: str,
) -> str:
    """One-line, JSON-safe rendering of the first few violations."""
    rendered = "; ".join(
        json.dumps(json_safe(violation), sort_keys=True, separators=(",", ":"))
        for violation in violations[:8]
    )
    return f"{summary}: {rendered}"


LOADER_METADATA_FIELDS = (
    "n_folds",
    "id_col",
    "target_col",
    "target_cols",
    "target_type",
    "n_targets",
)

REQUIRED_METADATA_BY_KIND: dict[str, tuple[str, ...]] = {
    "dense_tabular": (*LOADER_METADATA_FIELDS, "is_classification", "canonical_rows"),
    "media_filename": (*LOADER_METADATA_FIELDS, "is_classification", "canonical_rows"),
    "packed_image": (
        *LOADER_METADATA_FIELDS,
        "canonical_rows",
        "packed_image_contract",
        "task_type",
        "cv_strategy",
        "n_test",
    ),
}


def representation_kind_for(
    metadata: Mapping[str, object],
    contract: Mapping[str, object] | None = None,
) -> RepresentationKind:
    """Classify the declared canonical representation from what a producer wrote."""
    packed = bool(metadata.get("packed_image_contract")) or bool(
        (contract or {}).get("packed_image_contract")
    )
    if packed:
        return "packed_image"
    if metadata.get("target_source") or metadata.get("source"):
        return "media_filename"
    return "dense_tabular"


def violation(code: str, **details: object) -> dict[str, object]:
    return {"code": code, **json_safe_mapping(details)}


def load_canonical_array(path: Path, violations: list[dict[str, object]], name: str):
    try:
        return np.load(path, allow_pickle=True)
    except Exception as exc:
        violations.append(violation("unreadable_canonical_array", artifact=name, error=str(exc)))
        return None


def validate_metadata_agreement(  # noqa: PLR0912 - a flat list of contract-vs-metadata comparisons reads better than nested helpers
    contract: CanonicalDataContract,
    metadata: Mapping[str, object],
    state_metadata: Mapping[str, object] | None,
    kind: str,
    violations: list[dict[str, object]],
) -> None:
    missing = [
        field for field in REQUIRED_METADATA_BY_KIND[kind] if field not in metadata
    ]
    if missing:
        violations.append(violation("metadata_missing_required_fields", fields=missing))
        return

    comparisons: list[tuple[str, object, object]] = [
        ("n_folds", contract.n_folds, metadata.get("n_folds")),
        ("id_col", contract.id_col, metadata.get("id_col")),
        ("target_col", contract.target_col, metadata.get("target_col")),
        ("target_type", contract.target_type, metadata.get("target_type")),
        ("n_train", contract.n_train, metadata.get("canonical_rows")),
    ]
    for name, declared, observed in comparisons:
        if declared != observed:
            violations.append(
                violation(
                    "contract_metadata_disagreement",
                    field=name,
                    contract=declared,
                    metadata=observed,
                )
            )
    metadata_targets = metadata.get("target_cols")
    if list(contract.target_cols) != [str(value) for value in (metadata_targets or [])]:
        violations.append(
            violation(
                "contract_metadata_disagreement",
                field="target_cols",
                contract=list(contract.target_cols),
                metadata=metadata_targets,
            )
        )
    try:
        n_targets = int(metadata["n_targets"])
    except (TypeError, ValueError):
        violations.append(violation("invalid_metadata_field", field="n_targets"))
        n_targets = -1
    if n_targets != len(contract.target_cols):
        violations.append(
            violation(
                "target_count_disagreement",
                n_targets=n_targets,
                target_cols=list(contract.target_cols),
            )
        )
    if contract.target_type not in {"single", "multi_label", "multi_target"}:
        violations.append(violation("invalid_target_type", target_type=contract.target_type))
    if contract.target_cols and contract.target_col != contract.target_cols[0]:
        violations.append(violation("target_col_is_not_first_target"))
    if kind != "packed_image" and bool(metadata.get("is_classification")) != bool(
        contract.is_classification
    ):
        violations.append(violation("classification_flag_disagreement"))

    if isinstance(state_metadata, Mapping) and state_metadata:
        for field in ("id_col", "target_col", "target_type", "n_folds", "canonical_rows"):
            if field not in state_metadata or field not in metadata:
                continue
            if state_metadata[field] != metadata[field]:
                violations.append(
                    violation(
                        "state_metadata_disagreement",
                        field=field,
                        state=state_metadata[field],
                        metadata_json=metadata[field],
                    )
                )


def validate_feature_manifest(
    contract: CanonicalDataContract,
    metadata: Mapping[str, object],
    violations: list[dict[str, object]],
) -> None:
    path = Path(contract.feature_cols_path)
    try:
        feature_cols = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        violations.append(violation("invalid_feature_cols_json", error=str(exc)))
        return
    if not isinstance(feature_cols, list) or any(
        not isinstance(column, str) for column in feature_cols
    ):
        violations.append(violation("invalid_feature_cols_json", error="not a list of strings"))
        return
    declared = metadata.get("n_features")
    if declared is None:
        return
    try:
        declared_count = int(declared)
    except (TypeError, ValueError):
        violations.append(violation("invalid_metadata_field", field="n_features"))
        return
    if declared_count != len(feature_cols):
        violations.append(
            violation(
                "feature_manifest_disagreement",
                metadata_n_features=declared_count,
                feature_cols=len(feature_cols),
            )
        )


def validate_folds(
    contract: CanonicalDataContract,
    folds: np.ndarray,
    *,
    temporal: bool,
    violations: list[dict[str, object]],
) -> bool:
    """Validate fold dtype, emptiness, range and count. ``False`` stops the caller.

    Negative folds are legitimate ONLY in a temporal contract, where they mark
    warm-up rows that must stay out of every OOF prediction.
    """
    if not np.issubdtype(folds.dtype, np.integer):
        violations.append(violation("folds_are_not_integers", dtype=str(folds.dtype)))
        return False
    if folds.size == 0:
        violations.append(violation("empty_folds"))
        return False
    if not temporal and int(folds.min()) < 0:
        violations.append(violation("negative_folds_outside_temporal_contract"))
    assigned = sorted({int(value) for value in np.unique(folds) if int(value) >= 0})
    if assigned != list(range(contract.n_folds)):
        violations.append(
            violation(
                "fold_assignments_disagree_with_declared_fold_count",
                n_folds=contract.n_folds,
                observed=assigned,
            )
        )
    return True


def validate_train_ids(
    train_ids: np.ndarray,
    violations: list[dict[str, object]],
) -> None:
    """Canonical row names must be one scalar, unique key per row."""
    if train_ids.ndim != 1:
        violations.append(
            violation("train_ids_are_not_scalar", shape=list(train_ids.shape))
        )
        return
    try:
        keys = [str(value) for value in train_ids.tolist()]
    except TypeError:  # pragma: no cover - defensive
        violations.append(violation("train_ids_are_not_scalar"))
        return
    if keys and len(set(keys)) != len(keys):
        violations.append(violation("duplicate_train_ids", n_rows=len(keys)))


def validate_dense_rows(
    contract: CanonicalDataContract,
    metadata: Mapping[str, object],
    violations: list[dict[str, object]],
    arrays: tuple[object, object, object] | None = None,
) -> None:
    """Validate row counts, IDs, folds and target shape for a dense contract.

    ``arrays`` lets the producer pass ``(train_ids, folds, targets)`` it
    already holds in memory, so validating a freshly written multi-million-row
    contract does not re-read it from disk. Consumers pass ``None`` and the
    arrays are loaded once.
    """
    if arrays is None:
        train_ids = load_canonical_array(
            Path(contract.train_ids_path), violations, "train_ids"
        )
        folds = load_canonical_array(Path(contract.folds_path), violations, "folds")
        targets = load_canonical_array(Path(contract.y_path), violations, "y")
    else:
        train_ids, folds, targets = arrays
    if train_ids is None or folds is None or targets is None:
        return

    train_ids = np.asarray(train_ids)
    folds = np.asarray(folds)
    targets = np.asarray(targets)

    validate_train_ids(train_ids, violations)
    if folds.ndim != 1:
        violations.append(violation("folds_are_not_one_dimensional", shape=list(folds.shape)))
        return
    n_rows = int(train_ids.shape[0]) if train_ids.ndim == 1 else -1
    if not (len(folds) == len(targets) == n_rows):
        violations.append(
            violation(
                "row_count_disagreement",
                train_ids=n_rows,
                folds=len(folds),
                targets=len(targets),
            )
        )
        return
    if n_rows != contract.n_train:
        violations.append(
            violation("row_count_disagreement", contract=contract.n_train, arrays=n_rows)
        )
    temporal = contract.cv_strategy == "temporal_forward_chaining"
    if not validate_folds(contract, folds, temporal=temporal, violations=violations):
        return

    expected_shape = (
        (n_rows,)
        if contract.target_type == "single"
        else (n_rows, len(contract.target_cols))
    )
    if tuple(targets.shape) != expected_shape:
        violations.append(
            violation(
                "target_shape_disagreement",
                observed=list(targets.shape),
                expected=list(expected_shape),
            )
        )
        return

    if contract.is_classification and contract.target_type == "single":
        validate_class_order(metadata, targets, violations)

    if temporal:
        validate_temporal(contract, folds, violations)


def validate_class_order(
    metadata: Mapping[str, object],
    targets: np.ndarray,
    violations: list[dict[str, object]],
) -> None:
    class_order = metadata.get("class_order")
    if not isinstance(class_order, list) or not class_order:
        violations.append(violation("missing_class_order"))
        return
    declared = [str(value) for value in class_order]
    if len(set(declared)) != len(declared):
        violations.append(violation("duplicate_class_order", class_order=declared))
        return
    try:
        observed = {str(value) for value in targets.reshape(-1).tolist()}
    except TypeError:  # pragma: no cover - defensive
        violations.append(violation("non_scalar_target_labels"))
        return
    if observed != set(declared):
        violations.append(
            violation(
                "class_order_does_not_cover_labels",
                class_order=declared,
                observed=sorted(observed),
            )
        )


def validate_temporal(  # noqa: PLR0911 - each early return is one distinct temporal violation
    contract: CanonicalDataContract,
    folds: np.ndarray,
    violations: list[dict[str, object]],
) -> None:
    mask = load_canonical_array(Path(contract.oof_eligible_mask_path or ""), violations, "oof_eligible_mask")
    order = load_canonical_array(Path(contract.temporal_order_path or ""), violations, "temporal_order")
    if mask is None or order is None:
        return
    mask = np.asarray(mask, dtype=bool)
    order = np.asarray(order)
    if mask.shape != (len(folds),) or order.shape != (len(folds),):
        violations.append(
            violation(
                "temporal_arrays_are_not_row_aligned",
                folds=len(folds),
                mask=list(mask.shape),
                order=list(order.shape),
            )
        )
        return
    if not np.array_equal(folds >= 0, mask):
        violations.append(violation("temporal_folds_and_eligibility_disagree"))
    try:
        with np.load(Path(str(contract.temporal_splits_path)), allow_pickle=False) as splits:
            keys = set(splits.files)
            missing = [
                key
                for fold in range(contract.n_folds)
                for key in (f"train_{fold}", f"validation_{fold}")
                if key not in keys
            ]
            if missing:
                violations.append(violation("temporal_splits_missing_keys", keys=missing))
                return
            for fold in range(contract.n_folds):
                train_idx = np.asarray(splits[f"train_{fold}"], dtype=np.int64)
                val_idx = np.asarray(splits[f"validation_{fold}"], dtype=np.int64)
                if train_idx.size == 0 or val_idx.size == 0:
                    violations.append(violation("empty_temporal_partition", fold=fold))
                    return
                if np.intersect1d(train_idx, val_idx).size:
                    violations.append(violation("overlapping_temporal_partition", fold=fold))
                    return
                if order[train_idx].max() >= order[val_idx].min():
                    violations.append(violation("temporal_future_leakage", fold=fold))
                    return
                if not np.all(folds[val_idx] == fold):
                    violations.append(violation("temporal_validation_mismatch", fold=fold))
                    return
    except Exception as exc:
        violations.append(violation("unreadable_temporal_splits", error=str(exc)))


def validate_packed_representation(
    contract: CanonicalDataContract,
    violations: list[dict[str, object]],
) -> None:
    for message in validate_packed_canonical_contract(
        target_path=contract.y_path,
        train_ids_path=contract.train_ids_path,
        folds_path=contract.folds_path,
        test_ids_path=contract.test_ids_path,
        image_input_paths_path=contract.image_input_paths_path,
        image_test_input_paths_path=contract.image_test_input_paths_path,
        expected_n_train=contract.n_train,
        expected_n_test=contract.n_test,
    ):
        violations.append(violation("packed_image_contract_violation", detail=message))
