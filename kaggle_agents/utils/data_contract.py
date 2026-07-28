"""
Canonical Data Contract for Kaggle Agents.

This module provides a single "prepare once, consume many" data contract
that all model components must obey. It solves the problem of inconsistent
data handling across components (different sampling, filtering, column order).

Key artifacts generated:
- canonical/train_ids.npy - Stable row IDs after all filtering/sampling
- canonical/y.npy - Target aligned with train_ids
- canonical/folds.npy - Fold assignment per row
- canonical/feature_cols.json - Final feature list (intersection of train/test)
- canonical/metadata.json - Sampling info, original row count, etc.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from ..core.config import get_run_seed
from .target_inference import (
    TargetInferenceError,
    TargetType,
    infer_target_type_from_train,
    split_submission_schema,
)


# Common group column names for preventing data leakage
GROUP_COLUMN_CANDIDATES = [
    "PatientID", "patient_id", "patient", "subject_id", "subject",
    "StudyInstanceUID", "study_id", "SeriesInstanceUID", "series_id",
    "user_id", "userId", "group_id", "groupId", "session_id",
]

# Exact public-schema names that are strong enough to serve as temporal
# evidence for a task already declared as forecasting/time-series. Substring
# matching is intentionally avoided: a column such as ``lifetime_value`` must
# never silently change the evaluation protocol.
TEMPORAL_COLUMN_NAMES = {
    "date",
    "datetime",
    "timestamp",
    "time",
    "event_date",
    "event_time",
    "transaction_date",
    "order_date",
    "purchase_date",
    "created_at",
    "observed_at",
}
TEMPORAL_CONTRACT_KEYS = (
    "temporal_col",
    "time_col",
    "datetime_col",
    "timestamp_col",
    "order_col",
)

# Backward-compatible public constants. They describe generic task/identifier
# conventions only; no task maps to a benchmark-specific schema.
SEQ2SEQ_GROUP_CANDIDATES = ("_id", "Id", "ID")
SEQ2SEQ_TASK_INDICATORS: dict[str, dict[str, str]] = {
    "seq2seq": {},
    "text_normalization": {},
    "translation": {},
    "summarization": {},
}


def _detect_id_column(df: pd.DataFrame) -> str | None:
    """Detect the ID column in a dataframe."""
    candidates = ["id", "Id", "ID", "key", "Key", "index"]
    for col in candidates:
        if col in df.columns:
            return col
    # Fallback: first column if it looks like an ID
    first_col = df.columns[0]
    if df[first_col].nunique() == len(df):
        return first_col
    return None


def _detect_group_column(df: pd.DataFrame) -> str | None:
    """Auto-detect group column for GroupKFold to prevent data leakage."""
    for col in GROUP_COLUMN_CANDIDATES:
        if col in df.columns:
            n_unique = df[col].nunique()
            n_rows = len(df)
            if n_unique < n_rows * 0.9:  # At least 10% rows share groups
                return col
    return None


def _detect_seq2seq_group_column(df: pd.DataFrame) -> str | None:
    """
    Detect a repeated identifier suitable for grouped seq2seq validation.

    Args:
        df: DataFrame to check for seq2seq group columns

    Returns:
        Name of the seq2seq group column if found, None otherwise
    """
    if df.empty:
        return None

    n_rows = len(df)
    candidates: list[tuple[str, int]] = []
    for col in df.columns:
        name = str(col)
        looks_like_identifier = (
            name.lower() == "id"
            or name.lower().endswith("_id")
            or name.endswith(("Id", "ID"))
        )
        if not looks_like_identifier:
            continue

        n_unique = int(df[col].nunique(dropna=True))
        unique_ratio = n_unique / n_rows
        # Very-low-cardinality counters (for example, token positions) are not
        # safe grouping keys. One-to-one identifiers are not groups either.
        if n_unique > 1 and 0.01 <= unique_ratio < 0.9:
            candidates.append((name, n_unique))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[1], reverse=True)
    if len(candidates) > 1 and candidates[0][1] == candidates[1][1]:
        # Two equally plausible identifiers are ambiguous; do not guess a CV
        # grouping key because the wrong choice can itself introduce leakage.
        return None
    return candidates[0][0]


def _read_sample_submission_columns(
    sample_submission: str | Path | pd.DataFrame | None,
) -> list[str]:
    """Read only the schema needed for column resolution."""
    if sample_submission is None:
        return []
    if isinstance(sample_submission, pd.DataFrame):
        return [str(col) for col in sample_submission.columns]
    return [
        str(col)
        for col in pd.read_csv(Path(sample_submission), nrows=0).columns
    ]


def _contract_column(
    contract: Mapping[str, Any] | None,
    *keys: str,
) -> str | None:
    """Return the first non-empty column name present in a public contract."""
    if not contract:
        return None
    for key in keys:
        value = contract.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _is_temporal_task(task_type: str) -> bool:
    """Return whether the explicit task family requires temporal validation."""
    normalized = str(task_type or "").strip().lower().replace("-", "_")
    return any(
        marker in normalized
        for marker in ("time_series", "forecast", "temporal")
    )


def _parse_temporal_order(
    values: pd.Series,
    *,
    column_name: str,
    allow_numeric_order: bool,
) -> tuple[np.ndarray, str]:
    """Parse a complete, auditable temporal ordering vector.

    Numeric order is accepted only for a column explicitly named by a contract.
    Data-derived candidates must be parseable datetimes. Missing/unparseable
    values are rejected because their placement relative to a cutoff is unknown.
    """
    if values.isna().any():
        raise ValueError(
            f"Temporal column {column_name!r} contains missing values; "
            "cannot prove train-before-validation ordering"
        )

    if pd.api.types.is_numeric_dtype(values.dtype):
        if not allow_numeric_order:
            raise ValueError(
                f"Data-derived temporal column {column_name!r} is numeric. "
                "Declare it explicitly as temporal_col/order_col in the public "
                "column contract before using numeric order."
            )
        order = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(order)):
            raise ValueError(
                f"Temporal order column {column_name!r} contains non-finite values"
            )
        return order, "numeric_order"

    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        invalid_count = int(parsed.isna().sum())
        raise ValueError(
            f"Temporal column {column_name!r} has {invalid_count} unparseable "
            "values; cannot form a trustworthy temporal CV contract"
        )
    return parsed.astype("int64").to_numpy(dtype=np.int64), "datetime_utc_ns"


def _resolve_temporal_order(
    train_df: pd.DataFrame,
    *,
    temporal_col: str | None,
    column_contract: Mapping[str, Any] | None,
) -> tuple[str, np.ndarray, str, str]:
    """Resolve temporal order from an explicit contract or unambiguous schema.

    Resolution never uses competition identity, target values, or a fuzzy name
    match. If the public data exposes multiple plausible time axes, the caller
    must select one explicitly rather than letting the agent guess.
    """
    explicit_col = temporal_col or _contract_column(
        column_contract, *TEMPORAL_CONTRACT_KEYS
    )
    if explicit_col:
        if explicit_col not in train_df.columns:
            raise ValueError(
                f"Declared temporal column {explicit_col!r} is absent from "
                f"training data columns {list(train_df.columns)!r}"
            )
        order, value_type = _parse_temporal_order(
            train_df[explicit_col],
            column_name=explicit_col,
            allow_numeric_order=True,
        )
        evidence_source = (
            "explicit_argument"
            if temporal_col
            else "public_column_contract"
        )
        return explicit_col, order, value_type, evidence_source

    candidates: list[tuple[str, np.ndarray, str]] = []
    for raw_col in train_df.columns:
        col = str(raw_col)
        normalized = col.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized not in TEMPORAL_COLUMN_NAMES:
            continue
        try:
            order, value_type = _parse_temporal_order(
                train_df[col],
                column_name=col,
                allow_numeric_order=False,
            )
        except ValueError:
            continue
        candidates.append((col, order, value_type))

    if not candidates:
        raise ValueError(
            "Temporal task requires an explicit, complete temporal/order "
            "column. Provide temporal_col/order_col in the public column "
            "contract or a single parseable datetime column with an exact "
            f"temporal schema name ({sorted(TEMPORAL_COLUMN_NAMES)!r})."
        )
    if len(candidates) > 1:
        raise ValueError(
            "Temporal task has multiple plausible time axes "
            f"{[candidate[0] for candidate in candidates]!r}; declare exactly "
            "one temporal_col/order_col in the public column contract."
        )

    col, order, value_type = candidates[0]
    return col, order, value_type, "unambiguous_public_schema"


def _audit_time_value(value: int | float, value_type: str) -> str | int | float:
    """Convert a normalized order value into stable JSON audit metadata."""
    if value_type == "datetime_utc_ns":
        return pd.Timestamp(int(value), unit="ns", tz="UTC").isoformat()
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _build_forward_chaining_splits(
    order_values: np.ndarray,
    *,
    n_folds: int,
    value_type: str,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """Build expanding-window splits on whole timestamp groups.

    The oldest block is warm-up history and intentionally has no OOF
    prediction. Every later row appears in exactly one validation fold, and
    every fold satisfies ``max(train_time) < min(validation_time)``.
    """
    if n_folds < 2:
        raise ValueError("Temporal CV requires at least two forward splits")
    if order_values.ndim != 1 or len(order_values) == 0:
        raise ValueError("Temporal order must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(order_values)):
        raise ValueError("Temporal order contains NaN or Inf")

    unique_times = np.unique(order_values)
    required_unique = n_folds + 1
    if len(unique_times) < required_unique:
        raise ValueError(
            "Temporal CV cannot form the requested forward-chaining contract: "
            f"{len(unique_times)} distinct ordered values for {n_folds} folds "
            f"(need at least {required_unique})."
        )

    time_blocks = np.array_split(unique_times, required_unique)
    if any(len(block) == 0 for block in time_blocks):
        raise ValueError("Temporal CV produced an empty time block")

    fold_assignments = np.full(len(order_values), -1, dtype=np.int32)
    validation_counts = np.zeros(len(order_values), dtype=np.int32)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    cutoffs: list[dict[str, Any]] = []

    for fold_idx in range(n_folds):
        train_times = np.concatenate(time_blocks[: fold_idx + 1])
        val_times = time_blocks[fold_idx + 1]
        train_idx = np.flatnonzero(np.isin(order_values, train_times))
        val_idx = np.flatnonzero(np.isin(order_values, val_times))

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(
                f"Temporal fold {fold_idx} has an empty train/validation partition"
            )
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError(f"Temporal fold {fold_idx} train/validation overlap")

        train_max = order_values[train_idx].max()
        val_min = order_values[val_idx].min()
        if not train_max < val_min:
            raise ValueError(
                f"Temporal fold {fold_idx} violates strict ordering: "
                f"train_max={train_max!r}, validation_min={val_min!r}"
            )

        fold_assignments[val_idx] = fold_idx
        validation_counts[val_idx] += 1
        splits.append((train_idx.astype(np.int64), val_idx.astype(np.int64)))
        cutoffs.append(
            {
                "fold": fold_idx,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "train_time_min": _audit_time_value(
                    order_values[train_idx].min(), value_type
                ),
                "train_time_max": _audit_time_value(train_max, value_type),
                "validation_time_min": _audit_time_value(val_min, value_type),
                "validation_time_max": _audit_time_value(
                    order_values[val_idx].max(), value_type
                ),
            }
        )

    eligible_mask = fold_assignments >= 0
    if not np.all(validation_counts[eligible_mask] == 1):
        raise ValueError(
            "Temporal validation rows must appear in exactly one validation fold"
        )
    if np.any(validation_counts[~eligible_mask] != 0):
        raise ValueError("Temporal warm-up rows unexpectedly entered validation")
    if int(eligible_mask.sum()) + int((~eligible_mask).sum()) != len(order_values):
        raise ValueError("Temporal coverage metadata is inconsistent")

    return fold_assignments, eligible_mask, splits, cutoffs


def infer_seq2seq_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    *,
    target_col: str | None = None,
    source_col: str | None = None,
    class_col: str | None = None,
    seq2seq_group_col: str | None = None,
    id_col: str | None = None,
    sample_submission: str | Path | pd.DataFrame | None = None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, str | None]:
    """Resolve seq2seq column roles without assuming a competition schema.

    Resolution order is explicit arguments, canonical/submission contract,
    sample-submission plus train/test schema, and finally unambiguous
    data-derived structure. Ambiguous source/target/context roles raise
    ``ValueError`` rather than silently selecting a benchmark-shaped default.
    """
    if train_df.empty:
        raise ValueError("Cannot infer seq2seq columns from empty training data")

    train_columns = [str(col) for col in train_df.columns]
    test_columns = (
        [str(col) for col in test_df.columns]
        if test_df is not None
        else []
    )
    train_set = set(train_columns)
    test_set = set(test_columns)
    sample_columns = _read_sample_submission_columns(sample_submission)

    resolved_target = target_col or _contract_column(
        contract, "target_col", "output_col"
    )
    if resolved_target not in train_set:
        resolved_target = None

    if resolved_target is None:
        train_only = [col for col in train_columns if col not in test_set]
        sample_targets = sample_columns[1:] if len(sample_columns) > 1 else []
        sample_matches = [
            col for col in sample_targets if col in train_set and col not in test_set
        ]
        if len(sample_matches) == 1:
            resolved_target = sample_matches[0]
        elif len(train_only) == 1:
            resolved_target = train_only[0]
        else:
            raise ValueError(
                "Ambiguous seq2seq target column. Provide target_col or a "
                "canonical/submission contract; train-only candidates were "
                f"{train_only!r}."
            )

    resolved_id = id_col or _contract_column(contract, "id_col", "id_column")
    if resolved_id not in train_set:
        sample_id = sample_columns[0] if sample_columns else None
        id_candidates = [
            col
            for col in train_columns
            if col.lower() in {"id", "key", "index"}
            or col.lower().endswith("_id")
        ]
        exact_id_candidates = [
            col for col in id_candidates if col.lower() in {"id", "key", "index"}
        ]
        unique_id_candidates = [
            col
            for col in id_candidates
            if train_df[col].nunique(dropna=False) == len(train_df)
        ]
        if sample_id in train_set:
            resolved_id = sample_id
        elif len(exact_id_candidates) == 1:
            resolved_id = exact_id_candidates[0]
        elif len(unique_id_candidates) == 1:
            resolved_id = unique_id_candidates[0]
        elif len(id_candidates) == 1:
            resolved_id = id_candidates[0]
        else:
            # A unique text source is common in seq2seq data and must not be
            # mistaken for an ID merely because it is the first column.
            resolved_id = None

    resolved_group = seq2seq_group_col or _contract_column(
        contract, "seq2seq_group_col", "group_col"
    )
    if resolved_group not in train_set:
        resolved_group = _detect_seq2seq_group_column(train_df)

    resolved_source = source_col or _contract_column(
        contract, "source_col", "input_col"
    )
    if resolved_source not in train_set:
        resolved_source = None

    resolved_class = class_col or _contract_column(
        contract, "class_col", "context_col", "type_col"
    )
    if resolved_class not in train_set:
        resolved_class = None

    common_columns = [
        col for col in train_columns if not test_set or col in test_set
    ]
    excluded = {
        value
        for value in (resolved_target, resolved_id, resolved_group, resolved_class)
        if value is not None
    }
    feature_candidates = [col for col in common_columns if col not in excluded]

    def is_textual(col: str) -> bool:
        series = train_df[col]
        return bool(
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )

    def is_low_cardinality(col: str) -> bool:
        n_unique = int(train_df[col].nunique(dropna=True))
        limit = max(20, int(np.sqrt(len(train_df))))
        return n_unique <= limit or n_unique / len(train_df) <= 0.01

    textual_candidates = [col for col in feature_candidates if is_textual(col)]
    if resolved_source is None:
        if len(textual_candidates) == 1:
            resolved_source = textual_candidates[0]
        else:
            non_context_text = [
                col for col in textual_candidates if not is_low_cardinality(col)
            ]
            if len(non_context_text) == 1:
                resolved_source = non_context_text[0]
            elif textual_candidates:
                cardinalities = sorted(
                    (
                        (col, int(train_df[col].nunique(dropna=True)))
                        for col in textual_candidates
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if (
                    len(cardinalities) == 1
                    or cardinalities[0][1] >= max(
                        2, cardinalities[1][1] * 4
                    )
                ):
                    resolved_source = cardinalities[0][0]

    if resolved_source is None:
        raise ValueError(
            "Ambiguous seq2seq source column. Provide source_col or a canonical "
            f"contract; textual feature candidates were {textual_candidates!r}."
        )
    if resolved_source == resolved_target:
        raise ValueError("Seq2seq source and target columns must be different")

    if resolved_class is None:
        context_candidates = [
            col
            for col in feature_candidates
            if col != resolved_source
            and is_low_cardinality(col)
            and is_textual(col)
        ]
        if len(context_candidates) == 1:
            resolved_class = context_candidates[0]
        elif len(context_candidates) > 1:
            raise ValueError(
                "Ambiguous seq2seq context column. Provide class_col/context_col "
                f"in the canonical contract; candidates were {context_candidates!r}."
            )

    return {
        "source_col": resolved_source,
        "target_col": resolved_target,
        "class_col": resolved_class,
        "seq2seq_group_col": resolved_group,
        "id_col": resolved_id,
    }


def validate_schema_parity(
    train_path: str | Path,
    test_path: str | Path,
    id_col: str | None = None,
    target_col: str | None = None,
    target_cols: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Validate that train and test have compatible schemas.

    Returns:
        Tuple of (common_feature_cols, missing_in_test)
    """
    train_cols = set(pd.read_csv(train_path, nrows=0).columns)
    test_cols = set(pd.read_csv(test_path, nrows=0).columns)

    # Columns to exclude from features
    exclude_cols = set()
    if id_col:
        exclude_cols.add(id_col)
    if target_col:
        exclude_cols.add(target_col)
    exclude_cols.update(str(column) for column in (target_cols or []))

    # Feature columns = intersection (excluding id/target)
    common = train_cols & test_cols - exclude_cols
    missing_in_test = train_cols - test_cols - exclude_cols

    # Deterministic order - convert to str to handle mixed types (str/float in column names)
    # This can happen with CSVs that have numeric column headers
    common_str = [str(c) for c in common]
    missing_str = [str(c) for c in missing_in_test]
    return sorted(common_str), sorted(missing_str)


def _supplied_test_columns(test_path: str | Path | None) -> set[str]:
    """Return public test columns that actually carry values.

    A column the test set provides is model input, not a prediction. Columns
    present but entirely empty are placeholders and carry no such evidence, so
    they are excluded to keep templates that ship a blank target column from
    being read as already-answered.
    """
    if not test_path:
        return set()
    try:
        sample = pd.read_csv(Path(test_path), nrows=1000)
    except Exception:
        return set()
    return {
        str(column)
        for column in sample.columns
        if bool(sample[column].notna().any())
    }


def _materialize_synthetic_id(
    table_path: str | Path | None,
    id_col: str,
) -> bool:
    """Write the synthetic identifier into a staged table, in place.

    The synthetic name is the row's position, so adding it is information the
    table already carried implicitly. Writing it makes the contract honest:
    every name components are told about is one they can actually look up.

    Returns:
        True when the column was added, False when it was already present or
        the table could not be rewritten.
    """
    if not table_path:
        return False
    path = Path(table_path)
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    if id_col in frame.columns:
        return False
    frame.insert(0, id_col, [str(index) for index in range(len(frame))])
    try:
        frame.to_csv(path, index=False)
    except Exception:
        return False
    print(f"   Materialized {id_col!r} into {path.name} ({len(frame):,} rows)")
    return True


def _resolve_canonical_test_ids(
    test_path: str | Path | None,
    id_col: str | None,
) -> tuple[np.ndarray | None, bool]:
    """Name every public test row exactly once.

    Prefers a unique key the test table already provides; falls back to row
    position when it provides none, which is the only identity such a
    competition actually has.

    Returns:
        Tuple of (ids as strings, whether they are positional). ``(None, False)``
        when the test table cannot be read as a table at all.
    """
    if not test_path:
        return None, False
    try:
        test_df = pd.read_csv(Path(test_path))
    except Exception:
        return None, False

    # Only a declared or conventionally named identifier counts. A free-text
    # field can be incidentally unique without being an identifier, and naming
    # rows by it would break the moment two rows share the same text.
    preferred = [str(id_col)] if id_col else []
    preferred += [
        str(column)
        for column in test_df.columns
        if str(column).lower() in {"id", "index", "key", "row_id"}
    ]
    for column in preferred:
        if column not in test_df.columns:
            continue
        values = test_df[column]
        if values.isna().any():
            continue
        as_text = values.astype(str)
        if as_text.duplicated().any():
            continue
        return np.asarray(as_text.tolist(), dtype=str), False

    positional = np.asarray([str(i) for i in range(len(test_df))], dtype=str)
    return positional, True


def _resolve_supervised_target_contract(
    train_df: pd.DataFrame,
    test_path: Path,
    *,
    target_col: str | None,
    target_cols: list[str] | None,
    target_type: TargetType | None,
    task_type: str,
    sample_submission: str | Path | pd.DataFrame | None,
    column_contract: Mapping[str, Any] | None,
) -> tuple[list[str], TargetType, str]:
    """Resolve ordered training targets from public schemas and real labels."""
    train_columns = [str(column) for column in train_df.columns]
    train_set = set(train_columns)
    sample_columns = _read_sample_submission_columns(sample_submission)
    # Columns the public test set supplies are inputs echoed back by the
    # template, never predictions.
    public_test_columns = _supplied_test_columns(test_path)
    sample_targets: list[str] = []
    if len(sample_columns) > 1:
        _, sample_targets = split_submission_schema(
            sample_columns,
            public_test_columns,
        )

    contract_targets: list[str] = []
    if column_contract:
        raw_contract_targets = column_contract.get("target_cols")
        if isinstance(raw_contract_targets, (list, tuple)):
            contract_targets = [
                str(column)
                for column in raw_contract_targets
                if isinstance(column, str) and column
            ]
    declared_targets = [
        str(column)
        for column in list(target_cols or contract_targets)
        if isinstance(column, str) and column
    ]
    if len(declared_targets) != len(set(declared_targets)):
        raise TargetInferenceError("Declared target_cols contain duplicates")

    # An upstream contract that names a supplied test column as a target was
    # resolved positionally and is wrong. Drop those names so resolution falls
    # through to schema evidence instead of scoring an input column.
    if public_test_columns:
        declared_targets = [
            column
            for column in declared_targets
            if column not in public_test_columns
        ]
        if target_col and str(target_col) in public_test_columns:
            target_col = None

    resolved_targets: list[str] = []
    if (
        sample_targets
        and len(sample_targets) > 1
        and all(column in train_set for column in sample_targets)
    ):
        # The public submission order is the canonical prediction/label order.
        resolved_targets = sample_targets
    elif declared_targets and all(
        column in train_set for column in declared_targets
    ):
        if sample_targets and set(declared_targets) == set(sample_targets):
            resolved_targets = sample_targets
        else:
            resolved_targets = declared_targets
    elif target_col and str(target_col) in train_set:
        resolved_targets = [str(target_col)]
    else:
        contract_target = _contract_column(
            column_contract,
            "target_col",
            "output_col",
        )
        if (
            contract_target
            and contract_target in train_set
            and contract_target not in public_test_columns
        ):
            resolved_targets = [contract_target]

    if not resolved_targets:
        try:
            test_columns = {
                str(column)
                for column in pd.read_csv(test_path, nrows=0).columns
            }
        except Exception as exc:
            raise TargetInferenceError(
                f"Cannot inspect public test schema for target resolution: {exc}"
            ) from exc
        train_only = [
            column for column in train_columns if column not in test_columns
        ]
        if (
            sample_targets
            and set(sample_targets).issubset(train_only)
            and len(sample_targets) == len(train_only)
        ):
            resolved_targets = sample_targets
        elif len(train_only) == 1:
            resolved_targets = train_only
        else:
            raise TargetInferenceError(
                "Ambiguous training target columns. Provide an explicit public "
                "target_cols/target_col contract; train-only candidates were "
                f"{train_only!r} and submission outputs were {sample_targets!r}."
            )

    explicit_target_type = target_type
    if explicit_target_type is None and column_contract:
        raw_target_type = column_contract.get("target_type")
        if raw_target_type in {"single", "multi_label", "multi_target"}:
            explicit_target_type = raw_target_type

    resolved_type, type_source = infer_target_type_from_train(
        train_df,
        resolved_targets,
        problem_type=task_type,
        explicit_target_type=explicit_target_type,
    )
    return resolved_targets, resolved_type, type_source


def _build_multilabel_fold_assignments(
    y: np.ndarray,
    *,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Deterministically balance independent binary labels across folds.

    This is a lightweight iterative-stratification equivalent that greedily
    assigns rare/high-cardinality rows to the fold with the largest current
    per-label deficit, then balances fold sizes as a deterministic tie-breaker.
    """
    labels = np.asarray(y, dtype=np.int8)
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("Multilabel stratification requires a 2D target matrix")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Multilabel stratification requires binary indicators")
    if n_folds < 2 or len(labels) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds} multilabel folds from {len(labels)} rows"
        )

    rng = np.random.default_rng(seed)
    label_totals = labels.sum(axis=0).astype(float)
    desired_labels = label_totals / n_folds
    desired_size = len(labels) / n_folds
    rarity = np.where(label_totals > 0, 1.0 / label_totals, 0.0)
    row_priority = labels @ rarity
    tie_breakers = rng.random(len(labels))
    order = np.lexsort(
        (
            tie_breakers,
            -labels.sum(axis=1),
            -row_priority,
        )
    )

    fold_sizes = np.zeros(n_folds, dtype=int)
    fold_label_counts = np.zeros((n_folds, labels.shape[1]), dtype=float)
    assignments = np.full(len(labels), -1, dtype=np.int32)
    for row_index in order:
        positive = labels[row_index].astype(bool)
        label_deficit = (
            desired_labels[None, :] - fold_label_counts
        )[:, positive].sum(axis=1)
        size_deficit = desired_size - fold_sizes
        combined = label_deficit + size_deficit / max(
            1.0,
            float(labels.shape[1]),
        )
        best = np.flatnonzero(np.isclose(combined, combined.max()))
        if len(best) > 1:
            smallest = fold_sizes[best].min()
            best = best[fold_sizes[best] == smallest]
        chosen = int(best[0])
        assignments[row_index] = chosen
        fold_sizes[chosen] += 1
        fold_label_counts[chosen] += labels[row_index]

    if np.any(assignments < 0) or np.any(fold_sizes == 0):
        raise ValueError("Multilabel fold assignment did not cover every row")
    return assignments


def select_cv_strategy(
    n_rows: int,
    timeout_s: int | None = None,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """
    Select CV strategy based on dataset size and budget.

    Args:
        n_rows: Number of training rows
        timeout_s: Component timeout in seconds
        fast_mode: Whether running in fast mode

    Returns:
        Dict with n_folds and strategy name
    """
    if fast_mode or n_rows > 2_000_000:
        return {"n_folds": 3, "strategy": "kfold"}
    if n_rows > 500_000:
        return {"n_folds": 3, "strategy": "stratified_kfold"}
    if n_rows > 200_000:
        return {"n_folds": 4, "strategy": "stratified_kfold"}
    return {"n_folds": 5, "strategy": "stratified_kfold"}


def _deterministic_hash(value: str, seed: int = 42) -> int:
    """
    Deterministic hash using MD5 + seed.

    Unlike Python's built-in hash(), this produces the same result
    across different Python processes (PYTHONHASHSEED independent).

    Args:
        value: String value to hash
        seed: Random seed for reproducibility

    Returns:
        Integer hash value (0 to 2^32-1)
    """
    combined = f"{seed}_{value}"
    return int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)


def _ensure_id_column(
    df: pd.DataFrame,
    id_col: str | None,
) -> tuple[pd.DataFrame, str, bool]:
    """
    Ensure a valid ID column exists for deterministic sampling.

    If no ID column is found, creates a synthetic '_row_id' based on
    the original row index. This MUST be done BEFORE any transformations.

    Args:
        df: DataFrame to check
        id_col: Detected or specified ID column name

    Returns:
        Tuple of (df_with_id, id_col_name, is_synthetic)
    """
    is_synthetic = False

    if id_col is None or id_col not in df.columns:
        # Create synthetic ID based on original index (preserves order)
        # IMPORTANT: Must be done BEFORE any transformation/shuffle
        df = df.copy()
        df["_row_id"] = df.index.astype(str)
        id_col = "_row_id"
        is_synthetic = True
        print("[LOG:WARN] No ID column found, using synthetic '_row_id' for sampling")

    return df, id_col, is_synthetic


def _remove_synthetic_id_from_features(
    df: pd.DataFrame,
    is_synthetic: bool,
) -> pd.DataFrame:
    """
    Remove synthetic _row_id from features AFTER artifacts are generated.

    Args:
        df: DataFrame potentially containing _row_id
        is_synthetic: Whether the ID was synthetically created

    Returns:
        DataFrame without _row_id column (if synthetic)
    """
    if is_synthetic and "_row_id" in df.columns:
        return df.drop(columns=["_row_id"])
    return df


def _hash_based_sample(
    df: pd.DataFrame,
    id_col: str | None,
    max_rows: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Deterministic hash-based sampling using MD5.

    Uses MD5 hash of ID (not Python's built-in hash) to ensure
    the same rows are selected across different Python processes.

    Args:
        df: DataFrame to sample
        id_col: ID column name (can be None, will be created)
        max_rows: Maximum rows to keep
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sampled_df, sampling_metadata)
    """
    # Step 1: Ensure ID column exists
    df, id_col, is_synthetic = _ensure_id_column(df, id_col)

    original_rows = len(df)
    if original_rows <= max_rows:
        return df, {
            "sampled": False,
            "original_rows": original_rows,
            "id_column": id_col,
            "id_is_synthetic": is_synthetic,
        }

    # Step 2: Calculate threshold for sampling
    sample_frac = max_rows / original_rows
    threshold = int(10000 * sample_frac)

    # Step 3: Apply deterministic MD5 hash
    def should_include(id_val):
        return (_deterministic_hash(str(id_val), seed) % 10000) < threshold

    sample_mask = df[id_col].apply(should_include).values
    sampled_df = df[sample_mask].reset_index(drop=True)

    # Step 4: Keep _row_id for now (needed for alignment)
    # Will be removed AFTER generating artifacts (folds, train_ids)

    metadata = {
        "sampled": True,
        "original_rows": original_rows,
        "sampled_rows": len(sampled_df),
        "sampling_method": "hash_based_md5",
        "sampling_threshold": threshold,
        "sampling_seed": seed,
        "hash_method": "md5",
        "id_column": id_col,
        "id_is_synthetic": is_synthetic,
        "deterministic": True,
        # Note: canonical_version is set in prepare_canonical_data, not here
    }

    return sampled_df, metadata


def prepare_canonical_data(
    train_path: str | Path,
    test_path: str | Path,
    target_col: str | None,
    output_dir: str | Path,
    id_col: str | None = None,
    target_cols: list[str] | None = None,
    target_type: TargetType | None = None,
    max_rows: int | None = None,
    n_folds: int | None = None,
    fast_mode: bool = False,
    timeout_s: int | None = None,
    # Seq2seq specific parameters
    task_type: str = "tabular",
    source_col: str | None = None,
    class_col: str | None = None,
    seq2seq_group_col: str | None = None,
    temporal_col: str | None = None,
    sample_submission: str | Path | pd.DataFrame | None = None,
    column_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Prepare canonical data artifacts that all model components must use.

    This is the single source of truth for:
    - Which rows to use (train_ids)
    - Target values (y)
    - Fold assignments (folds)
    - Feature columns (feature_cols)

    Args:
        train_path: Path to training data
        test_path: Path to test data
        target_col: Name of target column
        target_cols: Ordered training target columns for multi-output tasks
        target_type: Explicit public target semantics, when available
        output_dir: Working directory for competition
        id_col: ID column name (auto-detected if None)
        max_rows: Maximum rows to use (hash-based sampling if exceeded)
        n_folds: Number of CV folds (auto-selected if None)
        fast_mode: Whether running in fast mode
        timeout_s: Component timeout in seconds
        task_type: Type of task ("tabular", "seq2seq", "text_normalization", etc.)
        source_col: Source column for seq2seq tasks (derived when omitted)
        class_col: Optional context/type column for conditional mappings
        seq2seq_group_col: Repeated record/group ID used to prevent CV leakage
        temporal_col: Explicit public temporal/order column for forecasting
        sample_submission: Optional public submission schema for target/ID roles
        column_contract: Optional canonical mapping of column names to roles

    Returns:
        Dict with paths to all canonical artifacts
    """
    train_path = Path(train_path)
    test_path = Path(test_path)
    output_dir = Path(output_dir)

    # Create canonical directory
    canonical_dir = output_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    print("\n   Preparing canonical data contract...")
    run_seed = get_run_seed()

    # Step 1: Load training data RAW (no transformations yet)
    train_df = pd.read_csv(train_path)
    original_rows = len(train_df)
    print(f"   Loaded {original_rows:,} training rows")

    # Step 1.5: Detect seq2seq task type and auto-configure columns
    is_seq2seq = task_type in ("seq2seq", "text_normalization", "translation", "summarization")

    if is_seq2seq:
        test_schema_df = pd.read_csv(test_path, nrows=10_000)
        resolved_columns = infer_seq2seq_columns(
            train_df,
            test_schema_df,
            target_col=target_col,
            source_col=source_col,
            class_col=class_col,
            seq2seq_group_col=seq2seq_group_col,
            id_col=id_col,
            sample_submission=sample_submission,
            contract=column_contract,
        )
        target_col = str(resolved_columns["target_col"])
        source_col = str(resolved_columns["source_col"])
        class_col = resolved_columns["class_col"]
        seq2seq_group_col = resolved_columns["seq2seq_group_col"]
        id_col = resolved_columns["id_col"]

        if seq2seq_group_col:
            print(f"   Detected seq2seq group column: {seq2seq_group_col}")

        print(f"   Task type: {task_type} (seq2seq)")
        print(f"   Source column: {source_col}")
        print(f"   Target column: {target_col}")
        if class_col:
            print(f"   Context column: {class_col}")
        resolved_target_cols = [target_col]
        resolved_target_type: TargetType = "single"
        target_type_source = "seq2seq_contract"
    else:
        (
            resolved_target_cols,
            resolved_target_type,
            target_type_source,
        ) = _resolve_supervised_target_contract(
            train_df,
            test_path,
            target_col=target_col,
            target_cols=target_cols,
            target_type=target_type,
            task_type=task_type,
            sample_submission=sample_submission,
            column_contract=column_contract,
        )
        target_col = resolved_target_cols[0]
        print(
            "   Training targets: "
            f"{resolved_target_cols} ({resolved_target_type})"
        )

    # Step 2: Detect ID column BEFORE any operations
    if id_col is None and not is_seq2seq:
        id_col = _detect_id_column(train_df)

    # Validate target exists
    missing_targets = [
        column
        for column in resolved_target_cols
        if column not in train_df.columns
    ]
    if missing_targets:
        raise TargetInferenceError(
            f"Target columns not found in training data: {missing_targets}"
        )

    # Step 3: Ensure ID exists and apply sampling (proper order)
    # The _hash_based_sample function handles _ensure_id_column internally
    is_synthetic_id = False
    sampling_metadata = {"sampled": False, "original_rows": original_rows}

    if max_rows and len(train_df) > max_rows:
        train_df, sampling_metadata = _hash_based_sample(
            train_df, id_col, max_rows, seed=run_seed
        )
        id_col = sampling_metadata.get("id_column", id_col)
        is_synthetic_id = sampling_metadata.get("id_is_synthetic", False)
        print(
            f"   Sampled {len(train_df):,} rows via deterministic MD5 hash "
            f"(seed={run_seed})"
        )
    else:
        # Even without sampling, ensure ID column exists
        train_df, id_col, is_synthetic_id = _ensure_id_column(train_df, id_col)
        sampling_metadata["id_column"] = id_col
        sampling_metadata["id_is_synthetic"] = is_synthetic_id

    if is_synthetic_id:
        print(f"   Using synthetic ID column: {id_col}")
        # Materialize it in the staged tables. A name that appears in the
        # contract but in no file is the single most expensive failure mode
        # here: components index by it, raise KeyError, and burn their whole
        # repair budget on a column that never existed.
        _materialize_synthetic_id(train_path, id_col)
        _materialize_synthetic_id(test_path, id_col)
    else:
        print(f"   Using ID column: {id_col}")

    # Schema parity check
    feature_cols, missing_in_test = validate_schema_parity(
        train_path,
        test_path,
        id_col,
        target_col,
        target_cols=resolved_target_cols,
    )

    # Remove synthetic _row_id from feature columns if present
    if is_synthetic_id and "_row_id" in feature_cols:
        feature_cols.remove("_row_id")

    if missing_in_test:
        print(f"   Warning: {len(missing_in_test)} columns missing in test: {missing_in_test[:5]}...")

    print(f"   Using {len(feature_cols)} feature columns")

    normalized_task_type = str(task_type).strip().lower()
    is_temporal = _is_temporal_task(normalized_task_type)

    # Select CV strategy. A temporal task may change the fold count for budget
    # reasons, but it must never change the strategy to shuffled KFold.
    if n_folds is None:
        cv_config = select_cv_strategy(len(train_df), timeout_s, fast_mode)
        n_folds = cv_config["n_folds"]
    else:
        cv_config = {"n_folds": n_folds, "strategy": "stratified_kfold"}
    if is_temporal:
        cv_config["strategy"] = "temporal_forward_chaining"

    # Detect group column for preventing data leakage
    # For seq2seq tasks, prioritize seq2seq-specific group column (e.g., sentence_id)
    if is_temporal:
        group_col = None
    elif is_seq2seq and seq2seq_group_col:
        group_col = seq2seq_group_col
        print(f"   Using seq2seq group column: {group_col} (GroupKFold to prevent data leakage)")
    else:
        group_col = _detect_group_column(train_df)
        if group_col:
            print(f"   Detected group column: {group_col} (using GroupKFold)")

    # Generate fold assignments
    y = (
        train_df[target_col].to_numpy()
        if len(resolved_target_cols) == 1
        else train_df.loc[:, resolved_target_cols].to_numpy()
    )

    # Detect if target is string type
    target_is_string = y.dtype == object or np.issubdtype(y.dtype, np.str_)

    # Determine classification vs regression/seq2seq. The task contract takes
    # precedence over target cardinality so low-cardinality regression and
    # high-cardinality classification are both handled correctly.
    task_declares_classification = "classification" in normalized_task_type
    task_declares_regression = (
        "regression" in normalized_task_type
        or "forecast" in normalized_task_type
    )
    task_type_source = "explicit_task_contract"
    if is_seq2seq:
        # Seq2seq tasks (text normalization, translation, summarization) are NOT classification
        # They have high-cardinality string targets that shouldn't be stratified
        is_classification = False
        n_unique = None  # Don't count unique strings (could be millions)
        print("   Target type: seq2seq (string, non-classification)")
    elif resolved_target_type == "multi_label":
        n_unique = None
        is_classification = True
        print(
            "   Target type: multilabel classification "
            f"({len(resolved_target_cols)} independent labels)"
        )
    elif resolved_target_type == "multi_target":
        n_unique = None
        is_classification = False
        print(
            "   Target type: multi-target regression "
            f"({len(resolved_target_cols)} outputs)"
        )
    elif task_declares_classification:
        n_unique = len(np.unique(y))
        is_classification = True
        print(f"   Target type: declared classification ({n_unique} classes)")
    elif task_declares_regression:
        n_unique = len(np.unique(y))
        is_classification = False
        print("   Target type: declared regression")
    else:
        # Backward-compatible inference for callers that supply only "tabular".
        # It uses target structure, never target names or competition identity.
        n_unique = len(np.unique(y))
        task_type_source = "observed_target_structure"
        if target_is_string:
            is_classification = True
            print(f"   Target type: string classification ({n_unique} classes)")
        else:
            numeric_y = np.asarray(y, dtype=float)
            finite_y = numeric_y[np.isfinite(numeric_y)]
            integer_like = bool(
                finite_y.size
                and np.allclose(finite_y, np.round(finite_y))
            )
            class_counts = pd.Series(y).value_counts(dropna=False)
            repeated_classes = bool(
                n_unique >= 2
                and n_unique <= max(2, int(np.sqrt(len(y))))
                and not class_counts.empty
                and int(class_counts.min()) >= 2
            )
            is_classification = integer_like and repeated_classes
            inferred_kind = "classification" if is_classification else "regression"
            print(
                "   Target type inferred from observed dtype/cardinality: "
                f"{inferred_kind}"
            )

    temporal_metadata: dict[str, Any] | None = None
    temporal_splits_path: Path | None = None
    oof_eligible_mask_path: Path | None = None
    temporal_order_path: Path | None = None

    if is_temporal:
        (
            resolved_temporal_col,
            temporal_order,
            temporal_value_type,
            temporal_evidence_source,
        ) = _resolve_temporal_order(
            train_df,
            temporal_col=temporal_col,
            column_contract=column_contract,
        )
        if resolved_temporal_col in resolved_target_cols:
            raise ValueError(
                "Target column cannot serve as the temporal ordering contract"
            )
        (
            fold_assignments,
            oof_eligible_mask,
            temporal_splits,
            temporal_cutoffs,
        ) = _build_forward_chaining_splits(
            temporal_order,
            n_folds=n_folds,
            value_type=temporal_value_type,
        )

        temporal_splits_path = canonical_dir / "temporal_splits.npz"
        oof_eligible_mask_path = canonical_dir / "oof_eligible_mask.npy"
        temporal_order_path = canonical_dir / "temporal_order.npy"
        split_arrays: dict[str, np.ndarray] = {}
        for fold_idx, (train_idx, val_idx) in enumerate(temporal_splits):
            split_arrays[f"train_{fold_idx}"] = train_idx
            split_arrays[f"validation_{fold_idx}"] = val_idx
        np.savez_compressed(temporal_splits_path, **split_arrays)
        np.save(oof_eligible_mask_path, oof_eligible_mask)
        np.save(temporal_order_path, temporal_order)

        temporal_metadata = {
            "strategy": "expanding_window_forward_chaining",
            "temporal_col": resolved_temporal_col,
            "evidence_source": temporal_evidence_source,
            "value_type": temporal_value_type,
            "strict_train_before_validation": True,
            "n_unique_order_values": int(np.unique(temporal_order).size),
            "warmup_rows": int((~oof_eligible_mask).sum()),
            "oof_eligible_rows": int(oof_eligible_mask.sum()),
            "oof_coverage_fraction": float(oof_eligible_mask.mean()),
            "splits_path": str(temporal_splits_path),
            "oof_eligible_mask_path": str(oof_eligible_mask_path),
            "order_values_path": str(temporal_order_path),
            "fold_cutoffs": temporal_cutoffs,
        }
        print(
            f"   Temporal order: {resolved_temporal_col} "
            f"({temporal_evidence_source}); "
            f"OOF coverage={oof_eligible_mask.mean():.1%}"
        )
    else:
        fold_assignments = np.zeros(len(train_df), dtype=int)

    if is_temporal:
        # Splits were fully materialized and validated above. Do not derive
        # training rows as ``folds != fold`` for this strategy.
        pass
    elif group_col:
        groups = train_df[group_col].values
        # For seq2seq tasks or when stratification is not possible, use GroupKFold
        if is_classification and n_unique is not None and n_unique <= 10:
            try:
                kf = StratifiedGroupKFold(
                    n_splits=n_folds, shuffle=True, random_state=run_seed
                )
                for fold, (_, val_idx) in enumerate(kf.split(train_df, y, groups)):
                    fold_assignments[val_idx] = fold
                cv_config["strategy"] = "stratified_group_kfold"
            except Exception:
                kf = GroupKFold(n_splits=n_folds)
                for fold, (_, val_idx) in enumerate(kf.split(train_df, groups=groups)):
                    fold_assignments[val_idx] = fold
                cv_config["strategy"] = "group_kfold"
        else:
            # Seq2seq tasks or high-cardinality targets use GroupKFold
            kf = GroupKFold(n_splits=n_folds)
            for fold, (_, val_idx) in enumerate(kf.split(train_df, groups=groups)):
                fold_assignments[val_idx] = fold
            cv_config["strategy"] = "group_kfold"
    elif resolved_target_type == "multi_label":
        fold_assignments = _build_multilabel_fold_assignments(
            y,
            n_folds=n_folds,
            seed=run_seed,
        )
        cv_config["strategy"] = "multilabel_stratified_kfold"
    elif is_classification:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=run_seed)
        for fold, (_, val_idx) in enumerate(kf.split(train_df, y)):
            fold_assignments[val_idx] = fold
        cv_config["strategy"] = "stratified_kfold"
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=run_seed)
        for fold, (_, val_idx) in enumerate(kf.split(train_df)):
            fold_assignments[val_idx] = fold
        cv_config["strategy"] = "kfold"

    print(f"   CV strategy: {n_folds} folds ({cv_config['strategy']})")

    # Extract canonical data. String IDs must be stored as str dtype, not
    # object: candidate code re-saves these IDs with allow_pickle=False and
    # the trusted scorer refuses to load pickled artifacts.
    train_ids = train_df[id_col].values
    if train_ids.dtype == object:
        train_ids = np.asarray([str(v) for v in train_ids])

    # Save canonical artifacts
    np.save(canonical_dir / "train_ids.npy", train_ids, allow_pickle=False)

    # Every component must name its test rows the same way, or the ensemble
    # cannot align them. Competitions whose public test table carries no unique
    # key leave each component to invent one, and the ones they reach for
    # (a repeated date, a placeholder target) are not unique - the artifacts
    # are then rejected however good the model was.
    test_ids, test_ids_are_positional = _resolve_canonical_test_ids(
        test_path, id_col
    )
    if test_ids is not None:
        np.save(canonical_dir / "test_ids.npy", test_ids, allow_pickle=False)

    # Save targets - use allow_pickle=True for string/object arrays (seq2seq tasks)
    if target_is_string or y.dtype == object:
        np.save(canonical_dir / "y.npy", y, allow_pickle=True)
        print("   Saved string targets (dtype=object) with allow_pickle=True")
    else:
        np.save(canonical_dir / "y.npy", y)

    np.save(canonical_dir / "folds.npy", fold_assignments)

    with open(canonical_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save metadata
    metadata = {
        "original_rows": original_rows,
        "canonical_rows": len(train_df),
        "n_folds": n_folds,
        "random_seed": run_seed,
        "cv_strategy": cv_config["strategy"],
        "id_col": id_col,
        "id_is_synthetic": is_synthetic_id,
        "test_ids_are_positional": bool(test_ids_are_positional),
        "n_test": int(len(test_ids)) if test_ids is not None else None,
        "target_col": target_col,
        "target_cols": resolved_target_cols,
        "target_type": resolved_target_type,
        "target_type_source": target_type_source,
        "n_targets": len(resolved_target_cols),
        "n_features": len(feature_cols),
        "group_col": group_col,
        "is_classification": is_classification,
        "n_classes": n_unique if is_classification else None,
        "canonical_version": "1.5",
        # Seq2seq specific metadata
        "task_type": task_type,
        "task_type_source": task_type_source,
        "is_temporal": is_temporal,
        "temporal_cv": temporal_metadata,
        "is_seq2seq": is_seq2seq,
        "source_col": source_col,
        "class_col": class_col,
        "seq2seq_group_col": seq2seq_group_col,
        "target_dtype": str(y.dtype),
        "target_dtypes": [
            str(train_df[column].dtype)
            for column in resolved_target_cols
        ],
        **sampling_metadata,
    }

    with open(canonical_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   Saved canonical artifacts to {canonical_dir}")

    result = {
        "canonical_dir": str(canonical_dir),
        "train_ids_path": str(canonical_dir / "train_ids.npy"),
        "y_path": str(canonical_dir / "y.npy"),
        "folds_path": str(canonical_dir / "folds.npy"),
        "feature_cols_path": str(canonical_dir / "feature_cols.json"),
        "metadata_path": str(canonical_dir / "metadata.json"),
        "metadata": metadata,
    }
    if temporal_splits_path is not None:
        result["temporal_splits_path"] = str(temporal_splits_path)
    if oof_eligible_mask_path is not None:
        result["oof_eligible_mask_path"] = str(oof_eligible_mask_path)
    if temporal_order_path is not None:
        result["temporal_order_path"] = str(temporal_order_path)
    return result


def load_canonical_data(working_dir: str | Path) -> dict[str, Any]:
    """
    Load all canonical data artifacts.

    Args:
        working_dir: Competition working directory

    Returns:
        Dict with all canonical data loaded:
        - train_ids: np.ndarray of row IDs
        - y: np.ndarray of target values
        - folds: np.ndarray of fold assignments
        - feature_cols: list of feature column names
        - metadata: dict with sampling/CV info
    """
    canonical_dir = Path(working_dir) / "canonical"

    if not canonical_dir.exists():
        raise FileNotFoundError(
            f"Canonical data not found at {canonical_dir}. "
            "Run prepare_canonical_data() first."
        )

    train_ids = np.load(canonical_dir / "train_ids.npy", allow_pickle=True)
    y = np.load(canonical_dir / "y.npy", allow_pickle=True)
    folds = np.load(canonical_dir / "folds.npy")

    with open(canonical_dir / "feature_cols.json") as f:
        feature_cols = json.load(f)

    with open(canonical_dir / "metadata.json") as f:
        metadata = json.load(f)

    result = {
        "train_ids": train_ids,
        "y": y,
        "folds": folds,
        "feature_cols": feature_cols,
        "metadata": metadata,
        "canonical_dir": str(canonical_dir),
    }
    temporal_cv = metadata.get("temporal_cv")
    if isinstance(temporal_cv, dict):
        required_paths = {
            "temporal_splits": temporal_cv.get("splits_path"),
            "oof_eligible_mask": temporal_cv.get("oof_eligible_mask_path"),
            "temporal_order": temporal_cv.get("order_values_path"),
        }
        for key, raw_path in required_paths.items():
            if not raw_path or not Path(raw_path).is_file():
                raise FileNotFoundError(
                    f"Temporal canonical contract is missing {key}: {raw_path!r}"
                )
        with np.load(
            required_paths["temporal_splits"], allow_pickle=False
        ) as split_archive:
            temporal_splits = {
                key: np.asarray(split_archive[key], dtype=np.int64)
                for key in split_archive.files
            }
        result["temporal_splits"] = temporal_splits
        result["oof_eligible_mask"] = np.asarray(
            np.load(required_paths["oof_eligible_mask"], allow_pickle=False),
            dtype=bool,
        )
        result["temporal_order"] = np.load(
            required_paths["temporal_order"], allow_pickle=False
        )
        mask = result["oof_eligible_mask"]
        order = result["temporal_order"]
        if mask.shape != (len(train_ids),) or order.shape != (len(train_ids),):
            raise ValueError(
                "Temporal mask/order arrays are not aligned with canonical rows"
            )
        if not np.array_equal(folds >= 0, mask):
            raise ValueError(
                "Temporal folds and OOF eligibility mask are inconsistent"
            )

        validation_counts = np.zeros(len(train_ids), dtype=np.int32)
        for fold_idx in range(int(metadata["n_folds"])):
            train_key = f"train_{fold_idx}"
            validation_key = f"validation_{fold_idx}"
            if (
                train_key not in temporal_splits
                or validation_key not in temporal_splits
            ):
                raise ValueError(
                    f"Temporal split artifact is missing fold {fold_idx}"
                )
            train_idx = temporal_splits[train_key]
            val_idx = temporal_splits[validation_key]
            if (
                len(train_idx) == 0
                or len(val_idx) == 0
                or np.intersect1d(train_idx, val_idx).size
            ):
                raise ValueError(
                    f"Temporal fold {fold_idx} has an invalid partition"
                )
            if (
                np.any(train_idx < 0)
                or np.any(train_idx >= len(train_ids))
                or np.any(val_idx < 0)
                or np.any(val_idx >= len(train_ids))
            ):
                raise ValueError(
                    f"Temporal fold {fold_idx} contains out-of-range indices"
                )
            if not order[train_idx].max() < order[val_idx].min():
                raise ValueError(
                    f"Temporal fold {fold_idx} violates train-before-validation"
                )
            if not np.all(folds[val_idx] == fold_idx):
                raise ValueError(
                    f"Temporal fold {fold_idx} validation assignments disagree"
                )
            validation_counts[val_idx] += 1
        if not np.all(validation_counts[mask] == 1):
            raise ValueError(
                "Temporal eligible rows must appear in exactly one validation fold"
            )
        if np.any(validation_counts[~mask]):
            raise ValueError("Temporal warm-up rows entered a validation fold")
    else:
        result["oof_eligible_mask"] = np.ones(len(train_ids), dtype=bool)
    return result


def validate_oof_alignment(
    oof: np.ndarray,
    working_dir: str | Path,
    model_train_ids: np.ndarray | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate that OOF predictions align with canonical data.

    Args:
        oof: OOF predictions array
        working_dir: Competition working directory
        model_train_ids: Train IDs used by the model (optional)

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    canonical = load_canonical_data(working_dir)
    canonical_ids = canonical["train_ids"]
    n_canonical = len(canonical_ids)

    # Check shape
    if oof.shape[0] != n_canonical:
        issues.append(
            f"OOF shape mismatch: {oof.shape[0]} rows vs {n_canonical} canonical rows"
        )

    # Check ID alignment if provided
    if model_train_ids is not None:
        if not np.array_equal(model_train_ids, canonical_ids):
            # Check overlap
            common = np.intersect1d(model_train_ids, canonical_ids)
            overlap_pct = len(common) / n_canonical * 100
            issues.append(
                f"Train ID mismatch: {overlap_pct:.1f}% overlap with canonical IDs"
            )

    eligible_mask = np.asarray(canonical["oof_eligible_mask"], dtype=bool)
    if eligible_mask.shape != (n_canonical,):
        issues.append(
            "Canonical OOF eligibility mask shape mismatch: "
            f"{eligible_mask.shape} vs {(n_canonical,)}"
        )
        return False, issues

    if oof.shape[0] == n_canonical:
        eligible_oof = oof[eligible_mask]
        warmup_oof = oof[~eligible_mask]

        # Only validation rows may participate in metrics/ensembles.
        if not np.isfinite(eligible_oof).all():
            n_invalid = int((~np.isfinite(eligible_oof)).sum())
            issues.append(
                f"OOF contains {n_invalid} NaN/Inf values on eligible rows"
            )

        # Warm-up history has no honest out-of-fold prediction. Keeping it NaN
        # prevents accidental scoring as in-sample data or fabricated coverage.
        if warmup_oof.size and not np.isnan(warmup_oof).all():
            issues.append(
                "Temporal warm-up OOF rows must remain NaN and excluded by the "
                "canonical eligibility mask"
            )
    else:
        eligible_oof = np.asarray([])

    # Check for empty rows
    if eligible_oof.ndim > 1:
        empty_mask = eligible_oof.sum(axis=1) == 0
    else:
        empty_mask = np.abs(eligible_oof) < 1e-10
    n_empty = empty_mask.sum()
    if n_empty > 0:
        issues.append(f"OOF has {n_empty} empty/zero rows")

    return len(issues) == 0, issues


def align_oof_by_id(
    oof: np.ndarray,
    model_ids: np.ndarray,
    canonical_ids: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Align OOF predictions to canonical ID order.

    Useful when model was trained on a subset or different order.

    Args:
        oof: OOF predictions from model
        model_ids: IDs corresponding to oof rows
        canonical_ids: Target canonical ID order
        fill_value: Value to use for missing predictions

    Returns:
        OOF aligned to canonical ID order
    """
    # Create ID to index mapping for model predictions
    model_id_to_idx = {id_val: idx for idx, id_val in enumerate(model_ids)}

    # Initialize aligned OOF
    if oof.ndim > 1:
        aligned_oof = np.full((len(canonical_ids), oof.shape[1]), fill_value)
    else:
        aligned_oof = np.full(len(canonical_ids), fill_value)

    # Map model predictions to canonical order
    for canonical_idx, canonical_id in enumerate(canonical_ids):
        if canonical_id in model_id_to_idx:
            model_idx = model_id_to_idx[canonical_id]
            aligned_oof[canonical_idx] = oof[model_idx]

    return aligned_oof


# ==================== Code Validation ====================


def validate_canonical_data_usage(
    generated_code: str,
    working_dir: str | Path,
    component_type: str = "model",
) -> tuple[bool, str, list[str]]:
    """
    Validate that generated code uses canonical data correctly.

    Checks for:
    1. Use of canonical data loading (load_canonical_data or npy files)
    2. Proper fold usage for CV
    3. No independent sampling or fold creation

    Args:
        generated_code: The code to validate
        working_dir: Working directory path
        component_type: Type of component (model, feature_engineering, etc.)

    Returns:
        Tuple of (is_valid, error_message, warnings)
    """
    import re

    warnings = []
    code_lower = generated_code.lower()

    # Check if canonical directory exists
    canonical_dir = Path(working_dir) / "canonical"
    canonical_exists = canonical_dir.exists()

    if not canonical_exists:
        # Canonical data not yet prepared - this is OK for early components
        return True, "", ["Canonical data not yet prepared - will be created"]

    # Patterns indicating proper canonical data usage
    canonical_patterns = [
        r"load_canonical_data\s*\(",
        r"canonical/train_ids\.npy",
        r"canonical/folds\.npy",
        r"canonical/y\.npy",
        r"np\.load\s*\([^)]*canonical[^)]*\)",
    ]

    # Check if any canonical pattern is present
    uses_canonical = any(re.search(p, generated_code) for p in canonical_patterns)

    # Anti-patterns: things that suggest independent data handling
    # NOTE: shuffle=True is intentionally NOT blocked here because DataLoader(shuffle=True)
    # is standard practice for batch randomization and does NOT affect canonical fold alignment.
    # Actual fold shuffling violations are caught by KFold/StratifiedKFold patterns above.
    anti_patterns = [
        (r"train_test_split\s*\(", "Using train_test_split - should use canonical folds"),
        (r"StratifiedKFold\s*\(", "Creating new folds - should use canonical folds"),
        (r"KFold\s*\(", "Creating new folds - should use canonical folds"),
        (r"GroupKFold\s*\(", "Creating new folds - should use canonical folds"),
        (r"\.sample\s*\(", "Sampling data - may cause alignment issues with canonical"),
    ]

    # Check if canonical folds are being loaded (primary path)
    # If so, fallback KFold/StratifiedKFold is acceptable
    loads_canonical_folds = bool(
        re.search(r"np\.load.*CANONICAL_FOLDS_PATH|CANONICAL_FOLDS\s*=\s*np\.load", generated_code)
        or re.search(r"np\.load.*canonical.*folds\.npy", generated_code, re.IGNORECASE)
    )

    violations = []
    for pattern, message in anti_patterns:
        if re.search(pattern, generated_code):
            # Exception: If it's used to create canonical data, that's OK
            if "prepare_canonical_data" in generated_code:
                continue
            # Exception: train_test_split for Optuna subsampling is OK
            # (subsampling for speed, not creating CV splits)
            # Note: uses_canonical check removed - subsampling is always OK for Optuna
            # because canonical enforcement already happens at lines 756-782
            if "train_test_split" in pattern:
                is_optuna_subsample = (
                    "optuna" in generated_code.lower()
                    and re.search(r"train_size\s*=\s*0\.[0-4]", generated_code)
                )
                if is_optuna_subsample:
                    continue
            # Exception: KFold/StratifiedKFold in fallback is OK if canonical folds ARE loaded
            # This allows code like: if canonical_exists: load(folds.npy) else: StratifiedKFold()
            if "KFold" in pattern and loads_canonical_folds:
                # Primary path uses canonical, fallback KFold is acceptable
                continue
            violations.append(message)

    # Model components MUST use canonical data (STRICT ENFORCEMENT)
    if component_type == "model":
        # Check for required canonical patterns (MUST have these)
        required_patterns = [
            (r"canonical.*folds\.npy|folds\.npy.*canonical", "Must load canonical/folds.npy"),
            (r"canonical.*train_ids\.npy|train_ids\.npy.*canonical", "Must load canonical/train_ids.npy"),
        ]

        missing_required = []
        for pattern, message in required_patterns:
            if not re.search(pattern, generated_code, re.IGNORECASE):
                missing_required.append(message)

        # Fail if violations exist (creating independent folds)
        if violations:
            error_msg = (
                "Model code violates canonical data contract. "
                f"Violations: {'; '.join(violations)}. "
                "MUST use canonical folds from canonical/folds.npy - do NOT create KFold/StratifiedKFold."
            )
            return False, error_msg, warnings

        # Fail if not using canonical data patterns
        if missing_required and not uses_canonical:
            error_msg = (
                "Model code does not use canonical data contract. "
                f"Missing: {'; '.join(missing_required)}. "
                "Use load_canonical_data() or load canonical/*.npy files for consistent OOF alignment."
            )
            return False, error_msg, warnings

        # Warn if missing required but has some canonical usage
        if missing_required:
            warnings.extend(missing_required)

    # Feature engineering components should be more flexible
    elif component_type == "feature_engineering":
        if violations:
            warnings.append(
                "Feature engineering code may modify data alignment. "
                "Ensure train_ids are preserved."
            )

    # Ensemble MUST use aligned predictions
    elif component_type == "ensemble":
        if not uses_canonical and not re.search(r"oof_.*\.npy|test_.*\.npy", generated_code):
            warnings.append(
                "Ensemble should verify OOF alignment with canonical train_ids"
            )

    return True, "", warnings


def get_canonical_data_instructions(working_dir: str | Path) -> str:
    """
    Generate instructions for using canonical data in generated code.

    Args:
        working_dir: Working directory path

    Returns:
        Instruction string to inject into developer prompt
    """
    canonical_dir = Path(working_dir) / "canonical"

    if canonical_dir.exists():
        # Load metadata for context
        try:
            with open(canonical_dir / "metadata.json") as f:
                metadata = json.load(f)
            n_rows = metadata.get("canonical_rows", "unknown")
            n_folds = metadata.get("n_folds", 5)
            id_col = metadata.get("id_col", "id")
        except Exception:
            n_rows = "unknown"
            n_folds = 5
            id_col = "id"

        return f'''
## MANDATORY: Canonical Data Contract

The canonical data has been prepared with {n_rows} rows and {n_folds} folds.
You MUST use the canonical data to ensure consistency across all models.

### How to Load Canonical Data:

```python
import numpy as np
import json
from pathlib import Path

# Load canonical data
canonical_dir = Path("{working_dir}/canonical")
train_ids = np.load(canonical_dir / "train_ids.npy", allow_pickle=True)
y = np.load(canonical_dir / "y.npy", allow_pickle=True)
folds = np.load(canonical_dir / "folds.npy")

with open(canonical_dir / "feature_cols.json") as f:
    feature_cols = json.load(f)

# Use folds for CV (DO NOT create your own folds!)
n_folds = {n_folds}
for fold_idx in range(n_folds):
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    # Train model...
    model.fit(X_train, y_train)

    # Store OOF predictions in order
    oof[val_mask] = model.predict_proba(X_val)
```

### CRITICAL RULES:
1. NEVER use train_test_split() - use canonical folds
2. NEVER create your own KFold/StratifiedKFold - folds are pre-defined
3. NEVER sample or shuffle the data independently
4. ALWAYS save OOF predictions in canonical order (aligned with train_ids)
5. ID column is: "{id_col}"

### Saving Predictions:
```python
# Save OOF aligned with canonical train_ids
np.save("models/oof_{{model_name}}.npy", oof)

# Verify alignment before saving
assert len(oof) == len(train_ids), "OOF must match canonical row count"
```
'''
    return '''
## Note: Canonical Data Will Be Prepared

The canonical data contract will be prepared before your component runs.
When it's ready, use load_canonical_data() to get train_ids, folds, and y.
Do NOT create your own folds or sampling strategy.
'''
