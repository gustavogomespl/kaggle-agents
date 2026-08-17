"""Submission validation and alignment functions."""

import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from ...utils.csv_utils import read_csv_auto
from ...utils.submission_artifacts import sha256_file
from .postprocessing import labels_from_oof_tuning


def prediction_positions(
    sample_sub: pd.DataFrame,
    target_cols: list[str] | None = None,
) -> list[int]:
    """Return the template positions the model is expected to fill.

    Templates that echo test input back to the grader, or that place the
    prediction before its context columns, break the ``columns[1:]``
    convention: predictions would be written over input columns while the
    graded column keeps its placeholder value. The resolved submission
    contract is authoritative whenever it names columns actually present.

    Args:
        sample_sub: The submission template
        target_cols: Resolved prediction column names, when known

    Returns:
        Ordered positional indices into ``sample_sub.columns``
    """
    columns = [str(column) for column in sample_sub.columns]
    positions = [
        columns.index(str(column))
        for column in (target_cols or [])
        if str(column) in columns
    ]
    if positions:
        return positions
    return list(range(1, len(columns)))


def format_ensemble_predictions(
    preds: np.ndarray,
    sample_sub: pd.DataFrame,
    problem_type: str,
    metric_name: str | None = None,
    oof_preds: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    target_cols: list[str] | None = None,
) -> np.ndarray:
    """Format predictions for submission based on metric and problem type.
    Converts probabilities to class labels when the metric or sample sub expects integers.

    When OOF predictions + training labels are provided, the decision rule is
    tuned on OOF (threshold search / QWK rounding boundaries) instead of the
    fixed 0.5 / argmax defaults.
    """
    if not problem_type or "class" not in problem_type.lower():
        return preds

    metric_lower = (metric_name or "").lower()
    prob_metrics = ("logloss", "log_loss", "log loss", "cross_entropy", "brier", "auc", "roc", "prc", "average_precision")
    label_metrics = ("accuracy", "f1", "precision", "recall", "kappa", "qwk", "quadratic_weighted_kappa", "mcc")
    expects_prob = any(m in metric_lower for m in prob_metrics)
    expects_label = any(m in metric_lower for m in label_metrics)

    sample_suggests_label = False
    if sample_sub.shape[1] >= 2:
        sample_vals = sample_sub.iloc[
            :, prediction_positions(sample_sub, target_cols)[0]
        ]
        if pd.api.types.is_numeric_dtype(sample_vals):
            svals = sample_vals.to_numpy()
            if svals.size and np.allclose(svals, np.round(svals)):
                sample_suggests_label = True

    if expects_label or (sample_suggests_label and not expects_prob):
        preds_array = np.asarray(preds)
        if preds_array.ndim == 1 or preds_array.shape[1] == 1:
            flat = preds_array.ravel()

            # Metric-aware decision rule tuned on OOF (falls back to 0.5)
            if oof_preds is not None and y_true is not None:
                try:
                    oof_flat = np.asarray(oof_preds, dtype=float).ravel()
                    if len(oof_flat) == len(np.asarray(y_true).ravel()):
                        labels, info = labels_from_oof_tuning(
                            flat, oof_flat, y_true, metric_name or "accuracy"
                        )
                        rule_params = {
                            k: v for k, v in info.items() if k in ("threshold", "boundaries")
                        }
                        print(
                            f"   [POSTPROC] {info['rule']}: OOF "
                            f"{info['oof_score_baseline']:.4f} -> {info['oof_score_tuned']:.4f} "
                            f"({rule_params})"
                        )
                        return labels if preds_array.ndim == 1 else labels.reshape(-1, 1)
                except Exception as e:
                    print(f"   [POSTPROC] OOF tuning failed, using 0.5 threshold: {e}")

            binary = (flat >= 0.5).astype(int)
            if y_true is not None:
                classes = np.unique(np.asarray(y_true).ravel())
                if len(classes) == 2:
                    binary = classes[binary]
            return binary if preds_array.ndim == 1 else binary.reshape(-1, 1)
        return np.argmax(preds_array, axis=1)

    return preds


def validate_and_align_submission(
    submission_path: Path,
    sample_submission_path: Path,
    output_path: Path | None = None,
    target_cols: list[str] | None = None,
) -> tuple[bool, str, Path | None]:
    """Validate submission against sample_submission schema.

    If IDs are same set but different order, reorders to match sample.

    Args:
        submission_path: Path to submission to validate
        sample_submission_path: Path to sample_submission.csv
        output_path: Where to save aligned submission (if None, overwrites in place)
        target_cols: Resolved prediction column names. Without them the first
            column is assumed to identify rows, which rejects valid work on
            templates whose first column is the prediction.

    Returns:
        Tuple of (is_valid, error_message, aligned_path)
    """
    output_path = output_path or submission_path

    try:
        sub_df = pd.read_csv(submission_path)
        sample_df = read_csv_auto(sample_submission_path)
    except Exception as e:
        return False, f"Failed to read CSV: {e}", None

    # Check columns match
    if list(sub_df.columns) != list(sample_df.columns):
        return False, f"Column mismatch: {sub_df.columns.tolist()} vs {sample_df.columns.tolist()}", None

    # Check row count
    if len(sub_df) != len(sample_df):
        return False, f"Row count mismatch: {len(sub_df)} vs {len(sample_df)}", None

    pred_positions = prediction_positions(sample_df, target_cols)
    pred_cols = [sample_df.columns[position] for position in pred_positions]
    echo_cols = [
        column for column in sample_df.columns if column not in set(pred_cols)
    ]

    if not echo_cols:
        # Nothing identifies a row; order is the only alignment available and
        # the row count was already checked.
        return _finalize_aligned_submission(sub_df, pred_cols, output_path)

    # Check ID column - same SET but possibly different order
    id_col = echo_cols[0]
    sub_ids = set(sub_df[id_col])
    sample_ids = set(sample_df[id_col])

    if sub_ids != sample_ids:
        missing = sample_ids - sub_ids
        extra = sub_ids - sample_ids
        return False, f"ID mismatch: missing={len(missing)}, extra={len(extra)}", None

    # If order differs, reorder to match sample. A non-unique identifier cannot
    # drive a merge without inventing rows, so only order-preserving templates
    # take this path.
    if not sub_df[id_col].equals(sample_df[id_col]):
        if sample_df[id_col].duplicated().any():
            return (
                False,
                f"Submission row order differs and '{id_col}' is not unique; "
                "cannot realign without inventing rows",
                None,
            )
        print("      [LOG:INFO] Reordering submission to match sample_submission ID order")
        # Reorder using merge
        sub_df = sample_df[[id_col]].merge(sub_df, on=id_col, how='left')

    return _finalize_aligned_submission(sub_df, pred_cols, output_path)


def _finalize_aligned_submission(
    sub_df: pd.DataFrame,
    pred_cols: list[str],
    output_path: Path,
) -> tuple[bool, str, Path | None]:
    """Reject unusable prediction values, then persist the aligned file."""
    nan_count = sub_df[pred_cols].isna().sum().sum()
    if nan_count > 0:
        return False, f"Submission contains {nan_count} NaN values", None
    numeric_predictions = sub_df[pred_cols].select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric_predictions.to_numpy(dtype=float)).sum())
    if inf_count > 0:
        return False, f"Submission contains {inf_count} infinite values", None

    # Save aligned submission
    sub_df.to_csv(output_path, index=False)
    return True, "", output_path


def safe_restore_submission(
    source_path: Path,
    dest_path: Path,
    sample_submission_path: Path | None,
    *,
    target_cols: list[str] | None = None,
    problem_type: str | None = None,
    expected_sha256: str | None = None,
    require_hash: bool = False,
) -> bool:
    """Atomically restore a submission only after all available checks pass.

    Args:
        source_path: Path to the preserved source submission.
        dest_path: Path to destination (e.g., submission.csv)
        sample_submission_path: Path to sample_submission.csv for validation
        target_cols: Resolved prediction column names, so row order is checked
            against a column the template supplies rather than one the model
            fills in.
        problem_type: Explicit prediction semantics, including label-format
            multiclass submissions whose template placeholder is numeric.
        expected_sha256: Optional immutable-snapshot digest.
        require_hash: Fail closed unless ``expected_sha256`` is supplied and
            matches both the source and restored destination.

    Returns:
        True if restoration succeeded, False otherwise
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    sample_path = (
        Path(sample_submission_path) if sample_submission_path is not None else None
    )
    expected_digest = str(expected_sha256 or "").strip().lower()

    if not source_path.is_file() or source_path.is_symlink():
        print(f"      Warning: Source submission not found: {source_path}")
        return False

    if require_hash and len(expected_digest) != 64:
        print("      Warning: Immutable submission digest is missing or malformed")
        return False
    if expected_digest:
        try:
            if sha256_file(source_path) != expected_digest:
                print("      Warning: Source submission failed hash verification")
                return False
        except OSError as exc:
            print(f"      Warning: Could not hash source submission: {exc}")
            return False

    # A restore operation without the public schema cannot establish that the
    # artifact is a valid submission. Callers that truly want an unchecked copy
    # should use an ordinary filesystem copy instead of this safety boundary.
    if sample_path is None or not sample_path.is_file():
        print("      Warning: sample_submission contract unavailable; restore blocked")
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path = dest_path.parent / (
        f".{dest_path.name}.validation-{uuid.uuid4().hex}.tmp"
    )
    restore_path = dest_path.parent / (
        f".{dest_path.name}.restore-{uuid.uuid4().hex}.tmp"
    )

    try:
        if expected_digest:
            # Immutable snapshots must already have canonical row order. An
            # alignment rewrite would sever the state digest from the bytes
            # subsequently graded. Validate exact schema, echo columns, order,
            # row count, and finite predictions in bounded chunks.
            from kaggle_agents.tools.code_executor.submission import (
                SubmissionValidationMixin,
            )

            # Structure, echo columns, order, and row count are re-checked;
            # the metric-dependent row-sum quality rule is NOT. The snapshot
            # was validated under the graded metric's rule when it was
            # accepted and the digest pins those exact bytes — re-litigating
            # quality here rejected, at restore time, submissions a ranking
            # metric would score unchanged.
            is_valid, error_msg = (
                SubmissionValidationMixin().validate_submission_format(
                    source_path,
                    sample_path,
                    component_type="model",
                    problem_type=problem_type,
                    target_cols=target_cols,
                    require_normalized_rows=False,
                )
            )
            if not is_valid:
                print(
                    "      Warning: Submission validation failed: "
                    f"{error_msg}"
                )
                return False
            shutil.copyfile(source_path, restore_path)
            if sha256_file(restore_path) != expected_digest:
                print("      Warning: Restored submission failed hash verification")
                return False
        else:
            # For regular Kaggle runs retain the safe historical behavior:
            # reorder a structurally valid artifact to the template ID order.
            is_valid, error_msg, _ = validate_and_align_submission(
                source_path,
                sample_path,
                validation_path,
                target_cols,
            )
            if not is_valid:
                print(
                    "      Warning: Submission validation failed: "
                    f"{error_msg}"
                )
                return False
            validation_path.replace(restore_path)

        restore_path.replace(dest_path)
        if expected_digest and sha256_file(dest_path) != expected_digest:
            # Do not report success if the final bytes differ from the snapshot.
            dest_path.unlink(missing_ok=True)
            print("      Warning: Destination submission failed hash verification")
            return False

        print(f"      OK: Validated and restored submission to {dest_path}")
        return True
    except Exception as exc:
        print(f"      Warning: Submission restore failed: {exc}")
        return False
    finally:
        validation_path.unlink(missing_ok=True)
        restore_path.unlink(missing_ok=True)
