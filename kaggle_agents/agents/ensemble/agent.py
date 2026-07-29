"""Ensemble agent for model stacking and blending."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import cross_val_predict

from ...core.config import (
    calculate_score_improvement,
    get_config,
    is_metric_minimization,
)
from ...core.state import KaggleState
from ...utils.csv_utils import read_csv_auto
from ...utils.llm_utils import get_text_content
from ...utils.submission_artifacts import (
    sha256_file,
    verified_accepted_submission,
    verified_best_candidate_submission,
)
from ...utils.submission_format import infer_submission_logic
from ...utils.telemetry import make_event
from .alignment import align_oof_by_canonical_ids, load_and_align_oof, validate_oof_alignment
from .fallback import (
    create_ensemble_with_fallback,
    fallback_to_best_single_model,
    recover_from_checkpoints,
)
from .meta_model import (
    constrained_meta_learner,
    diagnose_stacking_issues,
    dirichlet_weight_search,
    tune_meta_model,
)
from .postprocessing import labels_from_oof_tuning, metric_label_kind
from .prediction_pairs import find_prediction_pairs, validate_prediction_artifacts_contract
from .scoring import compute_oof_score, filter_by_score_threshold, score_predictions
from .stacking import load_cv_folds, stack_from_prediction_pairs
from .submission import (
    format_ensemble_predictions,
    prediction_positions,
    safe_restore_submission,
    validate_and_align_submission,
)
from .utils import class_orders_match, encode_labels


def _truncate_pred_cols(preds: np.ndarray, expected_cols: int) -> np.ndarray:
    """Truncate prediction columns to match submission format.

    For binary classification where models produce 2 columns [P(class=0), P(class=1)]
    but submission expects 1 column, selects column 1 (positive class).
    This follows the scikit-learn convention and matches stacking.py behavior.
    """
    if preds.ndim != 2 or preds.shape[1] <= expected_cols:
        return preds
    if preds.shape[1] == 2 and expected_cols == 1:
        # Binary classification: select positive class (column 1)
        return preds[:, 1:]
    # General case: take last expected_cols columns
    return preds[:, -expected_cols:]


def _load_temporal_oof_mask(
    models_dir: Path,
    n_rows: int,
) -> np.ndarray | None:
    """Load and validate the canonical temporal OOF eligibility mask."""
    mask_path = models_dir.parent / "canonical" / "oof_eligible_mask.npy"
    if not mask_path.is_file():
        return None
    mask = np.asarray(np.load(mask_path, allow_pickle=False), dtype=bool)
    if mask.shape != (n_rows,) or not mask.any():
        raise ValueError(
            "Canonical temporal OOF eligibility mask is invalid or unaligned"
        )
    return mask


def _eligible_temporal_oof(
    oof: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Return honest OOF rows and reject fabricated temporal warm-up values."""
    if mask is None:
        return oof
    if len(oof) != len(mask):
        raise ValueError("OOF artifact is not aligned with the temporal mask")
    warmup = oof[~mask]
    if warmup.size and not np.isnan(warmup).all():
        raise ValueError("Temporal warm-up OOF rows must remain NaN")
    eligible = oof[mask]
    if not np.all(np.isfinite(eligible)):
        raise ValueError("Temporal OOF-eligible rows contain NaN or Inf")
    return eligible


def _align_test_predictions_to_submission(
    predictions: np.ndarray,
    model_ids: np.ndarray,
    submission_ids: np.ndarray,
) -> np.ndarray | None:
    """Align record-level predictions to the exact submission row order.

    Most competitions use one submission row per test record, but some encode
    one row per ``(record, class)`` pair. The latter must be expanded using a
    relationship inferred from the supplied IDs and then verified against the
    *complete* submission ID set. No positional fallback is allowed.
    """
    model_ids_str = np.asarray(model_ids).astype(str)
    submission_ids_str = np.asarray(submission_ids).astype(str)
    model_id_set = set(model_ids_str.tolist())
    submission_id_set = set(submission_ids_str.tolist())

    if (
        len(model_ids_str) == len(submission_ids_str)
        and model_id_set == submission_id_set
    ):
        position_by_id = {
            model_id: idx for idx, model_id in enumerate(model_ids_str.tolist())
        }
        reorder = np.fromiter(
            (position_by_id[submission_id] for submission_id in submission_ids_str),
            dtype=np.int64,
            count=len(submission_ids_str),
        )
        return predictions[reorder]

    if (
        predictions.ndim != 2
        or predictions.shape[1] <= 1
        or len(submission_ids_str) != len(model_ids_str) * predictions.shape[1]
    ):
        return None

    logic = infer_submission_logic(
        model_ids_str.tolist(),
        submission_ids_str.tolist(),
        num_classes=predictions.shape[1],
    )
    pattern = logic.get("pattern")
    multiplier = logic.get("multiplier")
    prediction_by_submission_id: dict[str, float] = {}

    for row_idx, model_id in enumerate(model_ids_str):
        for class_idx in range(predictions.shape[1]):
            if pattern == "multiplier" and isinstance(multiplier, int):
                try:
                    submission_id = str(int(model_id) * multiplier + class_idx)
                except ValueError:
                    return None
            elif pattern == "underscore_concat":
                submission_id = f"{model_id}_{class_idx}"
            elif pattern == "dash_concat":
                submission_id = f"{model_id}-{class_idx}"
            else:
                return None

            if submission_id in prediction_by_submission_id:
                return None
            prediction_by_submission_id[submission_id] = float(
                predictions[row_idx, class_idx]
            )

    # Inference only proposes a mapping. Full-set equality is the safety gate.
    if set(prediction_by_submission_id) != submission_id_set:
        return None

    return np.asarray(
        [prediction_by_submission_id[submission_id] for submission_id in submission_ids_str],
        dtype=np.float64,
    ).reshape(-1, 1)


class EnsembleAgent:
    """Agent responsible for creating model ensembles."""

    def __init__(self):
        """Initialize ensemble agent."""
        pass

    # Delegate to module functions (for backward compatibility with method calls)
    def _find_prediction_pairs(self, models_dir: Path) -> dict[str, tuple[Path, Path]]:
        return find_prediction_pairs(models_dir)

    def _validate_prediction_artifacts_contract(self, prediction_pairs):
        return validate_prediction_artifacts_contract(prediction_pairs)

    def _validate_oof_alignment(self, models_dir, train_ids, expected_class_order):
        return validate_oof_alignment(models_dir, train_ids, expected_class_order)

    def _encode_labels(self, y, class_order):
        return encode_labels(y, class_order)

    def _score_predictions(self, preds, y_true, problem_type, metric_name):
        return score_predictions(preds, y_true, problem_type, metric_name)

    def _compute_oof_score(self, oof_path, y_true, metric_name="log_loss"):
        return compute_oof_score(oof_path, y_true, metric_name)

    def _filter_by_score_threshold(self, prediction_pairs, y_true, metric_name, model_scores=None, threshold_pct=0.20):
        return filter_by_score_threshold(prediction_pairs, y_true, metric_name, model_scores, threshold_pct)

    def _tune_meta_model(self, meta_X, y, problem_type, n_classes=None):
        return tune_meta_model(meta_X, y, problem_type, n_classes)

    def _diagnose_stacking_issues(self, meta_model, model_names, meta_X, y):
        return diagnose_stacking_issues(meta_model, model_names, meta_X, y)

    def _constrained_meta_learner(self, oof_stack, y_true, problem_type, metric_name):
        return constrained_meta_learner(oof_stack, y_true, problem_type, metric_name)

    def _dirichlet_weight_search(self, oof_stack, y_true, problem_type, metric_name, n_samples=300):
        return dirichlet_weight_search(oof_stack, y_true, problem_type, metric_name, n_samples)

    def _validate_and_align_submission(
        self,
        submission_path,
        sample_submission_path,
        output_path=None,
        target_cols=None,
    ):
        return validate_and_align_submission(
            submission_path, sample_submission_path, output_path, target_cols
        )

    @staticmethod
    def _submission_target_cols(state) -> list[str]:
        """Prediction column names resolved from the public submission schema.

        Empty when no contract was established, which leaves every consumer on
        the positional convention it used before.
        """
        contract = state.get("submission_contract") or {}
        declared = contract.get("target_cols") or []
        return [str(column) for column in declared if isinstance(column, str) and column]

    def _safe_restore_submission(
        self,
        source_path,
        dest_path,
        sample_submission_path,
        *,
        target_cols=None,
        problem_type=None,
        expected_sha256=None,
        require_hash=False,
    ):
        return safe_restore_submission(
            source_path,
            dest_path,
            sample_submission_path,
            target_cols=target_cols,
            problem_type=problem_type,
            expected_sha256=expected_sha256,
            require_hash=require_hash,
        )

    @staticmethod
    def _is_mlebench(state: KaggleState) -> bool:
        return (
            isinstance(state, dict)
            and str(state.get("run_mode", "")).strip().lower() == "mlebench"
        )

    def _restore_preserved_submission(
        self,
        state: KaggleState,
        working_dir: Path,
        output_path: Path,
        sample_submission_path: Path,
    ) -> bool:
        """Restore the best artifact under the trust policy for the run mode."""
        submission_target_cols = self._submission_target_cols(state)
        competition_info = state.get("competition_info")
        submission_problem_type = (
            getattr(competition_info, "problem_type", None)
            if competition_info is not None
            else None
        ) or state.get("problem_type")
        if not self._is_mlebench(state):
            return self._safe_restore_submission(
                working_dir / "submission_best.csv",
                output_path,
                sample_submission_path,
                target_cols=submission_target_cols,
                problem_type=submission_problem_type,
            )

        snapshot_owner = str(
            state.get("best_candidate_submission_component_name") or ""
        )
        approvals = state.get("robustness_approved_components") or {}
        oof_claims = state.get("oof_availability") or {}
        owner_approved = (
            isinstance(approvals, dict)
            and approvals.get(snapshot_owner) is True
        )
        # Unscored-fallback lane: an owner that never claimed OOF evidence and
        # was never reviewed (canonical-less domains) has nothing for
        # robustness to approve. Its snapshot is restorable as an
        # artifact-only candidate — score provenance still requires approval
        # plus a trusted score, so no CV is ever reported for it. An owner
        # explicitly rejected (approvals[owner] is False) stays blocked.
        owner_unscored_fallback = bool(
            snapshot_owner
            and not (
                isinstance(oof_claims, dict)
                and oof_claims.get(snapshot_owner)
            )
            and (
                not isinstance(approvals, dict)
                or snapshot_owner not in approvals
            )
        )
        best_snapshot = None
        if snapshot_owner and (owner_approved or owner_unscored_fallback):
            best_snapshot = verified_best_candidate_submission(
                state, working_dir
            )
        snapshot_candidates = (
            (
                best_snapshot,
                state.get("best_candidate_submission_sha256"),
                "best candidate",
            ),
            (
                verified_accepted_submission(state, working_dir),
                state.get("accepted_submission_sha256"),
                "accepted",
            ),
        )
        for snapshot, digest, label in snapshot_candidates:
            if snapshot is None:
                continue
            if self._safe_restore_submission(
                snapshot,
                output_path,
                sample_submission_path,
                target_cols=submission_target_cols,
                problem_type=submission_problem_type,
                expected_sha256=digest,
                require_hash=True,
            ):
                print(f"      OK: Restored verified immutable {label} snapshot")
                return True

        print(
            "      Warning: No hash-verified immutable submission snapshot "
            "could be restored"
        )
        return False

    @staticmethod
    def _fail_closed_restore(
        output_path: Path,
        *,
        reason: str,
        current_iteration: int,
    ) -> dict[str, Any]:
        """Remove any mutable output when a required MLE snapshot is unavailable."""
        try:
            if output_path.is_symlink() or output_path.is_file():
                output_path.unlink(missing_ok=True)
        except OSError:
            # State validity, not successful deletion, is the authoritative
            # submission gate; SubmissionAgent also refuses invalid workflows.
            pass
        message = f"MLE-bench submission restore blocked: {reason}"
        return {
            "ensemble_skipped": True,
            "skip_reason": "verified_snapshot_unavailable",
            "workflow_valid": False,
            "submission_validation_error": message,
            "telemetry_events": [
                make_event(
                    "ensemble",
                    "snapshot_restore_blocked",
                    iteration=current_iteration,
                    reason=reason,
                )
            ],
        }

    def _load_and_align_oof(self, oof_path, train_ids_path, reference_ids):
        return load_and_align_oof(oof_path, train_ids_path, reference_ids)

    def _align_oof_by_canonical_ids(self, oof, model_train_ids, canonical_train_ids, model_name="unknown"):
        return align_oof_by_canonical_ids(oof, model_train_ids, canonical_train_ids, model_name)

    def _recover_from_checkpoints(self, models_dir, component_names=None):
        return recover_from_checkpoints(models_dir, component_names)

    def _fallback_to_best_single_model(self, models_dir, problem_type="classification"):
        return fallback_to_best_single_model(models_dir, problem_type)

    def _load_cv_folds(self, name, models_dir, folds_path, n_samples):
        return load_cv_folds(name, models_dir, folds_path, n_samples)

    def _stack_from_prediction_pairs(self, prediction_pairs, y, problem_type, metric_name, models_dir, expected_class_order, train_ids, folds_path, enable_calibration, enable_post_calibration, n_targets, calibration_method="auto"):
        return stack_from_prediction_pairs(prediction_pairs, y, problem_type, metric_name, models_dir, expected_class_order, train_ids, folds_path, enable_calibration, enable_post_calibration, n_targets, calibration_method)

    def create_ensemble_with_fallback(self, models_dir, y, problem_type, metric_name, expected_class_order=None, train_ids=None, min_models=2):
        return create_ensemble_with_fallback(models_dir, y, problem_type, metric_name, expected_class_order, train_ids, min_models)

    def create_oof_weighted_blend(
        self,
        prediction_pairs: dict[str, tuple[Path, Path]],
        y_true: np.ndarray,
        problem_type: str,
        metric_name: str,
    ) -> tuple[np.ndarray, float, dict[str, float]]:
        """Weighted blend using saved OOF predictions."""
        from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

        names = list(prediction_pairs.keys())
        n_models = len(names)

        print(f"   Creating OOF weighted blend with {n_models} models...")

        oof_list = [
            np.load(oof, allow_pickle=False)
            for oof, _ in prediction_pairs.values()
        ]
        test_list = [
            np.load(test, allow_pickle=False)
            for _, test in prediction_pairs.values()
        ]
        temporal_mask = _load_temporal_oof_mask(
            next(iter(prediction_pairs.values()))[0].parent,
            len(oof_list[0]),
        )
        oof_list = [
            _eligible_temporal_oof(np.asarray(oof), temporal_mask)
            for oof in oof_list
        ]
        if temporal_mask is not None:
            y_true = np.asarray(y_true)[temporal_mask]
        oof_stack = np.stack(oof_list, axis=0)
        test_stack = np.stack(test_list, axis=0)

        def compute_score(blended: np.ndarray) -> float:
            if problem_type == "classification":
                blended = np.clip(blended, 1e-15, 1 - 1e-15)
                if blended.ndim > 1 and blended.shape[1] > 1:
                    blended = blended / blended.sum(axis=1, keepdims=True)
                if metric_name in ["auc", "roc_auc"]:
                    from sklearn.metrics import roc_auc_score
                    return -roc_auc_score(y_true, blended, multi_class="ovr", average="weighted")
                return log_loss(y_true, blended)
            if blended.ndim > 1:
                blended = blended.ravel()
            if metric_name in ["mae", "mean_absolute_error"]:
                return mean_absolute_error(y_true, blended)
            if metric_name in ["mse", "mean_squared_error"]:
                return mean_squared_error(y_true, blended)
            return np.sqrt(mean_squared_error(y_true, blended))

        def objective(weights: np.ndarray) -> float:
            weights = np.array(weights) / np.sum(weights)
            blended = np.average(oof_stack, axis=0, weights=weights)
            return compute_score(blended)

        try:
            from scipy.optimize import minimize
            init_weights = np.ones(n_models) / n_models
            result = minimize(
                objective,
                init_weights,
                method="SLSQP",
                bounds=[(0, 1)] * n_models,
                constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
            )
            opt_weights = result.x / result.x.sum()
        except ImportError:
            opt_weights = self._dirichlet_weight_search(oof_stack, y_true, problem_type, metric_name, n_samples=300)

        blended_oof = np.average(oof_stack, axis=0, weights=opt_weights)
        oof_score = compute_score(blended_oof)

        blended_test = np.average(test_stack, axis=0, weights=opt_weights)
        if problem_type == "classification":
            blended_test = np.clip(blended_test, 1e-15, 1 - 1e-15)
            if blended_test.ndim > 1 and blended_test.shape[1] > 1:
                blended_test = blended_test / blended_test.sum(axis=1, keepdims=True)

        weights_dict = dict(zip(names, opt_weights))
        return blended_test, oof_score, weights_dict

    def _validate_class_order(self, models_dir: Path, sample_submission_path: Path) -> tuple[bool, str]:
        """Validate that saved predictions use canonical class order."""
        class_order_path = models_dir / "class_order.npy"

        if not class_order_path.exists():
            return False, "class_order.npy not found"

        if not sample_submission_path.exists():
            return False, "sample_submission.csv not found"

        try:
            saved_order = np.load(
                class_order_path,
                allow_pickle=False,
            ).tolist()
            sample_sub = read_csv_auto(sample_submission_path)
            expected_order = sample_sub.columns[1:].tolist()

            if not class_orders_match(saved_order, expected_order):
                return False, "Class order mismatch"

            return True, f"Class order validated ({len(saved_order)} classes)"
        except Exception as e:
            return False, f"Class order validation error: {e}"

    def _ensemble_from_predictions(
        self,
        prediction_pairs: dict[str, tuple[Path, Path]],
        sample_submission_path: Path,
        output_path: Path,
        models_dir: Path | None = None,
        expected_n_test: int | None = None,
        problem_type: str = "",
        metric_name: str = "",
        oof_preds: np.ndarray | None = None,
        y_true: np.ndarray | None = None,
        target_cols: list[str] | None = None,
    ) -> bool:
        """Create a simple average ensemble directly from saved predictions.

        Every prediction array is aligned to ``sample_submission`` by its
        persisted test IDs before averaging. Missing, duplicate, partial, or
        extra IDs fail closed; positional averaging is never used.

        Args:
            expected_n_test: Optional canonical test count used as an audit signal.
        """
        if not sample_submission_path.exists():
            print("   Sample submission not found")
            return False

        sample_sub = read_csv_auto(sample_submission_path)
        n_test = len(sample_sub)
        if sample_sub.shape[1] < 2:
            print("   ERROR: sample_submission must contain an ID and prediction column")
            return False
        if n_test == 0:
            print("   ERROR: sample_submission is empty")
            return False
        if (
            expected_n_test is not None
            and expected_n_test > 0
            and expected_n_test != n_test
        ):
            print(
                f"   INFO: canonical test count ({expected_n_test}) differs from "
                f"submission rows ({n_test}); exact artifact IDs remain authoritative"
            )
        if models_dir is None:
            print("   ERROR: models_dir is required for ID-safe ensembling")
            return False

        reference_ids = sample_sub.iloc[:, 0]
        if reference_ids.isna().any() or reference_ids.astype(str).duplicated().any():
            print("   ERROR: sample_submission IDs must be non-null and unique")
            return False
        reference_ids_str = reference_ids.astype(str).to_numpy()
        reference_id_set = set(reference_ids_str.tolist())

        aligned_predictions: list[np.ndarray] = []
        for name, (_, test_path) in prediction_pairs.items():
            try:
                preds = np.asarray(
                    np.load(test_path, allow_pickle=False),
                    dtype=np.float64,
                )
            except Exception as exc:
                print(f"   ERROR: Failed to load test predictions for {name}: {exc}")
                return False

            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            if preds.ndim != 2 or preds.shape[0] == 0 or preds.shape[1] == 0:
                print(f"   ERROR: Invalid prediction shape for {name}: {preds.shape}")
                return False
            if not np.all(np.isfinite(preds)):
                print(f"   ERROR: Non-finite test predictions for {name}")
                return False

            test_ids_path = models_dir / f"test_ids_{name}.npy"
            if not test_ids_path.exists():
                print(f"   ERROR: Missing test_ids_{name}.npy; refusing positional ensemble")
                return False
            try:
                model_ids = np.asarray(
                    np.load(test_ids_path, allow_pickle=False)
                ).reshape(-1)
            except Exception as exc:
                print(f"   ERROR: Failed to load test IDs for {name}: {exc}")
                return False

            if len(model_ids) != len(preds):
                print(
                    f"   ERROR: ID/prediction row mismatch for {name}: "
                    f"{len(model_ids)} vs {len(preds)}"
                )
                return False

            model_ids_series = pd.Series(model_ids)
            if model_ids_series.isna().any():
                print(f"   ERROR: Null test IDs for {name}")
                return False
            model_ids_str = model_ids_series.astype(str)
            if model_ids_str.duplicated().any():
                print(f"   ERROR: Duplicate test IDs for {name}")
                return False

            aligned = _align_test_predictions_to_submission(
                preds,
                model_ids_str.to_numpy(),
                reference_ids_str,
            )
            if aligned is None:
                model_id_set = set(model_ids_str.tolist())
                missing = len(reference_id_set - model_id_set)
                extra = len(model_id_set - reference_id_set)
                print(
                    f"   ERROR: Could not prove complete ID coverage for {name}: "
                    f"missing={missing}, extra={extra}, rows={len(model_ids)}/{n_test}"
                )
                return False

            aligned_predictions.append(aligned)
            print(f"      {name}: ID-aligned {n_test}/{n_test}")

        if not aligned_predictions:
            print("   No prediction pairs found")
            return False

        shapes = {preds.shape for preds in aligned_predictions}
        if len(shapes) != 1:
            print(f"   ERROR: Prediction shape mismatch after ID alignment: {shapes}")
            return False

        ensemble_preds = np.stack(aligned_predictions, axis=0).mean(axis=0)
        if not np.all(np.isfinite(ensemble_preds)):
            print("   ERROR: Ensemble predictions contain NaN or Inf")
            return False

        # CRITICAL: Check for constant/near-constant predictions (ID alignment bug indicator)
        pred_std = float(np.max(np.std(ensemble_preds, axis=0)))

        if pred_std < 1e-6:
            print("   ERROR: Predictions are constant (std<1e-6) - likely test ID misalignment! Check test_ids_*.npy files.")
            return False
        if pred_std < 0.01:
            print(f"   WARNING: Very low variance (std={pred_std:.6f}) - possible ID alignment issue or broken model.")
        else:
            print(f"   Predictions: min={ensemble_preds.min():.4f}, max={ensemble_preds.max():.4f}, std={pred_std:.4f}")

        # Validate and assign predictions to sample submission
        pred_positions = prediction_positions(sample_sub, target_cols)
        expected_cols = len(pred_positions)

        if ensemble_preds.shape[0] != n_test:
            print(f"   Final row count mismatch: {ensemble_preds.shape[0]} vs {n_test}")
            return False

        metric_lower = metric_name.lower()
        label_metric = any(
            token in metric_lower
            for token in (
                "accuracy",
                "f1",
                "precision",
                "recall",
                "kappa",
                "qwk",
                "mcc",
            )
        )
        if (
            ensemble_preds.shape[1] == 2
            and expected_cols == 1
            and not label_metric
        ):
            ensemble_preds = _truncate_pred_cols(ensemble_preds, expected_cols)
        elif ensemble_preds.shape[1] != expected_cols and not (
            label_metric and expected_cols == 1
        ):
            print(
                f"   ERROR: Prediction column mismatch: "
                f"{ensemble_preds.shape[1]} vs {expected_cols}"
            )
            return False

        # Without OOF evidence this call falls back to a fixed 0.5 / argmax
        # rule, which is exactly the score left on the table for hard-label
        # metrics. Pass the OOF whenever it is available and aligned.
        formatted = np.asarray(
            format_ensemble_predictions(
                ensemble_preds,
                sample_sub,
                problem_type,
                metric_name,
                oof_preds=oof_preds,
                y_true=y_true,
                target_cols=target_cols,
            )
        )
        if formatted.ndim == 1:
            formatted = formatted.reshape(-1, 1)
        if formatted.shape != (n_test, expected_cols):
            print(
                f"   ERROR: Formatted prediction shape mismatch: "
                f"{formatted.shape} vs {(n_test, expected_cols)}"
            )
            return False
        if not np.issubdtype(formatted.dtype, np.number) or not np.all(
            np.isfinite(formatted)
        ):
            print("   ERROR: Formatted predictions must be finite and numeric")
            return False

        sample_sub.iloc[:, pred_positions] = formatted

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_sub.to_csv(output_path, index=False)
        print(f"   OK: Saved prediction-only ensemble to {output_path.name}")
        return True

    def create_stacking_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        working_dir: Path,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        metric_name: str = "",
        sample_submission_path: Path | None = None,
        train_ids: np.ndarray | None = None,
        expected_class_order: list[str] | None = None,
        n_targets: int | None = None,
        folds_path: Path | None = None,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        """Create stacking ensemble from best models using saved OOF predictions."""
        print(f"  Creating stacking ensemble with {len(models)} base models...")

        models_dir = working_dir / "models"
        temporal_mask_path = (
            working_dir / "canonical" / "oof_eligible_mask.npy"
        )
        if temporal_mask_path.is_file():
            raise RuntimeError(
                "Cross-sectional stacking/cross_val_predict is disabled for "
                "temporal canonical CV; use ID-safe test-prediction averaging"
            )
        prediction_pairs = {
            name: (models_dir / f"oof_{name}.npy", models_dir / f"test_{name}.npy")
            for name in model_names
            if (models_dir / f"oof_{name}.npy").exists()
            and (models_dir / f"test_{name}.npy").exists()
        }

        enable_calibration = os.getenv("KAGGLE_AGENTS_STACKING_CALIBRATION", "1").lower() not in {"0", "false", "no"}
        enable_post_calibration = os.getenv("KAGGLE_AGENTS_STACKING_POST_CALIBRATION", "1").lower() not in {"0", "false", "no"}
        calibration_method = os.getenv("KAGGLE_AGENTS_STACKING_CALIBRATION_METHOD", "auto").lower()

        if n_targets is None and sample_submission_path and sample_submission_path.exists():
            try:
                sample_head = read_csv_auto(sample_submission_path, nrows=1)
                n_targets = sample_head.shape[1] - 1
                if expected_class_order is None and sample_head.shape[1] > 2:
                    expected_class_order = sample_head.columns[1:].tolist()
            except Exception as e:
                print(f"   Warning: Failed to read sample submission: {e}")

        if prediction_pairs:
            print(f"  Found {len(prediction_pairs)} prediction pairs for stacking")
            ensemble, final_test_preds = self._stack_from_prediction_pairs(
                prediction_pairs=prediction_pairs,
                y=y,
                problem_type=problem_type,
                metric_name=metric_name,
                models_dir=models_dir,
                expected_class_order=expected_class_order,
                train_ids=train_ids,
                folds_path=folds_path,
                enable_calibration=enable_calibration,
                enable_post_calibration=enable_post_calibration,
                n_targets=n_targets,
                calibration_method=calibration_method,
            )
            if ensemble is not None and final_test_preds is not None:
                name_to_model = dict(zip(model_names, models, strict=False))
                ensemble["base_models"] = [
                    name_to_model[name]
                    for name in ensemble.get("base_model_names", [])
                    if name in name_to_model
                ]
                return ensemble, final_test_preds

        # Fallback to cross_val_predict
        meta_features = []
        valid_models = []
        valid_names = []

        for model, name in zip(models, model_names, strict=False):
            oof_path = working_dir / "models" / f"oof_{name}.npy"
            if oof_path.exists():
                oof_preds = np.load(oof_path, allow_pickle=False)
                meta_features.append(oof_preds)
                valid_models.append(model)
                valid_names.append(name)
            else:
                if problem_type == "classification":
                    oof_preds = cross_val_predict(model, X, y, cv=5, method="predict_proba", n_jobs=-1)
                    if oof_preds.ndim > 1:
                        meta_features.append(oof_preds[:, 1])
                    else:
                        meta_features.append(oof_preds)
                else:
                    oof_preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1)
                    meta_features.append(oof_preds)
                valid_models.append(model)
                valid_names.append(name)

        if not meta_features:
            raise ValueError("No meta-features could be generated")

        meta_X = np.column_stack(meta_features)
        y_arr = y.values if hasattr(y, "values") else y
        n_classes = len(np.unique(y_arr)) if problem_type == "classification" else None
        meta_model, _ = self._tune_meta_model(meta_X, y_arr, problem_type, n_classes)
        meta_model.fit(meta_X, y)

        return {
            "meta_model": meta_model,
            "base_models": valid_models,
            "base_model_names": valid_names,
            "stacking_method": "meta",
            "weights": None,
            "class_order": None,
        }, None

    def create_blending_ensemble(
        self,
        models: list[Any],
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
    ) -> dict[str, Any]:
        """Create blending ensemble using simple averaging."""
        print(f"  Creating blending ensemble with {len(models)} models...")
        weights = self.optimize_blending_weights(models, X, y, problem_type)
        return {"base_models": models, "weights": weights}

    def optimize_blending_weights(
        self,
        models: list[Any],
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
    ) -> list[float]:
        """Optimize blending weights using scipy.minimize."""
        from scipy.optimize import minimize
        from sklearn.metrics import log_loss, mean_squared_error

        oof_preds = []
        for model in models:
            if problem_type == "classification":
                preds = cross_val_predict(model, X, y, cv=5, method="predict_proba", n_jobs=-1)
                if preds.ndim > 1:
                    oof_preds.append(preds[:, 1])
                else:
                    oof_preds.append(preds)
            else:
                preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1)
                oof_preds.append(preds)

        oof_preds = np.column_stack(oof_preds)

        def loss_func(weights):
            weights = np.array(weights)
            weights /= weights.sum()
            final_preds = np.average(oof_preds, axis=1, weights=weights)
            if problem_type == "classification":
                final_preds = np.clip(final_preds, 1e-15, 1 - 1e-15)
                return log_loss(y, final_preds)
            return np.sqrt(mean_squared_error(y, final_preds))

        init_weights = [1.0 / len(models)] * len(models)
        constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}
        bounds = [(0, 1)] * len(models)

        result = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        opt_weights = result.x / result.x.sum()
        return opt_weights.tolist()

    def create_caruana_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        working_dir: Path,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        metric_name: str = "",
        n_iterations: int = 100,
    ) -> dict[str, Any]:
        """Create ensemble using Caruana's Hill Climbing."""
        from sklearn.metrics import log_loss, mean_squared_error

        oof_preds = []
        valid_models = []
        valid_names = []

        for model, name in zip(models, model_names, strict=False):
            oof_path = working_dir / "models" / f"oof_{name}.npy"
            if oof_path.exists():
                preds = np.load(oof_path, allow_pickle=False)
                oof_preds.append(preds)
                valid_models.append(model)
                valid_names.append(name)

        if not oof_preds:
            raise ValueError("No OOF predictions found for Caruana ensemble")

        temporal_mask = _load_temporal_oof_mask(
            working_dir / "models",
            len(oof_preds[0]),
        )
        oof_preds = [
            _eligible_temporal_oof(np.asarray(preds), temporal_mask)
            for preds in oof_preds
        ]
        if temporal_mask is not None:
            y = pd.Series(np.asarray(y)[temporal_mask])

        oof_preds = np.column_stack(oof_preds)
        n_models = oof_preds.shape[1]

        def get_score(y_true, y_pred):
            if problem_type == "classification":
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                return -log_loss(y_true, y_pred)
            return -np.sqrt(mean_squared_error(y_true, y_pred))

        current_ensemble_preds = np.zeros_like(oof_preds[:, 0])
        ensemble_counts = np.zeros(n_models, dtype=int)
        best_score = -float("inf")

        for i in range(n_models):
            score = get_score(y, oof_preds[:, i])
            if score > best_score:
                best_score = score
                best_init_idx = i

        current_ensemble_preds = oof_preds[:, best_init_idx]
        ensemble_counts[best_init_idx] = 1

        for it in range(n_iterations):
            best_iter_score = -float("inf")
            best_iter_idx = -1
            current_size = it + 2

            for i in range(n_models):
                current_sum = current_ensemble_preds * (current_size - 1)
                candidate_avg = (current_sum + oof_preds[:, i]) / current_size
                score = get_score(y, candidate_avg)
                if score > best_iter_score:
                    best_iter_score = score
                    best_iter_idx = i

            ensemble_counts[best_iter_idx] += 1
            current_ensemble_preds = (
                current_ensemble_preds * (current_size - 1) + oof_preds[:, best_iter_idx]
            ) / current_size

        weights = ensemble_counts / ensemble_counts.sum()
        oof_score = self._score_predictions(
            current_ensemble_preds,
            y.values if hasattr(y, "values") else y,
            problem_type,
            metric_name,
        )

        return {
            "base_models": valid_models,
            "base_model_names": valid_names,
            "weights": weights.tolist(),
            "oof_score": oof_score,
        }

    def create_rank_average_ensemble(
        self,
        prediction_pairs: dict[str, tuple[Path, Path]],
        weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, list[str], bool]:
        """Create ensemble by averaging prediction ranks."""
        test_preds: dict[str, np.ndarray] = {}
        for name, (_, test_path) in prediction_pairs.items():
            if test_path.exists():
                try:
                    preds = np.load(test_path, allow_pickle=False)
                    if np.isfinite(preds).all():
                        test_preds[name] = preds
                except Exception:
                    pass

        if len(test_preds) < 2:
            return None, [], False

        model_names = list(test_preds.keys())
        ranked_preds: list[np.ndarray] = []
        for preds in test_preds.values():
            if preds.ndim == 1:
                ranks = rankdata(preds) / len(preds)
            else:
                ranks = np.apply_along_axis(lambda x: rankdata(x) / len(x), axis=0, arr=preds)
            ranked_preds.append(ranks)

        if weights is None:
            weights = np.ones(len(ranked_preds)) / len(ranked_preds)
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()

        stacked = np.stack(ranked_preds, axis=0)
        final_ranks = np.average(stacked, axis=0, weights=weights)
        return final_ranks, model_names, True

    def create_temporal_ensemble(
        self,
        working_dir: Path,
        submissions: list[Any],
        current_iteration: int,
        metric_name: str,
    ) -> bool:
        """Create Temporal Ensemble by blending past best submissions."""
        print(f"\n  Temporal Ensemble (Iteration {current_iteration})")

        minimize = is_metric_minimization(metric_name)
        candidates = []

        valid_history = [
            s for s in submissions
            if s.file_path and Path(s.file_path).exists() and s.public_score is not None
        ]

        for f in working_dir.glob("submission_iter_*_score_*.csv"):
            if f.name not in [Path(s.file_path).name for s in valid_history]:
                try:
                    parts = f.stem.split("_")
                    if "score" in parts:
                        score_idx = parts.index("score") + 1
                        score = float(parts[score_idx])
                        candidates.append({"path": f, "score": score})
                except Exception:
                    continue

        for sub in valid_history:
            candidates.append({"path": Path(sub.file_path), "score": sub.public_score})

        unique_candidates = {str(c["path"]): c for c in candidates}.values()
        candidates = list(unique_candidates)

        if len(candidates) < 2:
            return False

        reverse_sort = not minimize
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=reverse_sort)
        top_k = sorted_candidates[:3]

        dfs = []
        for c in top_k:
            try:
                df = pd.read_csv(c["path"])
                if "id" in df.columns:
                    df = df.sort_values("id")
                dfs.append(df)
            except Exception:
                pass

        if not dfs:
            return False

        try:
            sample = dfs[0]
            if len(sample.columns) < 2:
                return False
            pred_col = sample.columns[1]

            weights = np.array([3.0, 2.0, 1.0])[: len(dfs)]
            weights /= weights.sum()

            final_preds = np.zeros_like(sample[pred_col], dtype=float)
            for df, w in zip(dfs, weights):
                final_preds += df[pred_col].values * w

            output = sample.copy()
            output[pred_col] = final_preds
            output.to_csv(working_dir / "submission.csv", index=False)
            print("   OK: Temporal ensemble saved")
            return True
        except Exception as e:
            print(f"   Warning: Temporal ensemble failed: {e}")
            return False

    def predict_stacking(
        self, ensemble: dict[str, Any], X: pd.DataFrame, problem_type: str
    ) -> np.ndarray:
        """Make predictions using stacking ensemble."""
        meta_model = ensemble.get("meta_model")
        base_models = ensemble.get("base_models", [])
        weights = ensemble.get("weights")

        if meta_model is None and weights is not None:
            # Weighted average
            predictions = []
            weights_array = np.array(weights)
            weights_array = weights_array / weights_array.sum()
            for model in base_models:
                if problem_type == "classification" and hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X)
                    if preds.ndim > 1:
                        predictions.append(preds[:, 1])
                    else:
                        predictions.append(preds)
                else:
                    predictions.append(model.predict(X))
            return np.average(predictions, axis=0, weights=weights_array)

        # Meta-model stacking
        meta_features = []
        binary_single_col = False
        for model in base_models:
            if problem_type == "classification" and hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)
                if preds.ndim > 1 and preds.shape[1] > 2:
                    meta_features.append(preds)
                elif preds.ndim > 1:
                    meta_features.append(preds[:, 1])
                    binary_single_col = True
                else:
                    meta_features.append(preds)
                    binary_single_col = True
            else:
                meta_features.append(model.predict(X))

        if meta_features and isinstance(meta_features[0], np.ndarray) and meta_features[0].ndim > 1:
            meta_X = np.concatenate(meta_features, axis=1)
        else:
            meta_X = np.column_stack(meta_features)

        if problem_type == "classification" and hasattr(meta_model, "predict_proba"):
            preds = meta_model.predict_proba(meta_X)
            if binary_single_col and preds.ndim > 1:
                return preds[:, 1]
            return preds
        return meta_model.predict(meta_X)

    def predict_blending(
        self, ensemble: dict[str, Any], X: pd.DataFrame, problem_type: str
    ) -> np.ndarray:
        """Make predictions using blending ensemble."""
        base_models = ensemble["base_models"]
        weights = ensemble["weights"]

        predictions = []
        for model in base_models:
            if problem_type == "classification" and hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)
                if preds.ndim > 1:
                    predictions.append(preds[:, 1])
                else:
                    predictions.append(preds)
            else:
                predictions.append(model.predict(X))

        return np.average(predictions, axis=0, weights=weights)

    def select_ensemble_strategy(
        self,
        oof_coverage: float,
        problem_type: str,
        metric_name: str,
    ) -> str:
        """Select ensemble strategy based on OOF coverage and problem type."""
        ranking_metrics = {'auc', 'roc_auc', 'map', 'ndcg', 'mrr', 'log_loss', 'logloss'}
        is_ranking_metric = any(m in metric_name.lower() for m in ranking_metrics)

        if oof_coverage >= 0.95:
            strategy = "stacking"
        elif oof_coverage >= 0.70:
            strategy = "intersection_stacking"
        elif is_ranking_metric or problem_type == "classification":
            strategy = "rank_averaging"
        else:
            strategy = "weighted_averaging"

        return strategy

    def plan_ensemble_strategy(
        self, models: list[Any], problem_type: str, eda_summary: dict[str, Any]
    ) -> dict[str, Any]:
        """Plan ensemble strategy using LLM."""
        import json

        from langchain_core.messages import HumanMessage

        from ...core.config import get_llm

        llm = get_llm()
        model_descriptions = [f"Model {i + 1}: {type(m).__name__}" for i, m in enumerate(models)]

        prompt = f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- We have {len(models)} trained models: {", ".join(model_descriptions)}.
- Problem Type: {problem_type}
- EDA Insights: {str(eda_summary)[:500]}...

# Your task
- Suggest a plan to ensemble these solutions.
- Consider: caruana_ensemble, stacking, weighted_blending, or rank_averaging.

# Response format
Return a JSON object with: strategy_name, description, meta_learner_config (if applicable)
"""
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = get_text_content(response.content).strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except Exception:
            return {
                "strategy_name": "weighted_blending",
                "description": "Fallback to weighted blending",
            }

    def _load_canonical_training_data(
        self, state: KaggleState
    ) -> tuple[np.ndarray | pd.Series | None, np.ndarray | None, Path | None]:
        """Load y / train_ids / folds path from the canonical contract (single source of truth)."""
        contract = state.get("canonical_contract") if isinstance(state, dict) else None
        if not isinstance(contract, dict):
            return None, None, None
        try:
            y_path = contract.get("y_path")
            if not y_path or not Path(y_path).exists():
                return None, None, None
            y_array = np.asarray(np.load(y_path, allow_pickle=True))
            y = pd.Series(y_array) if y_array.ndim == 1 else y_array

            train_ids = None
            train_ids_path = contract.get("train_ids_path")
            if train_ids_path and Path(train_ids_path).exists():
                train_ids = np.load(train_ids_path, allow_pickle=True)

            folds_path = contract.get("folds_path")
            folds_path = Path(folds_path) if folds_path else None
            return y, train_ids, folds_path
        except Exception as e:
            print(f"   Canonical training data unavailable for stacking: {e}")
            return None, None, None

    @staticmethod
    def _get_comparable_cv_score(state: KaggleState) -> tuple[float | None, str | None]:
        """Return an internal CV score that is comparable with ensemble OOF.

        Leaderboard/MLE-bench ``best_score`` is intentionally excluded: comparing
        it with OOF mixes different datasets and can incorrectly reject an ensemble.
        """
        if not isinstance(state, dict):
            return None, None
        for field in ("best_single_model_score", "baseline_cv_score"):
            value = state.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                score = float(value)
                if np.isfinite(score):
                    return score, field
        return None, None

    def _load_aligned_oof(
        self,
        prediction_pairs: dict[str, tuple[Path, Path]],
        y: Any,
        models_dir: Path | None,
    ) -> np.ndarray | None:
        """Mean 1-D OOF for the given pairs, aligned with canonical ``y``.

        Fails closed: any shape disagreement, non-finite value, or column count
        other than one yields ``None``, because a misaligned OOF would tune the
        decision rule against the wrong labels.
        """
        if models_dir is None or y is None:
            return None

        n_train = len(np.asarray(y))
        columns: list[np.ndarray] = []
        for name, (oof_path, _) in prediction_pairs.items():
            try:
                oof = np.asarray(np.load(oof_path, allow_pickle=False), dtype=np.float64)
            except Exception as exc:
                print(f"   OOF unavailable for postprocessing ({name}): {exc}")
                return None
            oof = oof.reshape(len(oof), -1)
            if oof.shape[0] != n_train or oof.shape[1] != 1:
                # Multi-column OOF needs the class-order machinery that only the
                # stacking path carries.
                return None
            if not np.all(np.isfinite(oof)):
                return None
            columns.append(oof[:, 0])

        if not columns:
            return None
        return np.mean(np.stack(columns, axis=0), axis=0)

    def _try_single_model_postprocessing(
        self,
        state: KaggleState,
        prediction_pairs: dict[str, tuple[Path, Path]],
        models_dir: Path,
        sample_path: Path,
        output_path: Path,
        problem_type: str,
        metric_name: str,
        current_iteration: int,
    ) -> dict[str, Any] | None:
        """OOF-tuned decision rule for a run with a single accepted model.

        Stacking requires at least two models, so a single-model run shipped
        whatever decision rule the generated script happened to pick -- a fixed
        0.5 threshold or a plain argmax -- even on competitions scored on hard
        labels. The tuning helpers existed and were tested, but no reachable
        path could call them. On accuracy/F1/QWK tasks that is a deterministic
        loss of score.

        Returns the node outcome dict when it produced a better-scoring
        artifact, or ``None`` to leave the existing behaviour untouched.
        """
        if len(prediction_pairs) != 1 or not sample_path.exists():
            return None
        if metric_label_kind(metric_name) is None:
            # Probability/regression metric: a decision rule would only destroy
            # information.
            return None

        y, _train_ids, _folds = self._load_canonical_training_data(state)
        if y is None:
            return None

        working_dir = Path(state["working_directory"])
        if (working_dir / "canonical" / "oof_eligible_mask.npy").is_file():
            # Temporal OOF carries warm-up NaNs; tuning would score them.
            return None

        oof_1d = self._load_aligned_oof(prediction_pairs, y, models_dir)
        if oof_1d is None:
            return None

        try:
            sample_sub = read_csv_auto(sample_path)
        except Exception:
            return None
        pred_positions = prediction_positions(
            sample_sub, self._submission_target_cols(state)
        )
        if len(pred_positions) != 1:
            # Multi-column submissions are the stacking path's business.
            return None

        name = next(iter(prediction_pairs))
        test_path = prediction_pairs[name][1]
        try:
            test_preds = np.asarray(
                np.load(test_path, allow_pickle=False), dtype=np.float64
            )
        except Exception:
            return None
        test_preds = test_preds.reshape(len(test_preds), -1)
        if test_preds.shape[1] != 1 or not np.all(np.isfinite(test_preds)):
            return None

        if is_metric_minimization(metric_name):
            # Every hard-label metric this path handles is a maximize metric;
            # the score convention below depends on that.
            return None

        y_values = np.asarray(y).reshape(-1)
        try:
            labels, info = labels_from_oof_tuning(
                test_preds[:, 0], oof_1d, y_values, metric_name
            )
        except Exception as exc:
            print(f"   [POSTPROC] Tuning unavailable: {exc}")
            return None

        # `oof_score_tuned` is the metric under the tuned rule, evaluated on the
        # ORIGINAL label values. Re-scoring through score_predictions instead
        # would be wrong: its hard-label path re-thresholds at 0.5, so a label
        # set like {1, 2} collapses to a single class and a perfect classifier
        # scores as noise. For a maximize label metric the trusted convention
        # (negate a lower-is-better loss) reduces to the raw metric, so this
        # value is directly comparable with trusted_component_scores.
        tuned_score = float(info["oof_score_tuned"])
        if not np.isfinite(tuned_score):
            return None

        best_existing, best_source = self._get_comparable_cv_score(state)
        if best_existing is not None and (
            calculate_score_improvement(tuned_score, float(best_existing), metric_name)
            < 0
        ):
            print(
                f"   [POSTPROC] Tuned rule scores {tuned_score:.6f} vs "
                f"{best_source}={float(best_existing):.6f}; keeping the existing artifact"
            )
            return None

        aligned = self._align_labels_to_submission(
            labels, models_dir / f"test_ids_{name}.npy", sample_sub
        )
        if aligned is None:
            return None

        sample_sub.iloc[:, pred_positions[0]] = aligned
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_sub.to_csv(output_path, index=False)
        print(
            f"   [POSTPROC] {info['rule']} on a single model: OOF "
            f"{info['oof_score_baseline']:.4f} -> {info['oof_score_tuned']:.4f}"
        )

        try:
            digest = sha256_file(output_path)
        except OSError:
            return None

        return {
            "ensemble_created": True,
            "ensemble_strategy": "single_model_postprocessing",
            "ensemble_oof_score": tuned_score,
            "ensemble_submission_sha256": digest,
            "ensemble_submission_owner": "ensemble",
            "ensemble_score_source": "host_oof_postprocessing",
            "n_models": 1,
            "telemetry_events": [
                make_event(
                    "ensemble",
                    "created",
                    iteration=current_iteration,
                    strategy="single_model_postprocessing",
                    n_models=1,
                    oof_score=tuned_score,
                    postprocessing=info["rule"],
                )
            ],
        }

    @staticmethod
    def _align_labels_to_submission(
        labels: np.ndarray,
        test_ids_path: Path,
        sample_sub: pd.DataFrame,
    ) -> np.ndarray | None:
        """Map per-row labels onto submission order by persisted test IDs."""
        if not test_ids_path.exists():
            print("   [POSTPROC] Missing test IDs; refusing positional alignment")
            return None
        try:
            model_ids = np.asarray(
                np.load(test_ids_path, allow_pickle=True)
            ).reshape(-1)
        except Exception:
            return None
        if len(model_ids) != len(labels):
            return None

        reference = sample_sub.iloc[:, 0]
        if reference.isna().any() or reference.astype(str).duplicated().any():
            return None
        model_ids_str = pd.Series(model_ids).astype(str)
        if model_ids_str.isna().any() or model_ids_str.duplicated().any():
            return None

        lookup = dict(zip(model_ids_str.tolist(), list(labels), strict=True))
        try:
            return np.asarray([lookup[key] for key in reference.astype(str).tolist()])
        except KeyError:
            print("   [POSTPROC] Incomplete test-ID coverage; leaving the artifact alone")
            return None

    def _try_oof_stacking(
        self,
        state: KaggleState,
        prediction_pairs: dict[str, tuple[Path, Path]],
        models_dir: Path,
        sample_path: Path,
        output_path: Path,
        problem_type: str,
        metric_name: str,
        current_iteration: int,
    ) -> dict[str, Any] | None:
        """
        OOF-scored ensemble route: compares simple average vs constrained weights
        vs tuned meta-model on OOF score (with base/post calibration), applies
        metric-aware postprocessing, and never overwrites a submission that beats
        the ensemble's OOF score.

        Returns the node outcome dict on success, or None to fall back to the
        simple-average path.
        """
        if len(prediction_pairs) < 2 or not sample_path.exists():
            return None

        y, train_ids, folds_path = self._load_canonical_training_data(state)
        if y is None:
            print("   No canonical y.npy - using simple average ensemble")
            return None

        canonical_contract = (
            state.get("canonical_contract")
            if isinstance(state, dict)
            else None
        ) or {}
        temporal_mask_path = Path(
            canonical_contract.get("oof_eligible_mask_path")
            or Path(state["working_directory"])
            / "canonical"
            / "oof_eligible_mask.npy"
        )
        if temporal_mask_path.is_file():
            # The current meta-model uses cross-sectional nested CV. Running it
            # on expanding-window OOF would reintroduce future leakage at the
            # ensemble layer. Test-prediction averaging remains available and
            # does not fit a second-level model on temporal OOF.
            print(
                "   Temporal OOF mask detected - disabling cross-sectional "
                "stacking and using the ID-safe prediction averaging path"
            )
            return None

        try:
            sample_sub = read_csv_auto(sample_path)
        except Exception as e:
            print(f"   Could not read sample submission ({e}) - using simple average")
            return None

        n_test = len(sample_sub)
        submission_target_cols = self._submission_target_cols(state)
        pred_positions = prediction_positions(sample_sub, submission_target_cols)
        expected_cols = len(pred_positions)

        contract = (state.get("canonical_contract") if isinstance(state, dict) else None) or {}
        is_classification = contract.get("is_classification")
        if is_classification is None:
            is_classification = "class" in (problem_type or "").lower()
        norm_problem = "classification" if is_classification else "regression"

        # Class order / target shape from the submission template. A two-column
        # file can still be a long multiclass grid, so infer that relationship
        # from canonical test IDs and the complete submission IDs.
        expected_class_order = (
            [str(sample_sub.columns[position]) for position in pred_positions]
            if expected_cols > 1
            else None
        )
        n_targets = expected_cols
        canonical_test_ids = (
            state.get("test_rec_ids", []) if isinstance(state, dict) else []
        )
        if expected_cols == 1 and canonical_test_ids:
            submission_logic = infer_submission_logic(
                list(canonical_test_ids),
                sample_sub.iloc[:, 0].tolist(),
            )
            inferred_classes = submission_logic.get("inferred_classes")
            if (
                submission_logic.get("pattern") != "direct"
                and isinstance(inferred_classes, int)
                and inferred_classes > 1
            ):
                n_targets = inferred_classes

        enable_calibration = os.getenv(
            "KAGGLE_AGENTS_STACKING_CALIBRATION", "1"
        ).lower() not in {"0", "false", "no"}
        enable_post_calibration = os.getenv(
            "KAGGLE_AGENTS_STACKING_POST_CALIBRATION", "1"
        ).lower() not in {"0", "false", "no"}
        calibration_method = os.getenv(
            "KAGGLE_AGENTS_STACKING_CALIBRATION_METHOD", "auto"
        ).lower()

        try:
            ensemble, final_test_preds = stack_from_prediction_pairs(
                prediction_pairs=prediction_pairs,
                y=y,
                problem_type=norm_problem,
                metric_name=metric_name,
                models_dir=models_dir,
                expected_class_order=expected_class_order,
                train_ids=train_ids,
                folds_path=folds_path,
                enable_calibration=enable_calibration,
                enable_post_calibration=enable_post_calibration,
                n_targets=n_targets,
                calibration_method=calibration_method,
                require_identity_artifacts=self._is_mlebench(state),
            )
        except Exception as e:
            print(f"   OOF stacking failed ({e}) - using simple average")
            return None

        if ensemble is None or final_test_preds is None:
            print("   OOF stacking not applicable - using simple average")
            return None

        final_test_preds = np.asarray(final_test_preds, dtype=float)
        if final_test_preds.ndim == 1:
            final_test_preds = final_test_preds.reshape(-1, 1)
        ensemble_test_ids = ensemble.get("test_ids")
        if ensemble_test_ids is None:
            print("   Stacking did not preserve test IDs - using ID-safe fallback")
            return None
        aligned_test_preds = _align_test_predictions_to_submission(
            final_test_preds,
            np.asarray(ensemble_test_ids),
            sample_sub.iloc[:, 0].astype(str).to_numpy(),
        )
        if aligned_test_preds is None:
            print(
                "   Could not prove stacked prediction alignment to every "
                "submission ID - using ID-safe fallback"
            )
            return None
        final_test_preds = aligned_test_preds

        n_models = len(ensemble.get("base_model_names", []))
        strategy = f"stacking_{ensemble.get('stacking_method', 'unknown')}"

        # score_predictions is lower-is-better (negated for maximize metrics);
        # convert back to raw metric units for comparison/reporting
        oof_internal = ensemble.get("oof_score")
        raw_oof_score = None
        if isinstance(oof_internal, (int, float)) and np.isfinite(oof_internal):
            raw_oof_score = (
                float(oof_internal)
                if is_metric_minimization(metric_name)
                else -float(oof_internal)
            )

        # Only compare OOF with an internal score measured on comparable CV data.
        # Never compare it with state.best_score (leaderboard/private holdout).
        best_existing, best_existing_source = self._get_comparable_cv_score(state)
        mlebench_mode = self._is_mlebench(state)
        if (
            (output_path.exists() or mlebench_mode)
            and raw_oof_score is not None
            and best_existing is not None
        ):
            if is_metric_minimization(metric_name):
                ensemble_better = raw_oof_score <= float(best_existing)
            else:
                ensemble_better = raw_oof_score >= float(best_existing)
            if not ensemble_better:
                print(
                    f"   Keeping existing submission: ensemble OOF {raw_oof_score:.6f} "
                    f"vs component CV {float(best_existing):.6f} "
                    f"from {best_existing_source} ({metric_name})"
                )
                if mlebench_mode:
                    working_dir = Path(state["working_directory"])
                    if not self._restore_preserved_submission(
                        state,
                        working_dir,
                        output_path,
                        sample_path,
                    ):
                        return self._fail_closed_restore(
                            output_path,
                            reason=(
                                "inferior stacked candidate has no verified "
                                "snapshot to restore"
                            ),
                            current_iteration=current_iteration,
                        )
                return {
                    "ensemble_created": True,
                    "ensemble_strategy": strategy,
                    "n_models": n_models,
                    "telemetry_events": [
                        make_event(
                            "ensemble",
                            "kept_existing_submission",
                            iteration=current_iteration,
                            strategy=strategy,
                            oof_score=raw_oof_score,
                            best_component_cv_score=float(best_existing),
                            cv_score_source=best_existing_source,
                        )
                    ],
                }

        # Format predictions (metric-aware postprocessing tuned on the ensemble OOF)
        from .postprocessing import metric_label_kind
        from .submission import format_ensemble_predictions

        selected_oof = ensemble.get("selected_oof")
        class_order = ensemble.get("class_order")
        postproc_rule = "none"

        if final_test_preds.shape[1] > 1 and expected_cols == 1:
            # Multiclass probabilities -> single-label column
            label_kind = metric_label_kind(metric_name)
            numeric_classes = None
            if class_order is not None and len(class_order) == final_test_preds.shape[1]:
                try:
                    numeric_classes = np.asarray([float(c) for c in class_order])
                except (TypeError, ValueError):
                    numeric_classes = None

            if (
                label_kind == "qwk"
                and numeric_classes is not None
                and selected_oof is not None
                and np.asarray(selected_oof).ndim == 2
            ):
                # Ordinal metric: expected value over classes + OOF-tuned rounding
                from .postprocessing import labels_from_oof_tuning

                test_expect = final_test_preds @ numeric_classes
                oof_expect = np.asarray(selected_oof) @ numeric_classes
                labels, info = labels_from_oof_tuning(
                    test_expect, oof_expect, np.asarray(y), metric_name
                )
                postproc_rule = info["rule"]
                print(
                    f"   [POSTPROC] {info['rule']}: OOF "
                    f"{info['oof_score_baseline']:.4f} -> {info['oof_score_tuned']:.4f}"
                )
                sample_sub.iloc[:, pred_positions[0]] = labels
            else:
                idx = np.argmax(final_test_preds, axis=1)
                if class_order is not None and len(class_order) == final_test_preds.shape[1]:
                    # argmax gives encoded indices; map back to original labels
                    sample_sub.iloc[:, pred_positions[0]] = np.asarray(class_order)[idx]
                    postproc_rule = "argmax_class_order"
                else:
                    sample_sub.iloc[:, pred_positions[0]] = idx
                    postproc_rule = "argmax"
        else:
            if final_test_preds.shape[1] > expected_cols:
                final_test_preds = _truncate_pred_cols(final_test_preds, expected_cols)

            formatted = format_ensemble_predictions(
                final_test_preds[:, 0] if final_test_preds.shape[1] == 1 else final_test_preds,
                sample_sub,
                norm_problem,
                metric_name,
                oof_preds=selected_oof,
                y_true=np.asarray(y),
                target_cols=submission_target_cols,
            )
            formatted = np.asarray(formatted)
            if formatted.ndim == 1:
                formatted = formatted.reshape(-1, 1)

            if formatted.shape[1] == 1:
                sample_sub.iloc[:, pred_positions[0]] = formatted[:, 0]
            elif formatted.shape[1] == expected_cols:
                sample_sub.iloc[:, pred_positions] = formatted
            else:
                print(f"   Unexpected formatted shape {formatted.shape} - using simple average")
                return None
            if metric_label_kind(metric_name) is not None:
                postproc_rule = "oof_tuned" if selected_oof is not None else "fixed_threshold"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_sub.to_csv(output_path, index=False)
        oof_display = f"{raw_oof_score:.6f}" if raw_oof_score is not None else "n/a"
        print(
            f"   OK: OOF-scored ensemble ({strategy}, {n_models} models, "
            f"OOF={oof_display}) saved to {output_path.name}"
        )

        weights = ensemble.get("weights")
        ensemble_weights = (
            dict(zip(ensemble.get("base_model_names", []), weights, strict=False))
            if weights
            else {}
        )
        ensemble_digest = None
        ensemble_score_source = None
        ensemble_owner = None
        if raw_oof_score is not None:
            try:
                ensemble_digest = sha256_file(output_path)
            except OSError as exc:
                print(
                    "   Could not bind ensemble OOF score to submission bytes "
                    f"({exc}); the artifact will remain unscored"
                )
            else:
                ensemble_score_source = "host_oof_ensemble"
                ensemble_owner = "ensemble"

        return {
            "ensemble_created": True,
            "ensemble_strategy": strategy,
            "ensemble_weights": ensemble_weights,
            "ensemble_oof_score": (
                raw_oof_score if ensemble_digest is not None else None
            ),
            "ensemble_submission_sha256": ensemble_digest,
            "ensemble_submission_owner": ensemble_owner,
            "ensemble_score_source": ensemble_score_source,
            "n_models": n_models,
            "telemetry_events": [
                make_event(
                    "ensemble",
                    "created",
                    iteration=current_iteration,
                    strategy=ensemble.get("stacking_method"),
                    n_models=n_models,
                    oof_score=raw_oof_score,
                    postprocessing=postproc_rule,
                )
            ],
        }

    def __call__(self, state: KaggleState) -> dict[str, Any]:
        """Create ensemble from trained models.

        This is the main entry point for the ensemble agent.
        The full implementation is in the original ensemble_agent.py file.
        """
        # Import the full __call__ implementation
        # For now, delegate to a simplified version
        print("\n" + "=" * 60)
        print("ENSEMBLE AGENT: Creating Model Ensemble")
        print("=" * 60)

        errors = []
        if isinstance(state, dict):
            errors = list(state.get("errors", []) or [])

        current_iteration = (
            state.get("current_iteration", 0) if isinstance(state, dict) else 0
        )

        # Ablation toggle: ensembling disabled -> keep best single-model submission
        toggles = getattr(get_config(), "ablation_toggles", None)
        if toggles and toggles.disable_ensemble:
            print("   ABLATION: Ensemble disabled - keeping best single-model submission")
            if self._is_mlebench(state):
                working_dir = Path(str(state.get("working_directory") or ""))
                output_path = working_dir / "submission.csv"
                sample_value = state.get("sample_submission_path")
                sample_path = (
                    Path(str(sample_value))
                    if sample_value
                    else working_dir / "sample_submission.csv"
                )
                if not self._restore_preserved_submission(
                    state,
                    working_dir,
                    output_path,
                    sample_path,
                ):
                    return self._fail_closed_restore(
                        output_path,
                        reason="ensemble disabled but no verified snapshot exists",
                        current_iteration=current_iteration,
                    )
            return {
                "ensemble_skipped": True,
                "skip_reason": "ablation_disabled",
                "telemetry_events": [
                    make_event(
                        "ablation",
                        "ensemble_skipped",
                        iteration=current_iteration,
                        component="ensemble",
                    )
                ],
            }

        try:
            working_dir_value = (
                state.get("working_directory", "")
                if isinstance(state, dict)
                else state.working_directory
            )
            working_dir = Path(working_dir_value) if working_dir_value else Path()
            models_dir = working_dir / "models"
            sample_submission_path = (
                state.get("sample_submission_path", "")
                if isinstance(state, dict)
                else state.sample_submission_path
            )

            # Problem type / metric from state contracts (single source of truth)
            competition_info = (
                state.get("competition_info") if isinstance(state, dict) else None
            )
            problem_type = str(getattr(competition_info, "problem_type", "") or "")
            metric_name = str(getattr(competition_info, "evaluation_metric", "") or "")
            metric_contract = (
                state.get("metric_contract") if isinstance(state, dict) else None
            ) or {}
            if isinstance(metric_contract, dict) and metric_contract.get("name"):
                metric_name = str(metric_contract["name"])

            sample_path = Path(sample_submission_path) if sample_submission_path else working_dir / "sample_submission.csv"
            output_path = working_dir / "submission.csv"
            best_submission = working_dir / "submission_best.csv"
            mlebench_mode = self._is_mlebench(state)

            # Find prediction pairs
            prediction_pairs = self._find_prediction_pairs(models_dir)
            print(f"   Found {len(prediction_pairs)} prediction pairs")

            # Only ensemble artifacts explicitly accepted by the hill-climb.
            # Filesystem presence is not evidence of acceptance: a rejected or
            # stale component can leave well-formed arrays behind.
            availability = (
                state.get("oof_availability", {})
                if isinstance(state, dict)
                else {}
            )
            robustness_approvals = (
                state.get("robustness_approved_components", {})
                if isinstance(state, dict)
                else {}
            )
            trusted_scores = (
                state.get("trusted_component_scores", {})
                if isinstance(state, dict)
                else {}
            )
            trusted_names: set[str] = set()
            if isinstance(trusted_scores, dict):
                for name, value in trusted_scores.items():
                    raw_score = (
                        value.get("score", value.get("cv_score"))
                        if isinstance(value, dict)
                        else value
                    )
                    try:
                        score = float(raw_score)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(score):
                        trusted_names.add(str(name))
            accepted = {
                str(name)
                for name, is_available in availability.items()
                if is_available is True
                and robustness_approvals.get(str(name)) is True
                and (not mlebench_mode or str(name) in trusted_names)
            }
            dropped = sorted(set(prediction_pairs) - accepted)
            prediction_pairs = {
                name: pair
                for name, pair in prediction_pairs.items()
                if name in accepted
            }
            if dropped:
                print(
                    "   Excluding prediction pairs without OOF, robustness "
                    "acceptance, and (for MLE-bench) a trusted score: "
                    + ", ".join(dropped)
                )

            if len(prediction_pairs) < 1:
                print("   No prediction pairs found, skipping ensemble")
                restored = self._restore_preserved_submission(
                    state,
                    working_dir,
                    output_path,
                    sample_path,
                )
                if mlebench_mode and not restored:
                    return self._fail_closed_restore(
                        output_path,
                        reason="no prediction pairs and no verified snapshot",
                        current_iteration=current_iteration,
                    )
                return {
                    "ensemble_skipped": True,
                    "skip_reason": "no_prediction_pairs",
                    "telemetry_events": [
                        make_event(
                            "ensemble",
                            "skipped",
                            iteration=current_iteration,
                            reason="no_prediction_pairs",
                        )
                    ],
                }

            # Get expected test count from CVfolds (if available) for validation
            test_rec_ids = state.get("test_rec_ids", []) if isinstance(state, dict) else []
            expected_n_test = len(test_rec_ids) if test_rec_ids else None

            # OOF-scored stacking first (average vs constrained weights vs
            # meta-model, compared on OOF + calibration + metric-aware
            # postprocessing); falls back to the simple average below
            stacking_outcome = self._try_oof_stacking(
                state=state,
                prediction_pairs=prediction_pairs,
                models_dir=models_dir,
                sample_path=sample_path,
                output_path=output_path,
                problem_type=problem_type,
                metric_name=metric_name,
                current_iteration=current_iteration,
            )
            if stacking_outcome is not None:
                return stacking_outcome

            # Stacking needs two models. A single accepted model on a
            # hard-label metric still deserves an OOF-tuned decision rule
            # instead of whatever threshold the generated script chose.
            postprocessing_outcome = self._try_single_model_postprocessing(
                state=state,
                prediction_pairs=prediction_pairs,
                models_dir=models_dir,
                sample_path=sample_path,
                output_path=output_path,
                problem_type=problem_type,
                metric_name=metric_name,
                current_iteration=current_iteration,
            )
            if postprocessing_outcome is not None:
                return postprocessing_outcome

            # OOF stacking unavailable (e.g. no canonical y.npy on image comps).
            # The simple average below is UNSCORED - never let it overwrite a
            # scored hill-climb best.
            best_existing, best_source = self._get_comparable_cv_score(state)
            if best_existing is not None and (
                mlebench_mode or best_submission.exists()
            ):
                print(
                    f"   Unscored fallback blocked: keeping scored best "
                    f"({best_source}={best_existing:.6f})"
                )
                restored = self._restore_preserved_submission(
                    state,
                    working_dir,
                    output_path,
                    sample_path,
                )
                if mlebench_mode and not restored:
                    return self._fail_closed_restore(
                        output_path,
                        reason="scored best has no verified snapshot",
                        current_iteration=current_iteration,
                    )
                if restored:
                    return {
                        "ensemble_skipped": True,
                        "skip_reason": "unscored_fallback_kept_scored_best",
                        "telemetry_events": [
                            make_event(
                                "ensemble",
                                "skipped",
                                iteration=current_iteration,
                                reason="unscored_fallback_kept_scored_best",
                                best_score=float(best_existing),
                                score_source=best_source,
                            )
                        ],
                    }
                print(
                    "   Preserved submission could not be validated; "
                    "continuing to the ID-safe fallback"
                )

            fallback_y, _, _ = self._load_canonical_training_data(state)
            fallback_oof = self._load_aligned_oof(
                prediction_pairs, fallback_y, models_dir
            )
            if self._ensemble_from_predictions(
                prediction_pairs,
                sample_path,
                output_path,
                models_dir,
                expected_n_test,
                problem_type=problem_type,
                metric_name=metric_name,
                oof_preds=fallback_oof,
                y_true=(
                    np.asarray(fallback_y).reshape(-1)
                    if fallback_oof is not None and fallback_y is not None
                    else None
                ),
                target_cols=self._submission_target_cols(state),
            ):
                return {
                    "ensemble_created": True,
                    # This fallback has no host-side OOF measurement. Clear any
                    # prior ensemble provenance so it cannot inherit a stale score.
                    "ensemble_oof_score": None,
                    "ensemble_submission_sha256": None,
                    "ensemble_submission_owner": None,
                    "ensemble_score_source": None,
                    "n_models": len(prediction_pairs),
                    "telemetry_events": [
                        make_event(
                            "ensemble",
                            "created",
                            iteration=current_iteration,
                            n_models=len(prediction_pairs),
                        )
                    ],
                }
            return {
                "ensemble_skipped": True,
                "skip_reason": "ensemble_creation_failed",
                "telemetry_events": [
                    make_event(
                        "ensemble",
                        "skipped",
                        iteration=current_iteration,
                        reason="ensemble_creation_failed",
                    )
                ],
            }

        except Exception as e:
            error_msg = f"Ensemble creation failed: {e!s}"
            print(f"Ensemble Agent ERROR: {error_msg}")
            errors.append(error_msg)
            return {
                "errors": errors,
                "ensemble_skipped": True,
                "skip_reason": "exception",
                "telemetry_events": [
                    make_event(
                        "ensemble",
                        "skipped",
                        iteration=current_iteration,
                        reason="exception",
                        error=str(e)[:300],
                    )
                ],
            }


def ensemble_agent_node(state: KaggleState) -> dict[str, Any]:
    """LangGraph node function for ensemble agent."""
    agent = EnsembleAgent()
    return agent(state)
