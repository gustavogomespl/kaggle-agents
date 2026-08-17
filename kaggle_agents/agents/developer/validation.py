"""
Score extraction and component validation.

Provides capabilities for extracting CV scores from stdout and
validating component improvements using hill climbing strategy.
"""

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...core.config import calculate_score_improvement, is_metric_minimization
from ...core.state import AblationComponent, KaggleState
from ...utils.bounded_array import (
    DEFAULT_CHUNK_ROWS,
    ProgressCallback,
    load_npy_readonly,
    string_arrays_equal,
)
from ...utils.image_to_image_contract import packed_image_rmse
from ..ensemble.scoring import score_predictions


if TYPE_CHECKING:
    from ...tools.code_executor import ExecutionResult


TRUSTED_OOF_CHUNK_ROWS = DEFAULT_CHUNK_ROWS
TRUSTED_OOF_HEARTBEAT_ROWS = 1_000_000


def _trusted_oof_progress(
    component_name: str,
    stage: str,
) -> ProgressCallback:
    """Build a throttled progress logger for host-side trusted validation."""
    last_reported = 0

    def report(processed: int, total: int) -> None:
        nonlocal last_reported
        if (
            processed == total
            or processed - last_reported >= TRUSTED_OOF_HEARTBEAT_ROWS
        ):
            print(
                "      [LOG:PROGRESS] trusted_oof "
                f"component={component_name} stage={stage} "
                f"rows={processed}/{total}"
            )
            last_reported = processed

    return report


# These four live here rather than beside their first caller because the
# developer decides which artifacts a component must write and the robustness
# gate decides whether the artifacts it finds are sufficient. When those two
# answers came from separate rules they disagreed, and the disagreement was
# unobservable: a multilabel run wrote exactly the artifacts it was asked for,
# the gate demanded one more, and grading was refused without printing why.


def _model_validation_problem_type(state: KaggleState) -> str:
    """Resolve validation semantics, preserving explicit image-to-image domain."""
    domain = str(state.get("domain_detected", "") or "").lower().replace("-", "_")
    if domain == "image_to_image":
        return "image_to_image"
    competition_info = state.get("competition_info")
    declared = (
        getattr(competition_info, "problem_type", "")
        if competition_info is not None
        else ""
    )
    return str(declared or state.get("problem_type", "classification"))


def _expected_class_order_for_state(
    state: KaggleState,
) -> list[str] | None:
    """Resolve probability-column order from public or canonical evidence."""
    submission_order = (
        state.get("submission_contract") or {}
    ).get("class_order")
    canonical_metadata = state.get("canonical_metadata") or {}
    canonical_order = canonical_metadata.get("class_order")
    raw_order = submission_order or canonical_order
    if raw_order is None:
        metadata_path = (
            state.get("canonical_contract") or {}
        ).get("metadata_path")
        if metadata_path and Path(metadata_path).is_file():
            try:
                raw_order = json.loads(
                    Path(metadata_path).read_text(encoding="utf-8")
                ).get("class_order")
            except (OSError, TypeError, ValueError):
                raw_order = None
    if not isinstance(raw_order, (list, tuple)) or len(raw_order) < 2:
        return None
    normalized = [str(value) for value in raw_order]
    if len(normalized) != len(set(normalized)):
        return None
    return normalized


def _validation_class_order_for_state(
    state: KaggleState,
    problem_type: str,
) -> list[str] | None:
    """Resolve order for wide outputs or label-format multiclass outputs."""
    normalized_problem = str(problem_type).lower()
    compact_problem = normalized_problem.replace("_", "").replace("-", "")
    is_classification = any(
        marker in compact_problem
        for marker in ("class", "binary", "multilabel")
    )
    if not is_classification:
        return None
    submission_order = (
        state.get("submission_contract") or {}
    ).get("class_order")
    if isinstance(submission_order, (list, tuple)) and len(submission_order) > 1:
        normalized = [str(value) for value in submission_order]
        return (
            normalized
            if len(normalized) == len(set(normalized))
            else None
        )

    canonical_metadata = state.get("canonical_metadata") or {}
    try:
        canonical_n_classes = int(
            canonical_metadata.get("n_classes") or 0
        )
    except (TypeError, ValueError):
        canonical_n_classes = 0
    if (
        "multiclass" not in compact_problem
        and canonical_n_classes <= 2
    ):
        return None
    return _expected_class_order_for_state(state)


def _requires_class_order_artifact(
    state: KaggleState,
    problem_type: str,
) -> bool:
    """Whether probability columns encode mutually exclusive classes.

    A wide multilabel submission also has more than two prediction columns, but
    those columns are independent labels with no ordering to record, so no
    component is ever asked to save one.
    """
    normalized_problem = str(problem_type).lower()
    compact_problem = normalized_problem.replace("_", "").replace("-", "")
    if not any(
        marker in compact_problem
        for marker in ("class", "binary")
    ):
        return False
    if _validation_class_order_for_state(state, problem_type) is None:
        return False
    return "multilabel" not in compact_problem


def quarantine_component_artifacts(
    models_dir: Path,
    component_name: str,
    *,
    quarantine_dir: Path | None = None,
) -> list[str]:
    """Move a rejected component's artifacts out of the ensemble namespace.

    Rollback only discards the in-memory result; the .npy files a rejected
    component wrote would otherwise be picked up by the ensemble's
    ``oof_*.npy``/``test_*.npy`` glob and averaged into the final submission.
    Rejected evidence is retained in a timestamped audit directory instead of
    being overwritten or permanently deleted.
    """
    safe_component = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in component_name
    )
    artifact_component = (
        component_name
        if Path(component_name).name == component_name
        and component_name not in {".", ".."}
        else safe_component
    )
    destination_root = quarantine_dir or (
        models_dir
        / ".rejected"
        / safe_component
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    moved: list[str] = []
    for prefix in ("oof_", "test_", "test_ids_", "train_ids_", "class_order_"):
        for suffix in (".npy", ".npz"):
            src = models_dir / f"{prefix}{artifact_component}{suffix}"
            if src.exists():
                destination_root.mkdir(parents=True, exist_ok=True)
                src.replace(destination_root / src.name)
                moved.append(src.name)
    return moved


class ValidationMixin:
    """Mixin providing validation capabilities."""

    @staticmethod
    def _is_score_implausible(
        score: float | None, metric_name: str, *, trusted: bool = False
    ) -> bool:
        """Reject non-finite, negative-loss, or out-of-range bounded scores.

        ``trusted`` marks scores independently recomputed from canonical OOF
        artifacts. A stdout-declared 0.0 on a minimization metric is almost
        always a broken validation calc (it once poisoned hill-climbing as an
        unbeatable baseline), so exactly-0.0 losses are only accepted when the
        value was recomputed, never when parsed from generated output.
        """
        if score is None:
            return False
        if not math.isfinite(float(score)):
            return True
        value = float(score)
        if is_metric_minimization(metric_name):
            return value < 0.0 if trusted else value <= 0.0

        metric = (metric_name or "").lower()
        bounded_metrics = (
            "auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "average_precision",
        )
        return any(name in metric for name in bounded_metrics) and not 0.0 <= value <= 1.0

    def _compute_trusted_oof_score(
        self,
        component: AblationComponent,
        state: KaggleState,
    ) -> float | None:
        """Recompute the public CV metric from canonical labels and exact OOF.

        Generated stdout is untrusted. In MLE-bench mode it may be used for
        diagnostics, but it must never decide promotion or rollback.
        """
        working_dir = Path(state["working_directory"])
        canonical_contract = state.get("canonical_contract") or {}
        y_path = Path(
            canonical_contract.get("y_path") or working_dir / "canonical" / "y.npy"
        )
        competition_info = state.get("competition_info")
        problem_type = str(
            getattr(competition_info, "problem_type", "")
            or state.get("problem_type", "")
        ).lower()
        domain = str(state.get("domain_detected", "") or "").lower()
        is_image_to_image = (
            domain == "image_to_image"
            or "image_to_image" in problem_type.replace("-", "_")
        )
        oof_path = (
            working_dir / "models" / f"oof_{component.name}.npz"
            if is_image_to_image
            else working_dir / "models" / f"oof_{component.name}.npy"
        )
        if not oof_path.is_file() or not y_path.is_file():
            print(
                "      Trusted OOF scoring unavailable "
                f"(oof={oof_path.is_file()}, canonical_y={y_path.is_file()})"
            )
            return None

        metric_name = (
            competition_info.evaluation_metric if competition_info is not None else ""
        )
        if is_image_to_image:
            try:
                if "rmse" not in str(metric_name).lower():
                    raise ValueError(
                        "packed image trusted scoring currently requires RMSE"
                    )
                return packed_image_rmse(oof_path, y_path)
            except Exception as exc:
                print(f"      Trusted OOF scoring failed: {exc}")
                return None
        if any(
            token in problem_type
            for token in ("seq2seq", "seq_to_seq", "sequence_to_sequence", "normalization")
        ):
            normalized_problem_type = "seq2seq"
        elif "regression" in problem_type:
            normalized_problem_type = "regression"
        else:
            normalized_problem_type = "classification"

        try:
            print(
                "      [LOG:INFO] Trusted OOF validation started: "
                f"component={component.name}"
            )
            oof = load_npy_readonly(oof_path, allow_pickle=False)
            y_true = load_npy_readonly(y_path, allow_pickle=True)
            if oof.shape[0] != y_true.shape[0]:
                raise ValueError(
                    f"OOF/target row mismatch: {oof.shape[0]} != {y_true.shape[0]}"
                )
            if (
                y_true.ndim == 2
                and y_true.shape[1] > 1
                and oof.shape != y_true.shape
            ):
                raise ValueError(
                    "Multi-output OOF shape must match canonical y exactly: "
                    f"{oof.shape} != {y_true.shape}"
                )
            oof_eligible_mask_path = Path(
                canonical_contract.get("oof_eligible_mask_path")
                or working_dir / "canonical" / "oof_eligible_mask.npy"
            )
            if oof_eligible_mask_path.is_file():
                oof_eligible_mask = np.asarray(
                    load_npy_readonly(
                        oof_eligible_mask_path,
                        allow_pickle=False,
                    ),
                    dtype=bool,
                )
                if oof_eligible_mask.shape != (len(y_true),):
                    raise ValueError(
                        "Canonical OOF eligibility mask is not target-aligned"
                    )
                if normalized_problem_type == "seq2seq":
                    if not np.all(oof_eligible_mask):
                        raise ValueError(
                            "Temporal seq2seq OOF has no supported text "
                            "warm-up sentinel"
                        )
                    oof_for_score = oof
                    y_for_score = y_true
                else:
                    warmup_oof = oof[~oof_eligible_mask]
                    if warmup_oof.size and not np.isnan(
                        np.asarray(warmup_oof, dtype=float)
                    ).all():
                        raise ValueError(
                            "Temporal warm-up OOF rows must remain NaN"
                        )
                    oof_for_score = oof[oof_eligible_mask]
                    y_for_score = y_true[oof_eligible_mask]
            else:
                oof_eligible_mask = None
                oof_for_score = oof
                y_for_score = y_true
            if (
                normalized_problem_type != "seq2seq"
                and not np.all(
                    np.isfinite(np.asarray(oof_for_score, dtype=float))
                )
            ):
                raise ValueError(
                    "OOF predictions contain NaN or Inf on eligible rows"
                )
            # Same directory fallback as y_path above: a contract missing the
            # key must not disqualify IDs that verifiably exist on disk.
            canonical_ids_path = Path(
                canonical_contract.get("train_ids_path")
                or working_dir / "canonical" / "train_ids.npy"
            )
            model_ids_path = (
                working_dir / "models" / f"train_ids_{component.name}.npy"
            )
            if not canonical_ids_path.is_file() or not model_ids_path.is_file():
                raise ValueError("Canonical/model train IDs are unavailable")
            canonical_ids = load_npy_readonly(
                canonical_ids_path,
                allow_pickle=True,
            ).reshape(-1)
            model_ids = load_npy_readonly(
                model_ids_path,
                allow_pickle=False,
            ).reshape(-1)
            if not string_arrays_equal(
                model_ids,
                canonical_ids,
                chunk_rows=TRUSTED_OOF_CHUNK_ROWS,
                progress=_trusted_oof_progress(
                    component.name,
                    "id_alignment",
                ),
            ):
                raise ValueError("OOF train IDs do not match canonical row order")

            # score_predictions encodes targets in sorted-label order, but in
            # mlebench the component's OOF columns are forced to follow the
            # submission-contract class order. Map contract order -> sorted
            # order before scoring, or a non-alphabetical class order (e.g.
            # Type_1..Type_10) is scored on permuted columns.
            class_order_path = (
                working_dir / "models" / f"class_order_{component.name}.npy"
            )
            if (
                oof_for_score.ndim == 2
                and oof_for_score.shape[1] > 1
                and class_order_path.is_file()
            ):
                class_order = np.asarray(
                    np.load(class_order_path, allow_pickle=False)
                ).reshape(-1)
                if class_order.shape[0] != oof_for_score.shape[1]:
                    raise ValueError(
                        "class_order length does not match OOF width"
                    )
                unique_labels = np.unique(y_for_score)
                if unique_labels.shape[0] == oof_for_score.shape[1]:
                    label_position = {
                        str(label): index
                        for index, label in enumerate(unique_labels)
                    }
                    destinations = [
                        label_position.get(str(label)) for label in class_order
                    ]
                    if any(dest is None for dest in destinations):
                        raise ValueError(
                            "class_order labels do not match canonical target"
                            " labels"
                        )
                    if destinations != list(range(len(destinations))):
                        aligned = np.empty_like(oof_for_score)
                        aligned[:, destinations] = oof_for_score
                        oof_for_score = aligned

            if normalized_problem_type == "seq2seq":
                internal_score = score_predictions(
                    oof_for_score,
                    y_for_score,
                    normalized_problem_type,
                    metric_name,
                    row_mask=oof_eligible_mask,
                    chunk_rows=TRUSTED_OOF_CHUNK_ROWS,
                    progress=_trusted_oof_progress(
                        component.name,
                        "seq2seq_score",
                    ),
                )
            else:
                internal_score = score_predictions(
                    oof_for_score,
                    y_for_score,
                    normalized_problem_type,
                    metric_name,
                )
            score = (
                float(internal_score)
                if is_metric_minimization(metric_name)
                else -float(internal_score)
            )
            if not math.isfinite(score) or self._is_score_implausible(
                score, metric_name, trusted=True
            ):
                raise ValueError(f"implausible recomputed score: {score}")
            print(
                "      [LOG:INFO] Trusted OOF validation complete: "
                f"component={component.name} score={score:.6f}"
            )
            return score
        except Exception as exc:
            print(f"      Trusted OOF scoring failed: {exc}")
            return None

    def _infer_metric_from_stdout(self, stdout: str) -> str | None:
        """Infer metric name from stdout patterns.

        Returns:
            Inferred metric name, or None if no pattern matched.
        """
        stdout_lower = stdout.lower()
        metric_patterns = [
            (r"(?:roc[_\s-]?auc|auroc)\s*[:=]", "auc"),
            (r"(?:log[_\s-]?loss|logloss)\s*[:=]", "log_loss"),
            (r"(?:accuracy)\s*[:=]", "accuracy"),
            (r"(?:rmse)\s*[:=]", "rmse"),
            (r"(?:mae)\s*[:=]", "mae"),
            (r"(?:f1[_\s-]?score|f1)\s*[:=]", "f1"),
        ]
        for pattern, metric in metric_patterns:
            if re.search(pattern, stdout_lower):
                return metric
        return None

    def _extract_cv_score(self, stdout: str) -> float | None:
        """
        Extract cross-validation score from stdout using regex patterns.

        Args:
            stdout: Standard output from code execution

        Returns:
            Extracted CV score, or None if not found
        """
        # Try multiple patterns to extract CV score
        number = r"([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf)"
        patterns = [
            rf"CV Score.*?{number}",
            rf"Final Validation Performance:\s*{number}",
            rf"ROC-AUC.*?{number}",
            rf"Accuracy.*?{number}",
            rf"RMSE.*?{number}",
            rf"Mean.*?{number}\s*\(",  # Mean score with std
        ]

        for pattern in patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if math.isnan(value) or math.isinf(value):
                        return None
                    return value
                except ValueError:
                    continue

        return None

    def _validate_component_improvement(
        self,
        component: AblationComponent,
        exec_result: "ExecutionResult",
        state: KaggleState,
    ) -> tuple[bool, float | None]:
        """
        Validate if component improves score using Hill Climbing strategy.

        Implements ablation studies by comparing CV score before and after component.

        Args:
            component: Component being tested
            exec_result: Execution result containing stdout
            state: Current workflow state

        Returns:
            (should_keep, new_score) - Whether to keep component and its CV score
        """
        competition_info = state.get("competition_info")
        metric_name = competition_info.evaluation_metric if competition_info else ""
        run_mode = str(state.get("run_mode", "")).lower()

        # Generated stdout is an untrusted artifact in MLE-bench. Inferring the
        # metric from it would let candidate code choose the objective used to
        # promote itself, so benchmark runs must abstain until the public metric
        # contract is available.
        metric_unknown = not metric_name or metric_name.lower() in ("unknown", "none", "")
        inferred_metric = None
        if metric_unknown and run_mode != "mlebench":
            inferred_metric = self._infer_metric_from_stdout(exec_result.stdout)
            if inferred_metric:
                print(f"\n   🔍 Inferred metric from stdout: {inferred_metric}")
                metric_name = inferred_metric

        is_minimize = is_metric_minimization(metric_name)

        declared_score = self._extract_cv_score(exec_result.stdout)
        if run_mode == "mlebench":
            if metric_unknown:
                print(
                    "      Trusted OOF scoring unavailable: public metric "
                    "contract is missing"
                )
                cv_score = None
            else:
                cv_score = self._compute_trusted_oof_score(component, state)
            if declared_score is not None and cv_score is not None:
                print(
                    "      Declared CV score "
                    f"{declared_score:.6f}; independently recomputed {cv_score:.6f}"
                )
        else:
            cv_score = declared_score

        if cv_score is None:
            print("\n   📊 Ablation Study (Hill Climbing):")
            print(
                f"      Metric:         {metric_name} ({'↓' if is_minimize else '↑'} {'minimize' if is_minimize else 'maximize'})"
            )
            if run_mode == "mlebench":
                canonical_contract = state.get("canonical_contract") or {}
                canonical_y = Path(
                    canonical_contract.get("y_path")
                    or Path(state["working_directory"]) / "canonical" / "y.npy"
                )
                if not canonical_y.is_file():
                    # Domains without a canonical contract (image without
                    # train.csv, audio without label tables) cannot produce a
                    # trusted score by construction. Keep the candidate
                    # unscored/unpromoted so the deterministic unscored
                    # fallback can preserve a gradable artifact.
                    print(
                        "      Canonical contract unavailable for this domain; "
                        "keeping the candidate without promotion (unscored)."
                    )
                    return True, None
                print(
                    "      No independently reproducible OOF score; candidate "
                    "is not approved for MLE-bench promotion."
                )
                return False, None
            print(
                "      No CV score found in stdout; skipping rollback and "
                "keeping component."
            )
            return True, None

        if self._is_score_implausible(
            cv_score, metric_name, trusted=run_mode == "mlebench"
        ):
            print("\n   📊 Ablation Study (Hill Climbing):")
            print(f"      Component CV:   {cv_score:.6f} ({metric_name}, lower is better)")
            print("      ❌ Component REJECTED (implausible score; validation calc likely broken)")
            return False, None

        baseline_score = state.get("baseline_cv_score")
        if baseline_score is None:
            baseline_score = float("inf") if is_minimize else float("-inf")

        # Detect metric mismatch: scores look like different metrics
        # (e.g., baseline=0.95 AUC vs component=0.27 LogLoss)
        scores_look_mismatched = False
        if baseline_score not in (float("inf"), float("-inf")):
            one_high = max(cv_score, baseline_score) > 0.5
            one_low = min(cv_score, baseline_score) < 0.5
            large_gap = abs(cv_score - baseline_score) > 0.3
            scores_look_mismatched = one_high and one_low and large_gap

        if scores_look_mismatched and metric_unknown and not inferred_metric:
            print("\n   📊 Ablation Study (Hill Climbing):")
            print(f"      Metric:         {metric_name} (unknown)")
            print(f"      Baseline CV:    {baseline_score:.4f}")
            print(f"      Component CV:   {cv_score:.4f}")
            print("      ⚠️  Scores appear to use different metrics (likely mismatch)")
            print("      ✅ Component ACCEPTED (metric mismatch detected, keeping by default)")
            return True, None  # Return None to NOT update baseline with mismatched score

        improvement = calculate_score_improvement(cv_score, baseline_score, metric_name)
        direction_symbol = "↓" if is_minimize else "↑"
        direction_text = "minimize" if is_minimize else "maximize"

        print("\n   📊 Ablation Study (Hill Climbing):")
        print(f"      Metric:         {metric_name} ({direction_symbol} {direction_text})")
        print(f"      Baseline CV:    {baseline_score:.4f}")
        print(f"      Component CV:   {cv_score:.4f}")
        print(f"      Improvement:    {improvement:+.4f}")

        min_improvement = 0.001
        should_keep = improvement >= min_improvement

        if not should_keep:
            print("      ❌ Component REJECTED (no improvement or negative impact)")
            print(f"      Reason: Delta ({improvement:+.4f}) < threshold ({min_improvement})")
        else:
            print("      ✅ Component ACCEPTED (positive improvement)")
            if baseline_score not in [float("inf"), float("-inf"), 0]:
                relative_gain = abs(improvement / baseline_score * 100)
                print(f"      Impact: {relative_gain:.2f}% relative improvement")

        return should_keep, cv_score
