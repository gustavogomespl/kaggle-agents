"""
Developer Agent with Code Generation and Auto-Retry.

This agent generates Python code to implement ablation components,
with automatic retry and debugging capabilities.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from ...core.config import (
    calculate_score_improvement,
    get_config,
    get_llm_for_role,
    is_metric_minimization,
    metric_reads_rows_as_distribution,
)
from ...core.state import (
    AblationComponent,
    CodeAttempt,
    DevelopmentResult,
    KaggleState,
    ReasoningTrace,
    SelfEvaluation,
)
from ...optimization import create_optimizer, create_preference_collector
from ...prompts.templates.developer_prompts import (
    DEVELOPER_CORE_IDENTITY,
    HARD_CONSTRAINTS,
)
from ...tools.code_executor import ArtifactValidator, CodeExecutor, ExecutionResult
from ...utils.bounded_array import load_npy_readonly
from ...utils.llm_utils import get_text_content
from ...utils.log_parser import format_feedback_for_llm, parse_training_logs
from ...utils.run_budget import (
    FINALIZATION_RESERVE_S,
    budget_exhausted,
    clamp_timeout_to_budget,
    format_remaining,
)
from ...utils.telemetry import make_event
from .code_contracts import (
    ARTIFACT_HELPER as _ARTIFACT_HELPER,
)
from .code_contracts import (
    HELPER_IMPORT_CONTRACT_ERROR,
    MISSING_CLASS_ORDER_ERROR,
    MISSING_SUBMISSION_HELPER_ERROR,
    SUBMISSION_CONTRACT_ERROR,
    handwritten_submission_write,
    missing_class_order_helper_argument,
    missing_submission_helper_call,
    requires_submission_helper,
    untrusted_contract_helper_import,
)
from .code_contracts import (
    SUBMISSION_HELPER as _SUBMISSION_HELPER,
)

# Re-export temperature utilities for backward compatibility
from .code_generator import (
    CodeGeneratorMixin,
)
from .dspy_modules import CodeFixerModule, CodeGeneratorModule
from .execution_failures import (
    GeneratedContractStructureError,
    RepeatedInjectedContractError,
    execute_generated_candidate,
    execution_failure_to_development_result,
)
from .grpo import GRPOMixin
from .quiet_star import QuietStarMixin
from .refinement import REFINEMENT_TRUST_BOUNDARY, RefinementMixin
from .retry import (
    RetryMixin,
    preserve_injected_header,
    require_oof_artifacts,
)
from .target_source import CanonicalTargetContractError
from .utils import DeveloperUtilsMixin
from .validation import (
    ValidationMixin,
    _model_validation_problem_type,
    _requires_class_order_artifact,
    _validation_class_order_for_state,
    quarantine_component_artifacts,
)


def _uses_packed_image_artifacts(working_dir: Path) -> bool:
    """Return whether the canonical metadata selects the packed image contract."""
    metadata_path = Path(working_dir) / "canonical" / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        metadata.get("packed_image_contract")
        and str(metadata.get("task_type")) == "image_to_image"
    )


def _component_prediction_artifact(
    working_dir: Path,
    component_name: str,
    kind: str,
) -> Path:
    """Resolve a component OOF/test path from the canonical artifact family."""
    models_dir = Path(working_dir) / "models"
    packed = models_dir / f"{kind}_{component_name}.npz"
    dense = models_dir / f"{kind}_{component_name}.npy"
    if _uses_packed_image_artifacts(working_dir) or (
        packed.is_file() and not dense.is_file()
    ):
        return packed
    return dense


def _oof_artifact_digest(working_dir: Path, component_name: str) -> str | None:
    """Content digest of a component's OOF file, or None when it is absent.

    Used to tell "this program produced new evidence" apart from "this program
    left the previous program's evidence in place", which the trusted scorer
    cannot distinguish on its own.
    """
    path = _component_prediction_artifact(working_dir, component_name, "oof")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def unsaved_expected_artifacts(
    code: str,
    expected_artifacts: list[str] | None,
    component_name: str,
) -> list[str]:
    """Expected artifacts with no matching save call anywhere in the code.

    The artifact contract is only enforced *after* execution, so a program that
    trains correctly for half an hour and then saves just ``oof_<name>.npy`` is
    failed and regenerated from scratch -- paying the full training cost twice.
    Both models in a smoke run did exactly that, burning roughly half the GPU
    time re-running work that had already succeeded.

    A save is matched either literally (``"oof_my_model.npy"``) or composed
    (``f"oof_{COMPONENT_NAME}.npy"``), which is how generated code writes it.
    The reading is deliberately conservative: an unparseable program, an
    unusual save expression, or no save calls at all yields no findings, so
    this can delay a run but never block a correct one.
    """
    if not expected_artifacts or not component_name:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    # The injected helper is defined in every model script, so the saves inside
    # its body prove nothing about the program that was generated. Exclude that
    # subtree, then treat a call to it as satisfying the whole contract.
    helper_body_nodes = {
        id(sub)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == _ARTIFACT_HELPER
        for sub in ast.walk(node)
    }

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == _ARTIFACT_HELPER
            and id(node) not in helper_body_nodes
        ):
            return []

    saved_targets: list[str] = []
    for node in ast.walk(tree):
        if id(node) in helper_body_nodes:
            continue
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        call_name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if call_name not in {"save", "savez", "savez_compressed", "tofile"} or not node.args:
            continue

        # The destination is not always the first argument: np.save(path, arr)
        # puts it first, torch.save(obj, path) puts it second. Scan every
        # argument and keep the ones that carry a string, rather than assuming
        # a position -- assuming arg 0 made this check silently no-op on any
        # component using torch.save, which is nearly all of them.
        string_args = [
            arg
            for arg in node.args
            if any(
                isinstance(sub, ast.JoinedStr)
                or (isinstance(sub, ast.Constant) and isinstance(sub.value, str))
                for sub in ast.walk(arg)
            )
        ]
        # A destination assembled entirely through variables cannot be read
        # statically. One opaque save makes the whole picture inconclusive, so
        # report nothing rather than risk blocking a correct program.
        if not string_args:
            return []
        try:
            saved_targets.extend(ast.unparse(arg) for arg in string_args)
        except Exception:
            return []

    if not saved_targets:
        # Only model components have expected artifacts, and the scan walks the
        # whole tree -- including the script's own wrapper functions -- for
        # np.save/torch.save/savez/tofile. Zero matches means nothing is being
        # persisted at all, which is a finding rather than an inconclusive read.
        return list(expected_artifacts)

    haystack = " ".join(saved_targets)
    missing = []
    for artifact in expected_artifacts:
        stem = Path(artifact).stem  # "test_ids_<component_name>"
        if not stem.endswith(component_name):
            continue
        kind = stem[: -len(component_name)]  # "test_ids_"
        if kind and f"{kind}{component_name}" not in haystack and f"{kind}{{" not in haystack:
            missing.append(artifact)
    return missing


def _expected_model_artifacts(
    component: AblationComponent,
    working_dir: Path,
    run_mode: str = "",
) -> list[str] | None:
    """Artifacts the executor must see before a model run counts as success.

    This mirrors the strict post-acceptance validation: mlebench requires
    train_ids and test_ids unconditionally, and trusted-OOF promotion cannot
    verify row alignment without train_ids. Enforcing the same files at
    execution keeps the failure inside the fix/retry loop instead of
    surfacing as a silent post-training rollback.
    """
    if component.component_type != "model" or not require_oof_artifacts():
        return None
    if _uses_packed_image_artifacts(working_dir):
        return [
            f"models/oof_{component.name}.npz",
            f"models/test_{component.name}.npz",
        ]
    expected = [
        f"models/oof_{component.name}.npy",
        f"models/test_{component.name}.npy",
    ]
    mlebench = str(run_mode).strip().lower() == "mlebench"
    if mlebench or (working_dir / "canonical" / "metadata.json").is_file():
        expected.append(f"models/train_ids_{component.name}.npy")
    if mlebench:
        expected.append(f"models/test_ids_{component.name}.npy")
    return expected


def _resolved_primary_score(
    result: Any,
    component_name: str,
    state: Mapping[str, Any],
    new_cv_score: float | None,
) -> float | None:
    """Trusted score used for promotion decisions, surviving cache reuse.

    A reused result deliberately skips re-scoring (its improvement gate ran
    when it first executed), so ``new_cv_score`` is ``None`` here. Reading
    that ``None`` as "no independently reproducible OOF score" rejected the
    reused component and quarantined the prior best model's artifacts on
    every refinement iteration. The score it earned is still in the trusted
    map, keyed by the same host-recomputed provenance — only cache reuse may
    substitute it; a fresh unscored result stays unscored and fail-closed.
    """
    try:
        primary = float(new_cv_score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        primary = None
    if primary is not None and not math.isfinite(primary):
        primary = None
    if primary is not None or not getattr(result, "reused_from_cache", False):
        return primary
    reused = (state.get("trusted_component_scores") or {}).get(component_name)
    if isinstance(reused, dict):
        reused = reused.get("score", reused.get("cv_score"))
    try:
        reused_score = float(reused)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return reused_score if math.isfinite(reused_score) else None


def _has_combinable_model_predictions(
    state: KaggleState,
    working_dir: Path,
) -> bool:
    """Whether any accepted model predictions exist for an ensemble to use.

    Rejected candidates are quarantined under ``rejected_*`` names, so a plain
    ``oof_*.npy`` glob only sees artifacts that were never rolled back.
    """
    domain = str(state.get("domain_detected", "") or "").lower().replace("-", "_")
    if domain == "image_to_image":
        # The current ensemble agent consumes fixed-shape .npy matrices.
        # Feeding it variable-size packed images would either crash or silently
        # lose ID/shape alignment, so keep packed blending disabled until it has
        # a dedicated ID-aligned implementation.
        return False
    if any(bool(flag) for flag in (state.get("oof_availability") or {}).values()):
        return True
    models_dir = working_dir / "models"
    return models_dir.is_dir() and (
        any(models_dir.glob("oof_*.npy"))
        or any(models_dir.glob("oof_*.npz"))
    )


class DeveloperAgent(
    GRPOMixin,
    QuietStarMixin,
    ValidationMixin,
    RetryMixin,
    DeveloperUtilsMixin,
    CodeGeneratorMixin,
    RefinementMixin,
):
    """
    Agent responsible for code generation and execution.

    Features:
    - Generate code from ablation components
    - Execute code in sandbox
    - Automatic retry on failure (5 attempts)
    - Debug iterations (10 max)
    - Artifact validation
    - DSPy optimization support
    - GRPO reasoning traces
    - Quiet-STaR self-evaluation
    - DPO preference collection
    """

    def __init__(self, use_dspy: bool = True):
        """
        Initialize the developer agent.

        Args:
            use_dspy: Whether to use DSPy modules
        """
        self.config = get_config()
        self.use_dspy = use_dspy and self.config.dspy.enabled

        timeout = self.config.ablation.testing_timeout
        self.executor = CodeExecutor(timeout=timeout)
        self.validator = ArtifactValidator()

        print(f"Component timeout set to: {timeout}s ({timeout / 60:.1f} min)")

        implementation_temperature = 0.1

        self.llm = get_llm_for_role(
            role="developer",
            temperature=implementation_temperature,
            max_tokens=self.config.llm.max_tokens,
        )

        if self.use_dspy:
            optimizer = create_optimizer()
            self.generator_module = optimizer.load_optimized_prompt("developer_generator")
            self.fixer_module = optimizer.load_optimized_prompt("developer_fixer")

            if self.generator_module is None:
                print("Using base (unoptimized) generator module")
                self.generator_module = CodeGeneratorModule()

            if self.fixer_module is None:
                print("Using base (unoptimized) fixer module")
                self.fixer_module = CodeFixerModule()

        # GRPO: Store last reasoning trace for state persistence
        self._last_reasoning_trace: ReasoningTrace | None = None

        # The resolved target decision for the component being generated, kept
        # here only until it is drained into the state updates below.
        self._last_target_source = None
        self._last_target_source_metadata: dict | None = None

        # DPO: Preference collector for learning from code fixes
        self._preference_collector = create_preference_collector()

        # Quiet-STaR: Store last self-evaluation for state persistence
        self._last_self_evaluation: SelfEvaluation | None = None

    @staticmethod
    def _recover_missing_submission(
        *,
        run_mode: str,
        submission_path: Path | None,
        working_dir: Path,
        component_name: str,
        sample_submission_path: Path,
        target_cols: list[str],
        id_col: str | None,
        test_ids_are_positional: bool,
    ) -> Path | None:
        """Rebuild a missing MLE-bench CSV from validated model evidence."""
        if submission_path is not None or run_mode != "mlebench":
            return submission_path
        from ...utils.submission_artifacts import (
            rebuild_submission_from_component_predictions,
        )

        return rebuild_submission_from_component_predictions(
            working_dir=working_dir,
            component_name=component_name,
            sample_submission_path=sample_submission_path,
            target_cols=target_cols,
            id_col=id_col,
            test_ids_are_positional=test_ids_are_positional,
        )

    def _write_execution_logs_and_manifest(
        self,
        component: AblationComponent,
        exec_result: ExecutionResult,
        working_dir: Path,
        attempt: int,
        expected_artifacts: list[str] | None,
    ) -> tuple[Path | None, Path | None]:
        logs_dir = working_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        safe_component = "".join(
            c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in component.name
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attempt_id = attempt + 1
        log_path = logs_dir / f"{safe_component}_attempt{attempt_id}_{timestamp}.log"
        manifest_path = logs_dir / f"{safe_component}_attempt{attempt_id}_{timestamp}.json"

        expected = expected_artifacts or []
        missing_expected = [
            rel for rel in expected if not (working_dir / rel).exists()
        ]

        models_dir = working_dir / "models"
        model_files: list[str] = []
        if models_dir.exists():
            for ext in (".pth", ".pt", ".keras", ".h5", ".joblib", ".pkl"):
                for p in models_dir.glob(f"*{ext}"):
                    model_files.append(str(p.relative_to(working_dir)))

        manifest = {
            "component": component.name,
            "component_type": component.component_type,
            "attempt": attempt_id,
            "success": exec_result.success,
            # The contract this attempt actually ran against, and how the
            # failure was classified. Without these the log cannot distinguish
            # an invalid harness attempt from a candidate defect.
            "failure_origin": exec_result.failure_origin,
            "retryable": exec_result.retryable,
            "header_sha256": exec_result.header_sha256,
            "contract_fingerprint": exec_result.contract_fingerprint,
            "execution_time_s": exec_result.execution_time,
            "exit_code": exec_result.exit_code,
            "expected_artifacts": expected,
            "missing_expected_artifacts": missing_expected,
            "artifacts_created": exec_result.artifacts_created,
            # This value is diagnostic only. Promotion in MLE-bench uses an
            # independently recomputed metric from canonical labels and OOF.
            "declared_cv_score": self._extract_cv_score(exec_result.stdout),
            "submission_exists": (working_dir / "submission.csv").exists(),
            "oof_exists": _component_prediction_artifact(
                working_dir, component.name, "oof"
            ).exists(),
            "test_preds_exists": _component_prediction_artifact(
                working_dir, component.name, "test"
            ).exists(),
            "model_files": sorted(model_files),
            "log_path": str(log_path),
        }

        try:
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write(f"component={component.name}\n")
                handle.write(f"component_type={component.component_type}\n")
                handle.write(f"attempt={attempt_id}\n")
                handle.write(f"success={exec_result.success}\n")
                handle.write(f"execution_time_s={exec_result.execution_time:.2f}\n")
                handle.write(f"exit_code={exec_result.exit_code}\n")
                handle.write("\n[STDOUT]\n")
                handle.write(exec_result.stdout or "")
                handle.write("\n\n[STDERR]\n")
                handle.write(exec_result.stderr or "")
        except Exception as exc:
            print(f"⚠️ Failed to write execution log: {exc}")
            log_path = None

        try:
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=True)
        except Exception as exc:
            print(f"⚠️ Failed to write execution manifest: {exc}")
            manifest_path = None

        if log_path:
            exec_result.artifacts_created.append(str(log_path.relative_to(working_dir)))
        if manifest_path:
            exec_result.artifacts_created.append(
                str(manifest_path.relative_to(working_dir))
            )

        return log_path, manifest_path

    def _reject_model_candidate(
        self,
        *,
        state: KaggleState,
        component: AblationComponent,
        working_dir: Path,
        current_index: int,
        attempt_records: list[CodeAttempt],
        reason: str,
        retry_invalid: bool,
    ) -> dict[str, Any]:
        """Quarantine a rejected candidate and restore only a verified snapshot."""
        from ...utils.submission_artifacts import (
            restore_accepted_submission,
            restore_best_candidate_submission,
        )

        rejected_dir = working_dir / ".rejected_submissions"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        current_submission = working_dir / "submission.csv"
        safe_component = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in component.name
        )
        artifact_component = (
            component.name
            if Path(component.name).name == component.name
            and component.name not in {".", ".."}
            else safe_component
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        candidate_submissions = (
            (current_submission, "current"),
            (
                working_dir / f"submission_{artifact_component}.csv",
                "component",
            ),
        )
        for candidate_submission, label in candidate_submissions:
            if not candidate_submission.is_file():
                continue
            candidate_submission.replace(
                rejected_dir
                / f"{safe_component}_{label}_{timestamp}.csv"
            )

        snapshot_owner = str(
            state.get("best_candidate_submission_component_name") or ""
        )
        # A trusted score is deliberately NOT required here. Domains where no
        # canonical contract could be prepared preserve their first structurally
        # valid candidate as an explicitly unscored fallback, so demanding a
        # finite score made that fallback impossible to restore: rejecting any
        # later candidate deleted the live submission.csv and left the run with
        # nothing to grade while the verified snapshot sat unused on disk.
        # What makes the snapshot safe to restore is not a score but its
        # provenance: restore_best_candidate_submission re-verifies the SHA-256
        # digest and confines the path to this run's snapshot store.
        best_snapshot_still_eligible = (
            bool(snapshot_owner) and snapshot_owner != component.name
        )

        restored = (
            restore_best_candidate_submission(state, working_dir)
            if best_snapshot_still_eligible
            else None
        )
        if restored is None:
            restored = restore_accepted_submission(state, working_dir)
        if restored is not None:
            print(f"   Restored verified accepted submission: {restored}")
        else:
            print("   No verified accepted submission exists; submission.csv remains absent")

        quarantined = quarantine_component_artifacts(
            working_dir / "models", component.name
        )
        if quarantined:
            print(f"   Quarantined rejected artifacts: {', '.join(quarantined)}")

        oof_availability = dict(state.get("oof_availability") or {})
        oof_availability[component.name] = False
        robustness_approvals = dict(
            state.get("robustness_approved_components") or {}
        )
        robustness_approvals[component.name] = False
        component_results = dict(state.get("component_results") or {})
        component_results.pop(component.name, None)
        trusted_scores = dict(state.get("trusted_component_scores") or {})
        trusted_scores.pop(component.name, None)

        try:
            retry_count = max(0, int(state.get("code_retry_count") or 0))
        except (TypeError, ValueError):
            retry_count = 0
        try:
            configured_retries = int(
                os.getenv("KAGGLE_AGENTS_MAX_COMPONENT_RETRIES", "3")
            )
        except ValueError:
            configured_retries = 3
        max_retries = max(1, configured_retries)
        retry_count = retry_count + 1 if retry_invalid else max_retries
        exhausted = retry_count >= max_retries

        updates: dict[str, Any] = {
            "development_results": [],
            "current_component_index": (
                current_index + 1 if exhausted else current_index
            ),
            "component_rollback": component.name,
            "rollback_reason": reason,
            "code_retry_count": 0 if exhausted else retry_count,
            "code_attempts": attempt_records,
            "oof_availability": oof_availability,
            "robustness_approved_components": robustness_approvals,
            "component_results": component_results,
            "trusted_component_scores": trusted_scores,
            "last_updated": datetime.now(),
        }
        if not best_snapshot_still_eligible:
            updates.update(
                {
                    "best_candidate_submission_snapshot_path": None,
                    "best_candidate_submission_sha256": None,
                    "best_candidate_submission_component_name": None,
                }
            )
        if str(state.get("best_single_model_name") or "") == component.name:
            updates.update(
                {
                    "best_single_model_name": None,
                    "best_single_model_score": None,
                    "baseline_cv_score": None,
                    # Declared `float`, not `float | None`. Writing None here
                    # crashed the next prompt build, because readers use
                    # `state.get(key, 0.0)` and a default does not apply to a
                    # key that exists holding None. 0.0 is the existing
                    # "no measured score" sentinel for this field.
                    "current_performance_score": 0.0,
                }
            )
        if exhausted:
            existing_failed = set(state.get("failed_component_names") or [])
            if component.name not in existing_failed:
                updates["failed_component_names"] = [component.name]
        return updates

    def _record_injected_contract_failure(
        self,
        *,
        state: KaggleState,
        component: AblationComponent,
        working_dir: Path,
        current_index: int,
        attempt_records: list[CodeAttempt],
        result: DevelopmentResult,
    ) -> dict[str, Any]:
        """Close a non-retryable contract failure without blaming the model.

        This is deliberately not ``_reject_model_candidate``: nothing here is
        evidence about the component. The candidate never reached its own body
        (or ran against inputs that had already changed), so there is no model
        to roll back, no score to revoke and no reason to spend another
        component retry. What must survive is everything the run had already
        earned - accepted component results, trusted scores, and above all the
        accepted-registry keys the single final grading pass reads.
        """
        from ...utils.submission_artifacts import restore_accepted_submission

        origin = str(getattr(result, "failure_origin", "") or "harness")
        fingerprint = str(getattr(result, "contract_fingerprint", "") or "")
        reason = (
            "injected_header_failure"
            if origin == "harness"
            else "protected_input_contract_failure"
        )
        detail_text = next(
            (
                str(value).strip()
                for value in (result.errors or [])
                if str(value).strip()
            ),
            (result.stderr or "").strip(),
        )
        print(
            f"\n🛑 Non-retryable {origin} failure in '{component.name}': "
            f"{reason}. Neither retry level may spend budget on it."
        )

        # Quarantine only what this unverified attempt actually produced. A
        # preamble failure creates nothing, so an unconditional quarantine here
        # would destroy evidence this same component had accepted earlier.
        created = {str(path) for path in (result.artifacts_created or [])}
        component_artifacts_written = any(
            Path(path).name.endswith(f"_{component.name}.npy")
            or Path(path).name.endswith(f"_{component.name}.npz")
            for path in created
        )
        quarantined: list[str] = []
        if component_artifacts_written:
            quarantined = quarantine_component_artifacts(
                working_dir / "models", component.name
            )
            if quarantined:
                print(f"   Quarantined unverified artifacts: {', '.join(quarantined)}")

        candidate_submission = working_dir / f"submission_{component.name}.csv"
        if candidate_submission.is_file():
            rejected_dir = working_dir / ".rejected_submissions"
            rejected_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_component = "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in component.name
            )
            candidate_submission.replace(
                rejected_dir / f"{safe_component}_component_{timestamp}.csv"
            )
        restored = restore_accepted_submission(state, working_dir)
        if restored is not None:
            print(f"   Restored verified accepted submission: {restored}")

        failed_contracts = dict(state.get("failed_contract_fingerprints") or {})
        terminal_detail: dict[str, Any] = {
            "reason": reason,
            "origin": origin,
            "component": component.name,
            "component_type": component.component_type,
            "contract_fingerprint": fingerprint,
            "header_sha256": str(getattr(result, "header_sha256", "") or ""),
            "error": detail_text[:1000],
        }
        if fingerprint:
            failed_contracts[fingerprint] = {
                "component": component.name,
                "component_type": component.component_type,
                "origin": origin,
                "reason": reason,
                "header_sha256": terminal_detail["header_sha256"],
                "error": detail_text[:500],
            }

        return {
            "development_results": [],
            "code_attempts": attempt_records,
            # Advance: a genuinely different contract may still be diagnosed,
            # so this does NOT set skip_remaining_components.
            "current_component_index": current_index + 1,
            "code_retry_count": 0,
            "failed_contract_fingerprints": failed_contracts,
            "workflow_valid": False,
            "terminal_failure_origin": origin,
            "terminal_failure_detail": terminal_detail,
            "telemetry_events": [
                make_event(
                    "harness",
                    "injected_header_failure",
                    iteration=state.get("current_iteration", 0),
                    component_name=component.name,
                    component_type=component.component_type,
                    origin=origin,
                    reason=reason,
                    contract_fingerprint=fingerprint,
                    header_sha256=terminal_detail["header_sha256"],
                    quarantined=quarantined,
                )
            ],
            "last_updated": datetime.now(),
        }

    def _skip_duplicate_injected_contract(
        self,
        *,
        state: KaggleState,
        component: AblationComponent,
        current_index: int,
        error: RepeatedInjectedContractError,
    ) -> dict[str, Any]:
        """Advance past a contract that already failed, at zero cost.

        No LLM generation, no execution, no fixer, no debugger: an identical
        normalized contract can only reproduce the failure that was already
        diagnosed, and the terminal origin recorded then still describes it.
        """
        print(
            f"\n⏭️  Skipping '{component.name}': its generated contract already "
            f"failed in this run ({error.contract_fingerprint[:12]})"
        )
        updates: dict[str, Any] = {
            "current_component_index": current_index + 1,
            "code_retry_count": 0,
            "workflow_valid": False,
            "telemetry_events": [
                make_event(
                    "harness",
                    "duplicate_injected_contract_skipped",
                    iteration=state.get("current_iteration", 0),
                    component_name=component.name,
                    component_type=component.component_type,
                    contract_fingerprint=error.contract_fingerprint,
                )
            ],
            "last_updated": datetime.now(),
        }
        # Preserve, never re-derive: the origin and detail belong to the
        # attempt that actually failed.
        existing_origin = state.get("terminal_failure_origin")
        if existing_origin:
            updates["terminal_failure_origin"] = str(existing_origin)
        existing_detail = state.get("terminal_failure_detail")
        if isinstance(existing_detail, dict):
            updates["terminal_failure_detail"] = dict(existing_detail)
        return updates

    def _record_pregeneration_contract_failure(
        self,
        *,
        state: KaggleState,
        component: AblationComponent,
        current_index: int,
        error: Exception,
    ) -> dict[str, Any]:
        """Stop the run on a contract that cannot be rendered at all.

        A corrupt canonical claim or a malformed generator-owned header is not
        a property of one component: every remaining component would render the
        same broken contract. Zero LLM, executor, fixer and debugger calls have
        happened at this point, and no component or model is marked as failed.
        """
        violations = [
            dict(violation)
            for violation in getattr(error, "violations", []) or []
            if isinstance(violation, Mapping)
        ]
        reason = (
            "canonical_target_contract_error"
            if isinstance(error, CanonicalTargetContractError)
            else "generated_contract_structure_error"
        )
        print(
            f"\n🛑 {reason} while preparing '{component.name}': {error}. "
            "No candidate can be generated against this contract."
        )
        terminal_detail: dict[str, Any] = {
            "reason": reason,
            "origin": "harness",
            "component": component.name,
            "component_type": component.component_type,
            "error": str(error)[:1000],
            "violations": violations,
        }
        plan_length = len(state.get("ablation_plan", []) or [])
        return {
            "development_results": [],
            "current_component_index": max(plan_length, current_index + 1),
            "code_retry_count": 0,
            "skip_remaining_components": True,
            "workflow_valid": False,
            "terminal_failure_origin": "harness",
            "terminal_failure_detail": terminal_detail,
            "telemetry_events": [
                make_event(
                    "harness",
                    "generated_contract_unavailable",
                    iteration=state.get("current_iteration", 0),
                    component_name=component.name,
                    component_type=component.component_type,
                    reason=reason,
                    violations=violations,
                )
            ],
            "last_updated": datetime.now(),
        }

    @staticmethod
    def _artifact_sha256(path: Path) -> str:
        """Hash one artifact without loading potentially large arrays in memory."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _snapshot_approved_component_artifacts(
        cls,
        state: KaggleState,
        working_dir: Path,
        *,
        active_component_name: str,
    ) -> dict[str, Any] | None:
        """Snapshot every other trusted component before generated code runs.

        Generated code for component B executes in the shared models directory
        and can accidentally overwrite component A's artifacts. The snapshot is
        kept outside the generated-code workspace and records both present and
        absent canonical component paths, so cross-component creation is also
        detected. Robustness approval happens only after the complete developer
        loop, so ``robustness_approved_components`` alone is too late to protect
        model A while model B is being implemented.
        """
        approvals = state.get("robustness_approved_components") or {}
        availability = state.get("oof_availability") or {}
        trusted_scores = state.get("trusted_component_scores") or {}

        protected_names_set: set[str] = set()
        if isinstance(approvals, dict):
            protected_names_set.update(
                str(name)
                for name, approved in approvals.items()
                if approved is True and str(name)
            )
        if isinstance(availability, dict):
            protected_names_set.update(
                str(name)
                for name, available in availability.items()
                if available is True and str(name)
            )
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
                if math.isfinite(score) and str(name):
                    protected_names_set.add(str(name))

        protected_names = sorted(
            name
            for name in protected_names_set
            if name != active_component_name
        )
        if not protected_names:
            return None

        snapshot_root = Path(
            tempfile.mkdtemp(prefix="kaggle_agents_approved_artifacts_")
        )
        records: list[dict[str, Any]] = []
        models_dir = working_dir / "models"
        prefixes = (
            "oof_",
            "test_",
            "test_ids_",
            "train_ids_",
            "class_order_",
        )
        for component_name in protected_names:
            if (
                Path(component_name).name != component_name
                or component_name in {".", ".."}
            ):
                continue
            safe_component = "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in component_name
            )
            for prefix in prefixes:
                suffixes = (
                    (".npy", ".npz")
                    if prefix in {"oof_", "test_"}
                    else (".npy",)
                )
                for suffix in suffixes:
                    path = models_dir / f"{prefix}{component_name}{suffix}"
                    exists = path.is_file()
                    snapshot_path = (
                        snapshot_root
                        / safe_component
                        / f"{prefix}{safe_component}{suffix}"
                    )
                    sha256 = None
                    if exists:
                        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, snapshot_path)
                        sha256 = cls._artifact_sha256(snapshot_path)
                    records.append(
                        {
                            "component_name": component_name,
                            "path": path,
                            "existed": exists,
                            "snapshot_path": snapshot_path,
                            "sha256": sha256,
                        }
                    )
        if not records:
            shutil.rmtree(snapshot_root, ignore_errors=True)
            return None
        return {"root": snapshot_root, "records": records}

    @classmethod
    def _verify_and_restore_approved_component_artifacts(
        cls,
        snapshot: dict[str, Any] | None,
        working_dir: Path,
        *,
        active_component_name: str,
    ) -> tuple[list[str], list[str]]:
        """Restore cross-component mutations and report unrecoverable owners."""
        if not snapshot:
            return [], []

        changed_components: set[str] = set()
        unrecovered_components: set[str] = set()
        safe_active = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in active_component_name
        )
        audit_root = (
            working_dir
            / "models"
            / ".rejected_cross_component"
            / safe_active
            / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        try:
            for record in snapshot.get("records", []):
                component_name = str(record["component_name"])
                path = Path(record["path"])
                existed = bool(record["existed"])
                expected_hash = record.get("sha256")
                currently_exists = path.is_file()
                unchanged = (
                    existed
                    and currently_exists
                    and cls._artifact_sha256(path) == expected_hash
                ) or (not existed and not currently_exists)
                if unchanged:
                    continue

                changed_components.add(component_name)
                if currently_exists:
                    destination = (
                        audit_root
                        / "".join(
                            char
                            if char.isalnum() or char in {"-", "_"}
                            else "_"
                            for char in component_name
                        )
                        / path.name
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(destination)

                if existed:
                    snapshot_path = Path(record["snapshot_path"])
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(snapshot_path, path)
                        if cls._artifact_sha256(path) != expected_hash:
                            raise OSError("restored artifact hash mismatch")
                    except Exception as exc:
                        unrecovered_components.add(component_name)
                        print(
                            "   ❌ Could not restore approved artifact "
                            f"{path.name}: {exc}"
                        )

            if changed_components:
                print(
                    "   ❌ Cross-component artifact mutation detected from "
                    f"{active_component_name}; restored owners: "
                    f"{sorted(changed_components - unrecovered_components)}"
                )
        finally:
            shutil.rmtree(Path(snapshot["root"]), ignore_errors=True)

        return sorted(changed_components), sorted(unrecovered_components)

    @staticmethod
    def _state_with_revoked_component_evidence(
        state: KaggleState,
        component_names: list[str],
    ) -> KaggleState:
        """Return a shallow state copy with unrecoverable evidence revoked."""
        updated: dict[str, Any] = dict(state)
        availability = dict(state.get("oof_availability") or {})
        approvals = dict(state.get("robustness_approved_components") or {})
        component_results = dict(state.get("component_results") or {})
        trusted_scores = dict(state.get("trusted_component_scores") or {})
        for component_name in component_names:
            availability[component_name] = False
            approvals[component_name] = False
            component_results.pop(component_name, None)
            trusted_scores.pop(component_name, None)
        updated["oof_availability"] = availability
        updated["robustness_approved_components"] = approvals
        updated["component_results"] = component_results
        updated["trusted_component_scores"] = trusted_scores
        return updated  # type: ignore[return-value]

    @staticmethod
    def _begin_candidate_transaction(
        working_dir: Path,
        component_name: str,
    ) -> tuple[Path, tuple[Path, ...]]:
        """Snapshot mutable candidate artifacts before a refinement attempt."""
        safe_component = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in component_name
        )
        transaction_dir = (
            working_dir
            / ".candidate_transactions"
            / f"{safe_component}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        )
        transaction_dir.mkdir(parents=True, exist_ok=False)
        relative_paths = (
            Path("submission.csv"),
            Path("models") / f"oof_{component_name}.npy",
            Path("models") / f"test_{component_name}.npy",
            Path("models") / f"oof_{component_name}.npz",
            Path("models") / f"test_{component_name}.npz",
            Path("models") / f"train_ids_{component_name}.npy",
            Path("models") / f"test_ids_{component_name}.npy",
            Path("models") / f"class_order_{component_name}.npy",
            Path("models") / "class_order.npy",
        )
        for relative in relative_paths:
            source = working_dir / relative
            if source.is_file():
                backup = transaction_dir / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, backup)
        return transaction_dir, relative_paths

    @staticmethod
    def _finish_candidate_transaction(
        working_dir: Path,
        transaction: tuple[Path, tuple[Path, ...]],
        *,
        commit: bool,
    ) -> None:
        """Commit or atomically restore the candidate files in a transaction."""
        transaction_dir, relative_paths = transaction
        if not commit:
            for relative in relative_paths:
                current = working_dir / relative
                backup = transaction_dir / relative
                if current.is_file() or current.is_symlink():
                    current.unlink()
                if backup.is_file():
                    current.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup, current)
        shutil.rmtree(transaction_dir, ignore_errors=True)

    def __call__(self, state: KaggleState) -> dict[str, Any]:
        """
        Execute the developer agent.

        Args:
            state: Current workflow state

        Returns:
            State updates with development results
        """
        print("\n" + "=" * 60)
        print("= DEVELOPER AGENT: Implementing Components")
        print("=" * 60)

        ablation_plan = state.get("ablation_plan", [])
        current_index = state.get("current_component_index", 0)

        working_dir = Path(state["working_directory"])
        competition_info = state["competition_info"]
        metric_name = getattr(competition_info, "evaluation_metric", "")
        # Multiclass rows that do not sum to 1 are only a defect when the
        # graded metric reads a row as a probability vector. Under a
        # column-wise ranking metric the grader accepts them and scores them
        # normally, so both submission-format and artifact validation must
        # stop treating them as fatal or they destroy scoring candidates.
        needs_normalized_rows = metric_reads_rows_as_distribution(metric_name)

        if not ablation_plan:
            print("No ablation plan found. Run Planner Agent first.")
            return {}

        if current_index >= len(ablation_plan):
            print("All components implemented!")
            return {"current_component_index": current_index}

        # Respect explicit workflow-level early stopping decisions.
        if state.get("skip_remaining_components"):
            print("Skipping remaining components (skip_remaining_components=True)")
            return {"current_component_index": len(ablation_plan)}

        # Starting a component that cannot finish before the deadline spends
        # budget for nothing and delays the closing stages. The reserve keeps
        # enough clock to ensemble, validate and snapshot what already exists.
        if budget_exhausted(state, reserve_s=FINALIZATION_RESERVE_S):
            print(
                f"Wall-clock budget exhausted ({format_remaining(state)}) - "
                "closing the run with the components already accepted"
            )
            return {
                "current_component_index": len(ablation_plan),
                "skip_remaining_components": True,
                "last_updated": datetime.now(),
            }

        run_mode = str(state.get("run_mode", "")).lower()
        # Persist mode on the executor so refinement/debug calls remain under
        # the same MLE-bench filesystem boundary even after temporary
        # environment overrides are restored.
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.run_mode = run_mode
            executor.mlebench_cache_path = str(
                state.get("mlebench_cache_path") or ""
            )
        component = ablation_plan[current_index]

        # Defense in depth for resumed/checkpointed states whose plan predates
        # the Planner-side filter. A system ensemble ablation must not execute
        # a Developer-generated ensemble component either.
        toggles = getattr(self.config, "ablation_toggles", None)
        if (
            toggles
            and toggles.disable_ensemble
            and component.component_type == "ensemble"
        ):
            print(f"   ABLATION: skipping ensemble component {component.name}")
            return {
                "current_component_index": current_index + 1,
                "code_retry_count": 0,
                "telemetry_events": [
                    make_event(
                        "ablation",
                        "developer_ensemble_component_skipped",
                        iteration=state.get("current_iteration", 0),
                        component="ensemble",
                        component_name=component.name,
                    )
                ],
                "last_updated": datetime.now(),
            }

        # An ensemble component with nothing accepted to combine can only fail
        # or fabricate a fallback (constant predictions, identity copies).
        if component.component_type == "ensemble" and not _has_combinable_model_predictions(
            state, working_dir
        ):
            print(
                f"   Skipping ensemble component {component.name} - "
                "no accepted model predictions to combine"
            )
            return {
                "current_component_index": current_index + 1,
                "code_retry_count": 0,
                "telemetry_events": [
                    make_event(
                        "developer",
                        "ensemble_component_skipped_no_models",
                        iteration=state.get("current_iteration", 0),
                        component="ensemble",
                        component_name=component.name,
                    )
                ],
                "last_updated": datetime.now(),
            }

        print(f"\n= Implementing: {component.name} ({component.component_type})")
        print(f"Estimated Impact: {component.estimated_impact:.1%}")

        # Track original data size for data loss detection in feature engineering
        n_train_original_to_save: int | None = None
        if component.component_type == "feature_engineering":
            if "n_train_original" not in state:
                train_path = state.get("train_path")
                if train_path:
                    train_path = Path(train_path)
                else:
                    train_path = working_dir / "train.csv"
                if train_path.exists():
                    try:
                        with open(train_path) as f:
                            n_train_original_to_save = sum(1 for _ in f) - 1  # Subtract header
                        print(f"   Tracking original train size: {n_train_original_to_save:,} rows")
                    except Exception as e:
                        print(f"   ⚠️  Could not count train rows: {e}")

        def _coerce_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _coerce_score(value: Any) -> float | None:
            try:
                score = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(score):
                return None
            return score

        # Allow runners (e.g., MLE-bench) to cap runtime per component via state.
        base_timeout = _coerce_int(
            state.get("timeout_per_component"),
            self.config.ablation.testing_timeout,
        )
        if base_timeout <= 0:
            base_timeout = self.config.ablation.testing_timeout or 300

        # Use ComponentTimeoutConfig for per-type timeouts
        component_timeout_config = self.config.component_timeout
        name_lower = component.name.lower()

        # Get timeout from config based on component type and model name
        config_timeout = component_timeout_config.get_timeout(
            component.component_type,
            component.name
        )

        # Cap at base_timeout (runner may impose limits)
        desired_timeout = min(config_timeout, base_timeout)

        # Special handling for Optuna tuning (always use heavy timeout)
        if "optuna" in name_lower:
            desired_timeout = min(base_timeout, component_timeout_config.model_heavy)

        # A component may not outlive the run's wall-clock deadline. Without
        # this, a 50-minute component started 5 minutes before the deadline
        # overruns the whole budget for the sweep.
        budgeted_timeout = clamp_timeout_to_budget(
            state,
            desired_timeout,
            reserve_s=FINALIZATION_RESERVE_S,
        )
        if budgeted_timeout < desired_timeout:
            print(
                f"Component timeout clamped by run budget: "
                f"{desired_timeout}s -> {budgeted_timeout}s "
                f"({format_remaining(state)} left)"
            )
            desired_timeout = budgeted_timeout

        if self.executor.timeout != desired_timeout:
            self.executor.timeout = desired_timeout
            print(f"Component timeout set to: {desired_timeout}s ({desired_timeout / 60:.1f} min)")

        approved_artifact_snapshot = (
            self._snapshot_approved_component_artifacts(
                state,
                working_dir,
                active_component_name=component.name,
            )
        )
        try:
            result, attempt_records = self._implement_component(component, state)
        except (
            CanonicalTargetContractError,
            GeneratedContractStructureError,
            RepeatedInjectedContractError,
        ) as exc:
            self._verify_and_restore_approved_component_artifacts(
                approved_artifact_snapshot,
                working_dir,
                active_component_name=component.name,
            )
            if isinstance(exc, RepeatedInjectedContractError):
                return self._skip_duplicate_injected_contract(
                    state=state,
                    component=component,
                    current_index=current_index,
                    error=exc,
                )
            return self._record_pregeneration_contract_failure(
                state=state,
                component=component,
                current_index=current_index,
                error=exc,
            )
        except Exception:
            self._verify_and_restore_approved_component_artifacts(
                approved_artifact_snapshot,
                working_dir,
                active_component_name=component.name,
            )
            raise

        changed_approved, unrecovered_approved = (
            self._verify_and_restore_approved_component_artifacts(
                approved_artifact_snapshot,
                working_dir,
                active_component_name=component.name,
            )
        )
        if changed_approved:
            integrity_error = (
                "Cross-component artifact mutation blocked: "
                + ", ".join(changed_approved)
            )
            result.success = False
            result.errors = [*(result.errors or []), integrity_error]
            result.stderr = (
                f"{result.stderr}\n{integrity_error}".strip()
            )
        if unrecovered_approved:
            state = self._state_with_revoked_component_evidence(
                state,
                unrecovered_approved,
            )

        if getattr(result, "retryable", True) is False:
            # An invalid harness attempt, not model-quality evidence: the
            # component is neither retried nor recorded as a model failure, and
            # everything already accepted in this run stays exactly as it is.
            return self._record_injected_contract_failure(
                state=state,
                component=component,
                working_dir=working_dir,
                current_index=current_index,
                attempt_records=attempt_records,
                result=result,
            )

        should_keep_component = True
        new_cv_score: float | None = None
        primary_score: float | None = None
        primary_score_source: str | None = None
        if (
            result.success
            and component.component_type == "model"
            and not getattr(result, "reused_from_cache", False)
        ):
            exec_result = ExecutionResult(
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=result.execution_time,
                exit_code=0 if result.success else -1,
                artifacts_created=result.artifacts_created,
                errors=result.errors,
            )

            should_keep_component, new_cv_score = self._validate_component_improvement(
                component, exec_result, state
            )

            if not should_keep_component:
                print("\nROLLBACK: Component did not improve score - discarding")
                return self._reject_model_candidate(
                    state=state,
                    component=component,
                    working_dir=working_dir,
                    current_index=current_index,
                    attempt_records=attempt_records,
                    reason="No independently recomputed OOF improvement",
                    retry_invalid=False,
                )

        code_retry_count = _coerce_int(state.get("code_retry_count"), 0)
        max_component_retries = _coerce_int(os.getenv("KAGGLE_AGENTS_MAX_COMPONENT_RETRIES"), 3)
        max_component_retries = max(1, max_component_retries)
        min_component_score = None
        min_component_score_env = os.getenv("KAGGLE_AGENTS_MIN_COMPONENT_SCORE")
        if min_component_score_env:
            try:
                min_component_score = float(min_component_score_env)
            except ValueError:
                print(f"⚠️ Invalid KAGGLE_AGENTS_MIN_COMPONENT_SCORE='{min_component_score_env}'")
                min_component_score = None
        if not result.success:
            failure_detail = next(
                (
                    str(value).strip()
                    for value in (result.errors or [])
                    if str(value).strip()
                ),
                "",
            )
            if not failure_detail:
                failure_detail = (result.stderr or "Component execution failed").strip()
            print(
                f"⚠️ Component execution failed; invalidating candidate before "
                f"retry: {component.name}"
            )
            return self._reject_model_candidate(
                state=state,
                component=component,
                working_dir=working_dir,
                current_index=current_index,
                attempt_records=attempt_records,
                reason=failure_detail[:1000],
                retry_invalid=True,
            )

        state_updates: dict[str, Any] = {
            "development_results": [result] if should_keep_component else [],
            "current_code": result.code,
            "code_retry_count": 0,
            "current_component_index": current_index + 1,
            "last_updated": datetime.now(),
            "code_attempts": attempt_records,
        }
        if (
            run_mode == "mlebench"
            and component.component_type == "model"
            # A reused result was scored when it first ran and its gate was
            # skipped above, so new_cv_score is None here. Rewriting the map
            # would strip the evidence the ensemble gate needs.
            and not getattr(result, "reused_from_cache", False)
        ):
            trusted_scores = dict(state.get("trusted_component_scores") or {})
            trusted_score = _coerce_score(new_cv_score)
            if result.success and trusted_score is not None:
                trusted_scores[component.name] = trusted_score
            else:
                trusted_scores.pop(component.name, None)
            state_updates["trusted_component_scores"] = trusted_scores

        # Save original train row count for data loss detection
        if n_train_original_to_save is not None:
            state_updates["n_train_original"] = n_train_original_to_save

        # Persist the target decision this candidate was generated against:
        # the fingerprint and protected-input manifest belong in the execution
        # record, not in an instance attribute nothing ever reads.
        if self._last_target_source_metadata is not None:
            state_updates["target_source_record"] = {
                "component_name": component.name,
                "component_type": component.component_type,
                **self._last_target_source_metadata,
            }
            self._last_target_source_metadata = None
            self._last_target_source = None

        # GRPO: Persist reasoning trace in state
        if self._last_reasoning_trace is not None:
            state_updates["reasoning_traces"] = [self._last_reasoning_trace]
            state_updates["current_reasoning"] = self._last_reasoning_trace
            self._last_reasoning_trace = None  # Reset for next component

        # DPO: Persist preference pairs in state
        preference_pairs = self._preference_collector.get_pairs_for_state()
        if preference_pairs:
            state_updates["preference_pairs"] = preference_pairs
            print(f"   📊 DPO: Collected {len(preference_pairs)} preference pairs")

        # Quiet-STaR: Persist self-evaluation in state
        if self._last_self_evaluation is not None:
            state_updates["self_evaluations"] = [self._last_self_evaluation]
            state_updates["last_self_evaluation"] = self._last_self_evaluation
            self._last_self_evaluation = None  # Reset for next component

        if result.success and component.component_type == "model":
            # === STRICT VALIDATION OF MODEL ARTIFACTS ===
            # This replaces the old warning-only approach with comprehensive validation
            from kaggle_agents.utils.strict_validation import (
                StrictValidationConfig,
                validate_model_artifacts,
                validate_prediction_quality,
            )

            validation_config = StrictValidationConfig.from_env()
            if run_mode == "mlebench":
                validation_config.strict_mode = True
                validation_config.require_train_ids = True
                validation_config.require_test_ids = True

            # Get expected values from state
            expected_n_train = state.get("expected_train_rows")
            expected_n_test = state.get("expected_test_rows")
            competition_info = state.get("competition_info")
            validation_config.require_normalized_rows = needs_normalized_rows
            problem_type = _model_validation_problem_type(state)
            expected_class_order = _validation_class_order_for_state(
                state,
                problem_type,
            )
            if (
                run_mode == "mlebench"
                and expected_class_order is not None
                and _requires_class_order_artifact(state, problem_type)
            ):
                validation_config.require_class_order = True
                validation_config.require_component_class_order = True
            canonical_contract = state.get("canonical_contract") or {}
            expected_train_ids = None
            canonical_train_ids_path = canonical_contract.get("train_ids_path")
            if canonical_train_ids_path and Path(canonical_train_ids_path).is_file():
                expected_train_ids = load_npy_readonly(
                    canonical_train_ids_path,
                    allow_pickle=True,
                ).reshape(-1)
            expected_test_ids = state.get("test_rec_ids") or None
            canonical_test_ids_path = canonical_contract.get("test_ids_path")
            if (
                expected_test_ids is None
                and canonical_test_ids_path
                and Path(canonical_test_ids_path).is_file()
            ):
                expected_test_ids = load_npy_readonly(
                    canonical_test_ids_path,
                    allow_pickle=False,
                ).reshape(-1)

            # Run comprehensive validation
            validation_result = validate_model_artifacts(
                working_dir=working_dir,
                component_name=component.name,
                expected_n_train=expected_n_train,
                expected_n_test=expected_n_test,
                expected_class_order=expected_class_order,
                expected_train_ids=expected_train_ids,
                expected_test_ids=expected_test_ids,
                problem_type=problem_type,
                config=validation_config,
            )

            # Report validation results
            if validation_result.is_valid:
                print(f"   Validated artifacts: {', '.join(validation_result.files_verified)}")
                # Log any warnings even if valid
                for warning in validation_result.warnings:
                    print(f"   Warning: {warning}")
            else:
                # Report all errors
                print(f"   Model {component.name} failed artifact validation:")
                for error in validation_result.errors:
                    print(f"      ERROR: {error}")
                for warning in validation_result.warnings:
                    print(f"      Warning: {warning}")

                # In strict mode, mark component as failed
                if validation_config.strict_mode:
                    print("   STRICT MODE: Marking component as FAILED due to validation errors")
                    result.success = False
                    if not hasattr(result, 'errors') or result.errors is None:
                        result.errors = []
                    result.errors.extend(validation_result.errors)
                else:
                    print("   Lenient mode: Continuing despite validation errors (enable KAGGLE_AGENTS_STRICT_MODE=1 for hard failures)")

            # Additionally, check for random/broken predictions if OOF exists
            oof_file = _component_prediction_artifact(
                working_dir, component.name, "oof"
            )
            if oof_file.exists() and problem_type != "image_to_image":
                try:
                    oof_preds = load_npy_readonly(
                        oof_file,
                        allow_pickle=False,
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
                        if oof_eligible_mask.shape != (len(oof_preds),):
                            raise ValueError(
                                "Canonical OOF eligibility mask is not "
                                "prediction-aligned"
                            )
                        if np.issubdtype(oof_preds.dtype, np.number):
                            warmup_oof = oof_preds[~oof_eligible_mask]
                            if (
                                warmup_oof.size
                                and not np.isnan(warmup_oof).all()
                            ):
                                raise ValueError(
                                    "Temporal warm-up OOF rows must remain NaN"
                                )
                            quality_oof = oof_preds[oof_eligible_mask]
                        else:
                            if not np.all(oof_eligible_mask):
                                raise ValueError(
                                    "Temporal seq2seq OOF has no supported "
                                    "text warm-up sentinel"
                                )
                            quality_oof = oof_preds
                    else:
                        quality_oof = oof_preds

                    # Check for empty OOF rows (unfilled predictions), which
                    # invalidates held-out evaluation and downstream ensembles.
                    # NOTE: Only check multiclass problems where OOF is a probability distribution
                    # For binary/regression, zero is a legitimate prediction value
                    if (
                        "multiclass" in str(problem_type).lower()
                        and quality_oof.ndim > 1
                    ):
                        # Multiclass OOF should be probability distributions summing to ~1
                        # Rows summing to 0 are definitely unfilled
                        empty_rows = int(
                            np.sum(quality_oof.sum(axis=1) == 0)
                        )

                        if empty_rows > 0:
                            empty_pct = (
                                empty_rows / quality_oof.shape[0] * 100
                            )
                            print(f"   ⚠️ Empty OOF rows detected: {empty_rows} ({empty_pct:.2f}%)")

                            # In MLE-bench mode, block submissions with >1% empty rows
                            if run_mode == "mlebench" and empty_pct > 1.0:
                                print(f"   ❌ BLOCKING: Too many empty OOF rows ({empty_pct:.2f}% > 1.0%) - held-out predictions are incomplete")
                                result.success = False
                                if not hasattr(result, 'errors') or result.errors is None:
                                    result.errors = []
                                result.errors.append(f"Empty OOF rows: {empty_rows} ({empty_pct:.2f}%) - blocking submission")

                    is_quality_ok, quality_issues = validate_prediction_quality(
                        quality_oof, problem_type=problem_type
                    )
                    if not is_quality_ok:
                        print(f"   Prediction quality issues for {component.name}:")
                        for issue in quality_issues:
                            print(f"      - {issue}")
                        if validation_config.strict_mode:
                            result.success = False
                            if not hasattr(result, 'errors') or result.errors is None:
                                result.errors = []
                            result.errors.extend(quality_issues)
                except Exception as e:
                    print(f"   Warning: Could not check prediction quality: {e}")
            # === END STRICT VALIDATION ===

            if not result.success:
                return self._reject_model_candidate(
                    state=state,
                    component=component,
                    working_dir=working_dir,
                    current_index=current_index,
                    attempt_records=attempt_records,
                    reason="Model artifact validation failed",
                    retry_invalid=True,
                )

            submission_candidates = [
                working_dir / "submission.csv",
                Path(state.get("submission_path"))
                if state.get("submission_path")
                else None,
            ]
            submission_path = next(
                (p for p in submission_candidates if p is not None and p.exists() and p.is_file()),
                None,
            )
            sample_sub_path = (
                Path(state.get("sample_submission_path"))
                if state.get("sample_submission_path")
                else working_dir / "sample_submission.csv"
            )
            if sample_sub_path.exists() and sample_sub_path.is_dir():
                inner_csvs = sorted(sample_sub_path.glob("*.csv"))
                if inner_csvs:
                    sample_sub_path = inner_csvs[0]
                    print(f"   📂 Resolved directory to file: {sample_sub_path.name}")
            target_cols = [
                str(column)
                for column in (
                    (state.get("submission_contract") or {}).get("target_cols")
                    or []
                )
                if isinstance(column, str) and column
            ]
            if _uses_packed_image_artifacts(working_dir):
                # The final packed artifact is authoritative. Regenerate even
                # when generated code already wrote a CSV, because it could
                # write from artifact A and then replace test_<component>.npz
                # with artifact B before exiting.
                from ...utils.submission_artifacts import (
                    rebuild_submission_from_component_predictions,
                )

                submission_path = (
                    rebuild_submission_from_component_predictions(
                        working_dir=working_dir,
                        component_name=component.name,
                        sample_submission_path=sample_sub_path,
                        target_cols=target_cols,
                        id_col=(state.get("submission_contract") or {}).get(
                            "id_col"
                        ),
                    )
                    if sample_sub_path.is_file()
                    else None
                )
            else:
                submission_path = self._recover_missing_submission(
                    run_mode=run_mode,
                    submission_path=submission_path,
                    working_dir=working_dir,
                    component_name=component.name,
                    sample_submission_path=sample_sub_path,
                    target_cols=target_cols,
                    id_col=(state.get("submission_contract") or {}).get(
                        "id_col"
                    ),
                    test_ids_are_positional=bool(
                        (state.get("canonical_metadata") or {}).get(
                            "test_ids_are_positional", False
                        )
                    ),
                )
            if run_mode == "mlebench" and submission_path is None:
                if not hasattr(result, "errors") or result.errors is None:
                    result.errors = []
                result.errors.append(
                    "Submission file is absent and could not be rebuilt from "
                    "validated test predictions"
                )
                result.success = False
                return self._reject_model_candidate(
                    state=state,
                    component=component,
                    working_dir=working_dir,
                    current_index=current_index,
                    attempt_records=attempt_records,
                    reason="Submission missing after artifact recovery",
                    retry_invalid=True,
                )
            if submission_path:
                backup_name = f"submission_{component.name}.csv"
                backup_path = working_dir / backup_name

                # Validate submission structure without exposing test-set feedback.
                submission_is_valid = False
                submission_validation_message = "Sample submission contract is unavailable"
                if sample_sub_path and sample_sub_path.exists() and sample_sub_path.is_file():
                    is_valid, validation_msg = self.executor.validate_submission_format(
                        submission_path=submission_path,
                        sample_submission_path=sample_sub_path,
                        component_type=component.component_type,
                        problem_type=problem_type,
                        target_cols=target_cols or None,
                        require_normalized_rows=needs_normalized_rows,
                    )
                    submission_is_valid = is_valid
                    submission_validation_message = validation_msg
                    if not is_valid:
                        print(f"   ❌ Submission validation FAILED: {validation_msg}")
                    else:
                        print(f"   {validation_msg}")

                if run_mode == "mlebench" and not submission_is_valid:
                    from ...utils.submission_artifacts import (
                        rebuild_submission_from_component_predictions,
                    )

                    rebuilt_submission = rebuild_submission_from_component_predictions(
                        working_dir=working_dir,
                        component_name=component.name,
                        sample_submission_path=sample_sub_path,
                        target_cols=target_cols,
                        id_col=(state.get("submission_contract") or {}).get("id_col"),
                        test_ids_are_positional=bool(
                            (state.get("canonical_metadata") or {}).get(
                                "test_ids_are_positional", False
                            )
                        ),
                    ) if sample_sub_path else None
                    if rebuilt_submission is not None:
                        is_valid, validation_msg = self.executor.validate_submission_format(
                            submission_path=rebuilt_submission,
                            sample_submission_path=sample_sub_path,
                            component_type=component.component_type,
                            problem_type=problem_type,
                            target_cols=target_cols or None,
                            require_normalized_rows=needs_normalized_rows,
                        )
                        submission_is_valid = is_valid
                        submission_validation_message = validation_msg
                        submission_path = rebuilt_submission
                        if is_valid:
                            print(
                                "   Rebuilt invalid submission.csv from validated "
                                f"{_component_prediction_artifact(working_dir, component.name, 'test').name}"
                            )

                if run_mode == "mlebench" and not submission_is_valid:
                    if not hasattr(result, "errors") or result.errors is None:
                        result.errors = []
                    result.errors.append(submission_validation_message)
                    result.success = False
                    return self._reject_model_candidate(
                        state=state,
                        component=component,
                        working_dir=working_dir,
                        current_index=current_index,
                        attempt_records=attempt_records,
                        reason=f"Submission validation failed: {submission_validation_message}",
                        retry_invalid=True,
                    )

                shutil.copy(submission_path, backup_path)
                print(f"Backup submission saved: {backup_name}")

                primary_score = _resolved_primary_score(
                    result, component.name, state, new_cv_score
                )
                if primary_score is not None:
                    primary_score_source = "cv"

                current_best_score = state.get("best_single_model_score")
                if run_mode == "mlebench" and primary_score is None:
                    previous_candidate = state.get(
                        "best_candidate_submission_snapshot_path"
                    ) or state.get("accepted_submission_snapshot_path")
                    if previous_candidate:
                        return self._reject_model_candidate(
                            state=state,
                            component=component,
                            working_dir=working_dir,
                            current_index=current_index,
                            attempt_records=attempt_records,
                            reason=(
                                "Candidate has no independently reproducible OOF "
                                "score and cannot replace a preserved candidate"
                            ),
                            retry_invalid=False,
                        )
                    if not state.get("run_id"):
                        return self._reject_model_candidate(
                            state=state,
                            component=component,
                            working_dir=working_dir,
                            current_index=current_index,
                            attempt_records=attempt_records,
                            reason="Unscored candidate cannot be preserved without run_id",
                            retry_invalid=False,
                        )

                    # Keep the first structurally valid model as a deterministic
                    # fallback, but do not assign it a fabricated CV score.
                    from ...utils.submission_artifacts import (
                        snapshot_best_candidate_submission,
                    )

                    snapshot, digest = snapshot_best_candidate_submission(
                        working_dir,
                        submission_path,
                        run_id=str(state["run_id"]),
                        iteration=int(state.get("current_iteration") or 0),
                    )
                    state_updates["best_candidate_submission_snapshot_path"] = str(
                        snapshot
                    )
                    state_updates["best_candidate_submission_sha256"] = digest
                    state_updates[
                        "best_candidate_submission_component_name"
                    ] = component.name
                    state_updates["best_single_model_name"] = component.name
                    shutil.copy(submission_path, working_dir / "submission_best.csv")
                    print(
                        "Preserved first valid candidate as an unscored fallback; "
                        "it is not treated as a CV improvement"
                    )

                if min_component_score is not None and not state_updates.get(
                    "skip_remaining_components"
                ):
                    score_for_gate = primary_score
                    score_source = primary_score_source
                    if score_for_gate is None:
                        cv_score = _coerce_score(new_cv_score)
                        if cv_score is not None:
                            score_for_gate = cv_score
                            score_source = "cv"
                    if score_for_gate is None:
                        is_minimize = is_metric_minimization(metric_name)
                        score_for_gate = float("inf") if is_minimize else float("-inf")
                        score_source = "missing"

                    is_minimize = is_metric_minimization(metric_name)
                    below_threshold = (
                        score_for_gate > min_component_score
                        if is_minimize
                        else score_for_gate < min_component_score
                    )
                    if below_threshold:
                        retry_next = code_retry_count + 1
                        action = (
                            "rejecting component after retry exhaustion"
                            if retry_next >= max_component_retries
                            else "retrying component"
                        )
                        print(
                            f"🔄 Score {score_for_gate} ({score_source}) below threshold "
                            f"{min_component_score:.5f}; {action} "
                            f"({retry_next}/{max_component_retries})."
                        )
                        return self._reject_model_candidate(
                            state=state,
                            component=component,
                            working_dir=working_dir,
                            current_index=current_index,
                            attempt_records=attempt_records,
                            reason=(
                                f"Trusted score {score_for_gate} ({score_source}) "
                                f"below required threshold {min_component_score}"
                            ),
                            retry_invalid=True,
                        )

                is_best = False
                if primary_score is not None and not self._is_score_implausible(
                    primary_score, metric_name, trusted=run_mode == "mlebench"
                ):
                    best_str = f"{current_best_score:.5f}" if current_best_score is not None else "None"
                    print(f"[SCORE COMPARISON] Current best: {best_str}, New score: {primary_score:.5f} (source: {primary_score_source})")

                    if current_best_score is None:
                        is_best = True
                        print("[SCORE COMPARISON] Action: UPDATE submission_best (no previous best)")
                    else:
                        improvement = calculate_score_improvement(
                            primary_score, current_best_score, metric_name
                        )
                        print(f"[SCORE COMPARISON] Improvement: {improvement:.5f}")
                        if improvement > 0:
                            is_best = True
                            print("[SCORE COMPARISON] Action: UPDATE submission_best (score improved)")
                        else:
                            print("[SCORE COMPARISON] Action: KEEP existing submission_best (no improvement)")

                if is_best:
                    if run_mode == "mlebench" and not state.get("run_id"):
                        print(
                            "❌ Refusing best-candidate promotion without a run_id"
                        )
                        is_best = False

                if is_best:
                    print(f"✅ New Best Single Model! ({primary_score:.4f}, source: {primary_score_source})")
                    state_updates["best_single_model_score"] = primary_score
                    state_updates["best_single_model_name"] = component.name

                    if run_mode == "mlebench":
                        from ...utils.submission_artifacts import (
                            snapshot_best_candidate_submission,
                        )

                        snapshot, digest = snapshot_best_candidate_submission(
                            working_dir,
                            submission_path,
                            run_id=str(state["run_id"]),
                            iteration=int(state.get("current_iteration") or 0),
                        )
                        state_updates["best_candidate_submission_snapshot_path"] = str(
                            snapshot
                        )
                        state_updates["best_candidate_submission_sha256"] = digest
                        state_updates[
                            "best_candidate_submission_component_name"
                        ] = component.name

                    best_path = working_dir / "submission_best.csv"
                    shutil.copy(submission_path, best_path)
                    print("Saved to submission_best.csv")

                    models_dir = working_dir / "models"
                    model_exts = {".pth", ".pt", ".keras", ".h5", ".joblib", ".pkl"}
                    model_candidates: list[Path] = []
                    if models_dir.exists():
                        for rel in result.artifacts_created:
                            rel_path = Path(rel)
                            if rel_path.parts[:1] == ("models",) and rel_path.suffix in model_exts:
                                model_candidates.append(working_dir / rel_path)
                        if not model_candidates:
                            for ext in model_exts:
                                model_candidates.extend(models_dir.glob(f"*{ext}"))
                            if model_candidates:
                                with_name = [
                                    p for p in model_candidates if component.name in p.name
                                ]
                                if with_name:
                                    model_candidates = with_name
                    if model_candidates:
                        try:
                            best_model_path = max(
                                model_candidates, key=lambda p: p.stat().st_mtime
                            )
                            best_model_target = (
                                models_dir / f"best_model{best_model_path.suffix}"
                            )
                            shutil.copy(best_model_path, best_model_target)
                            state_updates["best_single_model_checkpoint"] = str(
                                best_model_target
                            )
                            print(
                                f"Saved best model checkpoint to {best_model_target.name}"
                            )
                        except Exception as e:
                            print(f"⚠️ Failed to save best model checkpoint: {e}")
            else:
                print("Warning: submission.csv not found after successful execution")
                if run_mode == "mlebench":
                    return self._reject_model_candidate(
                        state=state,
                        component=component,
                        working_dir=working_dir,
                        current_index=current_index,
                        attempt_records=attempt_records,
                        reason="Model did not produce submission.csv",
                        retry_invalid=True,
                    )

            if (
                result.success
                and should_keep_component
                and (primary_score is not None or new_cv_score is not None)
            ):
                baseline_candidate = _coerce_score(new_cv_score)
                if baseline_candidate is not None:
                    baseline_current = _coerce_score(state.get("baseline_cv_score"))
                    should_update = baseline_current is None or calculate_score_improvement(
                        baseline_candidate, baseline_current, metric_name
                    ) > 0
                    if should_update:
                        state_updates["baseline_cv_score"] = baseline_candidate
                        print(f"Updated baseline CV score: {baseline_candidate:.4f}")
                        state_updates["current_performance_score"] = baseline_candidate

        if result.success and component.component_type in {"model", "ensemble"}:
            submission_path = working_dir / "submission.csv"
            best_submission = working_dir / "submission_best.csv"
            baseline_score = _coerce_score(state.get("baseline_cv_score"))
            if baseline_score is None:
                baseline_score = _coerce_score(state.get("best_single_model_score"))

            def restore_preserved_submission(reason: str) -> bool:
                """Restore a mutable legacy best only outside MLE-bench."""
                if run_mode == "mlebench":
                    from ...utils.submission_artifacts import (
                        restore_accepted_submission,
                        restore_best_candidate_submission,
                    )

                    restored = restore_best_candidate_submission(state, working_dir)
                    if restored is None:
                        restored = restore_accepted_submission(state, working_dir)
                    if restored is None:
                        submission_path.unlink(missing_ok=True)
                        state_updates["workflow_valid"] = False
                        state_updates["submission_validation_error"] = (
                            f"Candidate rejected ({reason}); no verified immutable "
                            "submission snapshot exists"
                        )
                        print(
                            "No verified immutable submission snapshot exists; "
                            "submission.csv removed"
                        )
                        return False
                    print(
                        "Restored submission.csv from a hash-verified immutable "
                        f"snapshot ({reason})"
                    )
                    return True

                if not best_submission.is_file():
                    return False
                shutil.copy(best_submission, submission_path)
                print(
                    f"Restored submission.csv from submission_best.csv ({reason})"
                )
                return True

            score_for_gate = primary_score
            if score_for_gate is None and run_mode != "mlebench":
                # Strict marker only: the lenient _extract_cv_score also matches
                # decorated lines like "Final Validation Performance (rmse): ..."
                # and once promoted a mocked score into submission_best.csv.
                extracted = self.executor.extract_performance_metric(result.stdout)
                score_for_gate = _coerce_score(extracted)
            if score_for_gate is not None and self._is_score_implausible(
                score_for_gate, metric_name, trusted=run_mode == "mlebench"
            ):
                print(
                    f"Implausible {metric_name} score {score_for_gate}; "
                    "ignoring for submission gating"
                )
                score_for_gate = None
                if submission_path.exists() and restore_preserved_submission(
                    "implausible score"
                ):
                    state_updates["submission_reverted"] = True
                    state_updates["submission_revert_reason"] = "implausible_score"

            if (
                isinstance(baseline_score, (int, float))
                and isinstance(score_for_gate, (int, float))
                and submission_path.exists()
                and (run_mode == "mlebench" or best_submission.exists())
            ):
                is_minimize = is_metric_minimization(metric_name)
                is_worse = (
                    score_for_gate > float(baseline_score)
                    if is_minimize
                    else score_for_gate < float(baseline_score)
                )
                if is_worse:
                    if restore_preserved_submission("score worse than baseline"):
                        state_updates["submission_reverted"] = True
                        state_updates["submission_revert_reason"] = (
                            "worse_than_baseline"
                        )

            if (
                run_mode == "mlebench"
                and component.component_type == "ensemble"
                and isinstance(score_for_gate, (int, float))
                and submission_path.exists()
            ):
                baseline_current = _coerce_score(baseline_score)
                if baseline_current is None:
                    is_minimize = is_metric_minimization(metric_name)
                    baseline_current = float("inf") if is_minimize else float("-inf")
                improvement = calculate_score_improvement(
                    score_for_gate, baseline_current, metric_name
                )
                if improvement > 0:
                    state_updates["baseline_cv_score"] = float(score_for_gate)
                    shutil.copy(submission_path, best_submission)
                    state_updates["current_performance_score"] = float(score_for_gate)
                    print("Updated submission_best.csv with improved CV/OOF score")

        # Persist explicit mappings declared in KaggleState. LangGraph drops
        # undeclared dynamic keys such as ``oof_available_<component>``.
        if result.success and component.component_type == "model":
            oof_file = _component_prediction_artifact(
                working_dir, component.name, "oof"
            )
            # In mlebench, an OOF file without a trusted recomputed score is
            # not ensemble evidence: marking it eligible would send the
            # component into the robustness evidence check, which rejects and
            # quarantines it — destroying the unscored fallback on
            # canonical-less domains.
            oof_is_evidence = run_mode != "mlebench" or new_cv_score is not None
            if oof_file.exists() and oof_is_evidence:
                oof_availability = dict(state.get("oof_availability") or {})
                oof_availability[component.name] = True
                state_updates["oof_availability"] = oof_availability
                robustness_approvals = dict(
                    state.get("robustness_approved_components") or {}
                )
                robustness_approvals[component.name] = False
                state_updates["robustness_approved_components"] = (
                    robustness_approvals
                )
                print(f"   OOF file available for ensemble: {component.name}")

        if result.success and should_keep_component:
            component_results = dict(state.get("component_results") or {})
            component_results[component.name] = result
            state_updates["component_results"] = component_results
            print(f"Cached successful result for: {component.name}")

            if (
                component.component_type == "model"
                and (run_mode != "mlebench" or new_cv_score is not None)
                and self._should_run_refinement(
                    component,
                    state,
                    new_cv_score,
                    execution_time_s=result.execution_time,
                    component_timeout_s=desired_timeout,
                )
            ):
                print("\nADK Refinement Loop: Trying to improve score...")
                best_code = result.code
                # None when the component was kept without a comparable score;
                # a refined score then improves on nothing measurable, so it
                # may become the local baseline but must not claim the global
                # best by "beating" a fabricated 0.0.
                best_score = new_cv_score
                best_stdout = result.stdout
                desired_direction = (
                    "LOWER"
                    if is_metric_minimization(metric_name)
                    else "HIGHER"
                )

                refinement_iters = self._get_refinement_iterations(state)
                for i in range(refinement_iters):
                    print(f"Refinement Iteration {i + 1}/{refinement_iters}")

                    # Parse training logs for structured feedback
                    training_feedback = parse_training_logs(best_stdout)
                    formatted_feedback = ""

                    if training_feedback.has_data():
                        formatted_feedback = format_feedback_for_llm(training_feedback)
                        print("📊 Training feedback extracted from logs")

                        if training_feedback.fold_scores:
                            print(
                                f"   CV: {training_feedback.cv_mean:.4f} ± {training_feedback.cv_std:.4f}"
                            )
                        if training_feedback.best_optuna_trial:
                            print(
                                f"   Best Optuna trial: {training_feedback.best_optuna_trial.get('score', 0):.4f}"
                            )
                        if training_feedback.slowest_step:
                            print(f"   Slowest step: {training_feedback.slowest_step}")

                        suggestions = training_feedback.get_improvement_suggestions()
                        if suggestions:
                            print("   Suggestions:")
                            for s in suggestions[:3]:
                                print(f"   - {s[:80]}...")

                    best_score_text = (
                        f"{best_score:.6f}" if best_score is not None else "not measured"
                    )
                    refine_prompt = f"""
## Current Performance
- CV Score: {best_score_text}

{formatted_feedback if formatted_feedback else "No structured training logs available."}

## Improvement Task
Based on the training results above, improve the model to achieve a {desired_direction} CV score.

**Improvement Guidelines**:
1. If CV std > 0.02: Add regularization or reduce model complexity
2. If overfitting detected: Increase reg_alpha/reg_lambda, reduce max_depth, add dropout
3. If underfitting detected: Increase model complexity, add features, reduce regularization
4. If Optuna best params available: Use them as starting point
5. If zero-importance features found: Remove them
6. If training is slow: Optimize hyperparameters for speed

**IMPORTANT**:
- Keep the same logging format ([LOG:FOLD], [LOG:OPTUNA], etc.) for the next iteration
- Return the complete updated Python code
- Focus on the most impactful change based on the feedback above
"""

                    system_prompt = (
                        f"{DEVELOPER_CORE_IDENTITY}\n\n{HARD_CONSTRAINTS}\n\n"
                        f"{REFINEMENT_TRUST_BOUNDARY}"
                    )
                    refine_messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(
                            content=f"Here is the current working code:\n```python\n{best_code}\n```\n\n{refine_prompt}"
                        ),
                    ]

                    transaction: tuple[Path, tuple[Path, ...]] | None = None
                    transaction_committed = False
                    try:
                        refined_response = self.llm.invoke(refine_messages)
                        refined_code = preserve_injected_header(
                            best_code,
                            self._extract_code_from_response(
                                get_text_content(refined_response.content)
                            ),
                        )
                        refinement_errors: list[str] = []
                        helper_shadow = untrusted_contract_helper_import(
                            refined_code
                        )
                        if helper_shadow:
                            refinement_errors.append(
                                HELPER_IMPORT_CONTRACT_ERROR
                            )
                        if requires_submission_helper(
                            component.component_type
                        ) and missing_submission_helper_call(refined_code):
                            refinement_errors.append(
                                MISSING_SUBMISSION_HELPER_ERROR
                            )
                        handwritten = (
                            handwritten_submission_write(refined_code)
                            if requires_submission_helper(
                                component.component_type
                            )
                            else None
                        )
                        if handwritten:
                            refinement_errors.append(
                                SUBMISSION_CONTRACT_ERROR
                            )
                        missing_artifacts = unsaved_expected_artifacts(
                            refined_code,
                            _expected_model_artifacts(
                                component,
                                working_dir,
                                run_mode,
                            ),
                            component.name,
                        )
                        if missing_artifacts:
                            refinement_errors.append(
                                "Missing expected artifacts: "
                                + ", ".join(missing_artifacts)
                            )
                        if (
                            component.component_type == "model"
                            and expected_class_order is not None
                            and _requires_class_order_artifact(
                                state,
                                problem_type,
                            )
                            and missing_class_order_helper_argument(
                                refined_code
                            )
                        ):
                            refinement_errors.append(
                                MISSING_CLASS_ORDER_ERROR
                            )
                        if refinement_errors:
                            print(
                                "Refined candidate rejected before execution: "
                                + "; ".join(refinement_errors)
                            )
                            continue

                        transaction = self._begin_candidate_transaction(
                            working_dir, component.name
                        )
                        print("Executing refined code...")
                        approved_refinement_snapshot = (
                            self._snapshot_approved_component_artifacts(
                                state,
                                working_dir,
                                active_component_name=component.name,
                            )
                        )
                        # The executor still holds the timeout computed for the
                        # first attempt. Re-clamp here or a refinement started
                        # minutes before the deadline runs for its full
                        # component budget past it.
                        self.executor.timeout = clamp_timeout_to_budget(
                            state,
                            self.executor.timeout,
                            reserve_s=FINALIZATION_RESERVE_S,
                        )
                        # A refinement that never rewrites its OOF file is
                        # scored on the accepted model's artifacts, so the gate
                        # compares the incumbent against itself and can neither
                        # accept a real gain nor notice a regression. Remember
                        # the evidence it must replace.
                        oof_before = _oof_artifact_digest(working_dir, component.name)
                        refinement_expected_artifacts = list(
                            _expected_model_artifacts(
                                component,
                                working_dir,
                                run_mode,
                            )
                            or []
                        )
                        if requires_submission_helper(component.component_type):
                            refinement_expected_artifacts.append(
                                "submission.csv"
                            )
                        if (
                            component.component_type == "model"
                            and expected_class_order is not None
                            and _requires_class_order_artifact(
                                state,
                                problem_type,
                            )
                        ):
                            refinement_expected_artifacts.append(
                                "models/"
                                f"class_order_{component.name}.npy"
                            )
                        try:
                            refined_exec = execute_generated_candidate(
                                self.executor,
                                refined_code,
                                working_dir=working_dir,
                                expected_artifacts=(
                                    refinement_expected_artifacts or None
                                ),
                                component_type=component.component_type,
                            )
                        except Exception:
                            self._verify_and_restore_approved_component_artifacts(
                                approved_refinement_snapshot,
                                working_dir,
                                active_component_name=component.name,
                            )
                            raise
                        (
                            changed_during_refinement,
                            unrecovered_during_refinement,
                        ) = self._verify_and_restore_approved_component_artifacts(
                            approved_refinement_snapshot,
                            working_dir,
                            active_component_name=component.name,
                        )
                        if changed_during_refinement:
                            refined_exec.success = False
                            refined_exec.errors = [
                                *(refined_exec.errors or []),
                                (
                                    "Cross-component artifact mutation blocked: "
                                    + ", ".join(changed_during_refinement)
                                ),
                            ]
                        if unrecovered_during_refinement:
                            revoked_state = (
                                self._state_with_revoked_component_evidence(
                                    state,
                                    unrecovered_during_refinement,
                                )
                            )
                            for key in (
                                "oof_availability",
                                "robustness_approved_components",
                                "component_results",
                                "trusted_component_scores",
                            ):
                                state_updates[key] = revoked_state[key]

                        if refined_exec.success:
                            if run_mode == "mlebench":
                                refined_validation = validate_model_artifacts(
                                    working_dir=working_dir,
                                    component_name=component.name,
                                    expected_n_train=expected_n_train,
                                    expected_n_test=expected_n_test,
                                    expected_class_order=expected_class_order,
                                    expected_train_ids=expected_train_ids,
                                    expected_test_ids=expected_test_ids,
                                    problem_type=problem_type,
                                    config=validation_config,
                                )
                                if _uses_packed_image_artifacts(working_dir):
                                    # Bind the CSV to the final packed test
                                    # artifact again after the child exits. A
                                    # refinement may otherwise write from
                                    # artifact A and replace the NPZ with B
                                    # before promotion.
                                    from ...utils.submission_artifacts import (
                                        rebuild_submission_from_component_predictions,
                                    )

                                    rebuild_submission_from_component_predictions(
                                        working_dir=working_dir,
                                        component_name=component.name,
                                        sample_submission_path=sample_sub_path,
                                        target_cols=target_cols,
                                        id_col=(
                                            state.get("submission_contract") or {}
                                        ).get("id_col"),
                                    )
                                refined_submission = working_dir / "submission.csv"
                                format_valid = False
                                if (
                                    refined_submission.is_file()
                                    and sample_sub_path
                                    and sample_sub_path.is_file()
                                ):
                                    format_valid, format_message = (
                                        self.executor.validate_submission_format(
                                            submission_path=refined_submission,
                                            sample_submission_path=sample_sub_path,
                                            component_type=component.component_type,
                                            problem_type=problem_type,
                                            target_cols=target_cols or None,
                                            require_normalized_rows=(
                                                needs_normalized_rows
                                            ),
                                        )
                                    )
                                    if not format_valid:
                                        print(
                                            "Refined submission validation failed: "
                                            f"{format_message}"
                                        )
                                oof_after = _oof_artifact_digest(
                                    working_dir, component.name
                                )
                                if not refined_validation.is_valid or not format_valid:
                                    refined_score = None
                                    print(
                                        "Refined candidate failed fail-closed artifact "
                                        "validation"
                                    )
                                elif oof_after is not None and oof_after == oof_before:
                                    refined_score = None
                                    print(
                                        "Refined candidate left oof_"
                                        f"{component.name}.npy untouched; the trusted "
                                        "score would describe the accepted model, not "
                                        "this one"
                                    )
                                else:
                                    refined_score = self._compute_trusted_oof_score(
                                        component, state
                                    )
                            else:
                                refined_score = self._extract_cv_score(
                                    refined_exec.stdout
                                )
                            if refined_score is not None:
                                improved_locally = (
                                    best_score is None
                                    or calculate_score_improvement(
                                        refined_score,
                                        best_score,
                                        competition_info.evaluation_metric,
                                    )
                                    > 0
                                )
                                if improved_locally:
                                    print(
                                        f"🚀 Improvement found: {refined_score:.6f} "
                                        f"(was {best_score_text})"
                                    )
                                    best_score = refined_score
                                    best_code = refined_code
                                    best_stdout = refined_exec.stdout
                                    result.code = best_code
                                    result.stdout = refined_exec.stdout
                                    state_updates["current_code"] = best_code
                                    state_updates["baseline_cv_score"] = best_score
                                    state_updates["current_performance_score"] = best_score
                                    if run_mode == "mlebench":
                                        trusted_scores = dict(
                                            state_updates.get(
                                                "trusted_component_scores",
                                                state.get(
                                                    "trusted_component_scores"
                                                )
                                                or {},
                                            )
                                        )
                                        trusted_scores[component.name] = float(
                                            refined_score
                                        )
                                        state_updates[
                                            "trusted_component_scores"
                                        ] = trusted_scores
                                    # The global best belongs to whichever
                                    # component actually holds it; a refined
                                    # candidate must beat that score, not just
                                    # its own pre-refinement score.
                                    global_best = _coerce_score(
                                        state.get("best_single_model_score")
                                    )
                                    promote_global = (
                                        global_best is None
                                        or calculate_score_improvement(
                                            refined_score,
                                            global_best,
                                            competition_info.evaluation_metric,
                                        )
                                        > 0
                                    )
                                    if promote_global:
                                        state_updates["best_single_model_score"] = best_score
                                        state_updates["best_single_model_name"] = component.name
                                        if run_mode == "mlebench":
                                            from ...utils.submission_artifacts import (
                                                snapshot_best_candidate_submission,
                                            )

                                            snapshot, digest = (
                                                snapshot_best_candidate_submission(
                                                    working_dir,
                                                    working_dir / "submission.csv",
                                                    run_id=str(state["run_id"]),
                                                    iteration=int(
                                                        state.get("current_iteration") or 0
                                                    ),
                                                )
                                            )
                                            state_updates[
                                                "best_candidate_submission_snapshot_path"
                                            ] = str(snapshot)
                                            state_updates[
                                                "best_candidate_submission_sha256"
                                            ] = digest
                                            state_updates[
                                                "best_candidate_submission_component_name"
                                            ] = component.name
                                        shutil.copy(
                                            working_dir / "submission.csv",
                                            working_dir / "submission_best.csv",
                                        )
                                    transaction_committed = True
                                else:
                                    print(
                                        f"No improvement ({refined_score:.6f} vs {best_score:.6f})"
                                    )
                            else:
                                print("Could not extract score from refined code")
                        else:
                            print("Refined code failed to execute")
                    except Exception as e:
                        print(f"Refinement failed: {e}")
                    finally:
                        if transaction is not None:
                            self._finish_candidate_transaction(
                                working_dir,
                                transaction,
                                commit=transaction_committed,
                            )

            if component.component_type == "feature_engineering":
                eng_train = working_dir / "train_engineered.csv"
                eng_test = working_dir / "test_engineered.csv"

                if eng_train.exists() and eng_test.exists():
                    # Validate that engineered data has actual features
                    try:
                        import pandas as pd
                        eng_train_df = pd.read_csv(eng_train, nrows=5)
                        eng_test_df = pd.read_csv(eng_test, nrows=5)

                        # Need at least 3 columns: id + target + 1 feature (for train)
                        # Or id + 1 feature (for test)
                        min_train_cols = 3
                        min_test_cols = 2

                        if len(eng_train_df.columns) >= min_train_cols and len(eng_test_df.columns) >= min_test_cols:
                            state_updates["current_train_path"] = str(eng_train)
                            state_updates["current_test_path"] = str(eng_test)
                            print("  🔄 Pipeline Update: Pointing subsequent agents to engineered data:")
                            print(f"     Train: {eng_train.name} ({len(eng_train_df.columns)} columns)")
                            print(f"     Test:  {eng_test.name} ({len(eng_test_df.columns)} columns)")
                        else:
                            print("  ⚠️ WARNING: Feature engineering produced insufficient columns:")
                            print(f"     Train: {len(eng_train_df.columns)} columns (need >= {min_train_cols})")
                            print(f"     Test:  {len(eng_test_df.columns)} columns (need >= {min_test_cols})")
                            print("     Keeping original train/test paths")
                    except Exception as e:
                        print(f"  ⚠️ WARNING: Failed to validate engineered data: {e}")
                        print("     Keeping original train/test paths")

        return state_updates

    # _get_refinement_iterations and _should_run_refinement are now in RefinementMixin

    def _implement_component(
        self,
        component: AblationComponent,
        state: KaggleState,
    ) -> tuple[DevelopmentResult, list[CodeAttempt]]:
        """
        Implement a single component with retry and debug.

        Args:
            component: Component to implement
            state: Current state

        Returns:
            (DevelopmentResult, attempt_records)
        """
        competition_info = state["competition_info"]
        working_dir = Path(state["working_directory"])
        domain = state.get("domain_detected", "tabular")
        attempt_records: list[CodeAttempt] = []

        # Resolve the target source and render the immutable contract FIRST:
        # before dataset inspection, GRPO/CoT reasoning, prompt composition or
        # any LLM call. A corrupt canonical claim must fail here, not after a
        # model has already been asked to write code against a contract that
        # does not hold - and a contract whose preamble already failed in this
        # run must not be regenerated or re-executed at all.
        prepared_contract = self._prepare_generated_contract(
            component,
            competition_info,
            working_dir,
            domain,
            state,
        )
        target_source = prepared_contract.target_source

        # Prefer paths discovered during data download/previous steps
        train_candidates = [
            state.get("current_train_path"),
            state.get("train_data_path"),
            str(working_dir / "train.csv"),
            str(working_dir / "train"),
            str(working_dir / "train_images"),
            str(working_dir / "images"),
            str(working_dir / "train.zip"),
        ]
        test_candidates = [
            state.get("current_test_path"),
            state.get("test_data_path"),
            str(working_dir / "test.csv"),
            str(working_dir / "test"),
            str(working_dir / "test_images"),
            str(working_dir / "images"),
            str(working_dir / "test.zip"),
        ]

        # Dynamic fallback: scan ALL subdirectories for train/test data
        # based on the actual workspace contents.
        exclude_dirs = {"models", "__pycache__", ".git", ".ipynb_checkpoints"}
        if working_dir.exists():
            for subdir in working_dir.iterdir():
                if not subdir.is_dir() or subdir.name in exclude_dirs:
                    continue
                # Add subdirectory-based candidates
                train_candidates.extend([
                    str(subdir / "train.csv"),
                    str(subdir / "train"),
                    str(subdir),  # The subdir itself may be the data source
                ])
                test_candidates.extend([
                    str(subdir / "test.csv"),
                    str(subdir / "test"),
                    # NOTE: Don't add str(subdir) here - arbitrary subdirs shouldn't count as test data.
                    # The validation logic handles the case where test doesn't exist but train is a dir.
                ])

        prefer_asset_dir = str(domain).startswith(("image", "audio"))

        def _first_existing_path(candidates: list[str | None], prefer_dir: bool) -> Path:
            existing: list[Path] = []
            for candidate in candidates:
                if not candidate:
                    continue
                path = Path(candidate)
                if path.exists():
                    existing.append(path)

            if existing:
                if prefer_dir:
                    for p in existing:
                        if p.is_dir():
                            return p
                    for p in existing:
                        if p.is_file() and p.suffix.lower() == ".zip":
                            return p
                return existing[0]

            # Fall back to first non-empty candidate to preserve error messaging
            for candidate in candidates:
                if candidate:
                    return Path(candidate)
            return Path()

        train_path = _first_existing_path(train_candidates, prefer_dir=prefer_asset_dir)
        test_path = _first_existing_path(test_candidates, prefer_dir=prefer_asset_dir)

        train_exists = train_path.exists()
        test_exists = test_path.exists()

        # Check if sample_submission exists - it contains test IDs for many competition types
        sample_sub_path = working_dir / "sample_submission.csv"
        has_sample_submission = sample_sub_path.exists()

        # Determine if we should fail due to missing data
        should_fail = False
        error_msg = ""

        if not train_exists:
            # Train data is ALWAYS required
            should_fail = True
            error_msg = f"Train data not found in {working_dir}\n"
            error_msg += f"Expected: {train_path.name}\n"
        elif not test_exists:
            # Allow proceeding without separate test data ONLY if:
            # - train_path is a DIRECTORY (not a CSV file)
            # - AND sample_submission.csv exists (contains test IDs that reference files in that directory)
            #
            # IMPORTANT: Never use train.csv as test.csv - they contain different rows!
            # Using train as test would produce invalid submissions (predictions on training data).
            if train_path.is_dir() and has_sample_submission:
                # For directory-based data (images, audio), test files may be in same dir with different IDs
                test_path = train_path
                test_exists = True
                print(f"   ℹ️ No separate test dir. Using {train_path.name}/ for both (test IDs from sample_submission.csv)")
            else:
                should_fail = True
                error_msg = f"Test data not found in {working_dir}\n"
                error_msg += f"Expected: {test_path.name}\n"
                if train_path.is_file():
                    error_msg += f"Note: Cannot use {train_path.name} as test - they must be separate files\n"

        if should_fail:
            if working_dir.exists():
                existing_items = sorted(
                    f.name + ("/" if f.is_dir() else "") for f in working_dir.iterdir()
                )
                error_msg += f"Found: {existing_items if existing_items else 'Empty dir'}\n"
            else:
                error_msg += "Working directory doesn't exist\n"

            error_msg += "\n💡 Possible causes:\n"
            error_msg += "  - Data download failed (check Kaggle credentials)\n"
            error_msg += "  - Competition data not downloaded yet\n"
            error_msg += "  - Wrong working directory path\n"

            print(f"\n❌ {error_msg}")

            return DevelopmentResult(
                code="",
                success=False,
                stdout="",
                stderr=error_msg,
                execution_time=0.0,
                artifacts_created=[],
                errors=[error_msg],
            ), attempt_records

        skip_result = self._should_skip_component(component, state)
        if skip_result is not None:
            return skip_result, attempt_records

        # GRPO: Generate reasoning trace before code generation
        reasoning_trace = None
        cot_result = None
        run_mode = str(state.get("run_mode", "")).lower()
        use_grpo = run_mode != "mlebench" and not state.get("fast_mode", False)

        if use_grpo:
            print("\n🧠 GRPO: Generating reasoning trace...")
            reasoning_trace = self._generate_reasoning_trace(component, state)

            # Validate reasoning quality
            step_scores = self._validate_reasoning(reasoning_trace, state)
            avg_score = sum(step_scores.values()) / len(step_scores) if step_scores else 0.0
            print(f"   Reasoning quality: {avg_score:.2f} (scores: {step_scores})")

            # Refine if quality is below threshold
            if avg_score < 0.6:
                reasoning_trace = self._refine_reasoning(reasoning_trace, step_scores, state)
                step_scores = self._validate_reasoning(reasoning_trace, state)
                avg_score = sum(step_scores.values()) / len(step_scores) if step_scores else 0.0
                print(f"   Refined reasoning quality: {avg_score:.2f}")

            # Store scores in trace
            reasoning_trace = reasoning_trace.__class__(
                component_name=reasoning_trace.component_name,
                requirements_analysis=reasoning_trace.requirements_analysis,
                potential_issues=reasoning_trace.potential_issues,
                solution_approach=reasoning_trace.solution_approach,
                implementation_plan=reasoning_trace.implementation_plan,
                validation_checklist=reasoning_trace.validation_checklist,
                step_scores=step_scores,
                final_score=avg_score,
                timestamp=reasoning_trace.timestamp,
            )

            # Store for state persistence
            self._last_reasoning_trace = reasoning_trace

            # Chain-of-Thought: Generate explicit step-by-step thinking
            print("\n💭 Chain-of-Thought: Step-by-step reasoning...")
            dataset_info = self._get_dataset_info(
                working_dir,
                state,
                target_source=target_source,
            )
            cot_result = self._generate_chain_of_thought(component, state, data_info=dataset_info)
            print(f"   Summary: {cot_result.thinking_summary[:100]}...")

            # Store CoT in state for debugging
            state["last_cot_thinking"] = {
                "data_analysis": cot_result.data_analysis,
                "transformation_plan": cot_result.transformation_plan,
                "model_architecture": cot_result.model_architecture,
                "validation_strategy": cot_result.validation_strategy,
                "output_format": cot_result.output_format,
                "summary": cot_result.thinking_summary,
            }

        print("\nGenerating code...")
        code = self._generate_code(
            component,
            competition_info,
            working_dir,
            domain,
            state,
            reasoning_trace=reasoning_trace,
            cot_result=cot_result,
            target_source=target_source,
            prepared_contract=prepared_contract,
        )
        # Execution metadata: the exact decision this candidate was generated
        # against, including the fingerprint of every eagerly read input.
        self._last_target_source = target_source
        self._last_target_source_metadata = target_source.execution_metadata()
        print(
            f"   Target source: {target_source.mode} "
            f"({target_source.representation_kind}, "
            f"fingerprint={target_source.target_source_fingerprint[:12]})"
        )

        # GRPO Enforcement: Verify code alignment with reasoning trace
        if use_grpo and reasoning_trace:
            print("\n🎯 GRPO Enforcement: Verifying code alignment...")
            alignment_score, missing_items = self._verify_code_alignment(
                code, reasoning_trace, state
            )
            print(f"   Alignment score: {alignment_score:.2f}")

            # If alignment is below threshold, regenerate with strict enforcement
            if alignment_score < 0.6 and missing_items:
                print(f"   ⚠️ Low alignment detected ({len(missing_items)} missing items)")
                code = self._regenerate_with_strict_enforcement(
                    original_code=code,
                    trace=reasoning_trace,
                    missing_items=missing_items,
                    component=component,
                    state=state,
                )
                # Re-verify after enforcement
                new_score, _ = self._verify_code_alignment(code, reasoning_trace, state)
                print(f"   Post-enforcement alignment: {new_score:.2f}")

        attempt_records.append(
            CodeAttempt(
                component_name=component.name,
                component_type=component.component_type,
                stage="generate",
                attempt=0,
                success=False,
                code_excerpt="\n".join(code.splitlines()[:140]),
                run_fidelity="full",
            )
        )

        if (
            self.config.ablation.enable_code_preview
            if hasattr(self.config.ablation, "enable_code_preview")
            else True
        ):
            print("\nGenerated code preview:")
            code_lines = code.split("\n")
            preview_lines = min(500, len(code_lines))
            for i, line in enumerate(code_lines[:preview_lines], 1):
                print(f"      {i:3d} | {line}")
            if len(code_lines) > preview_lines:
                print(f"      ... ({len(code_lines) - preview_lines} more lines)")
            print()

        if (
            self.config.ablation.save_generated_code
            if hasattr(self.config.ablation, "save_generated_code")
            else True
        ):
            code_file = working_dir / f"generated_code_{component.name}.py"
            try:
                code_file.write_text(code)
                print(f"Code saved to: {code_file.name}")
            except Exception:
                pass  # Continue even if save fails

        # Pre-execution validation: Check canonical data usage
        # SKIP validation if canonical data was intentionally not prepared (audio/image)
        canonical_data_prepared = state.get("canonical_data_prepared", False)
        if component.component_type in ("model", "ensemble") and canonical_data_prepared:
            try:
                from kaggle_agents.utils.data_contract import validate_canonical_data_usage

                is_valid, error_msg, warnings = validate_canonical_data_usage(
                    code, working_dir, component.component_type
                )
                if not is_valid:
                    print(f"   ❌ Canonical data validation FAILED: {error_msg}")
                    print("   Blocking execution to prevent OOF alignment issues")
                    print("   Fix: Use canonical/folds.npy and canonical/train_ids.npy instead of creating KFold")
                    # Return error result matching expected (DevelopmentResult, attempt_records) contract
                    canonical_error = f"Canonical data contract violated: {error_msg}. Models MUST use canonical folds from canonical/folds.npy."
                    return DevelopmentResult(
                        code=code,
                        success=False,
                        stdout="",
                        stderr=canonical_error,
                        execution_time=0.0,
                        artifacts_created=[],
                        errors=[canonical_error],
                    ), attempt_records
                if warnings:
                    for warning in warnings:
                        print(f"   Canonical data warning: {warning}")
            except Exception as exc:
                print(f"   Canonical data validation skipped: {exc}")
        elif component.component_type in ("model", "ensemble") and not canonical_data_prepared:
            canonical_reason = state.get("canonical_data_skipped_reason", "unknown")
            print(f"   Skipping canonical data validation ({canonical_reason})")

        is_valid, syntax_error = self.executor.validate_syntax(code)
        if not is_valid:
            print(f"Syntax error detected: {syntax_error}")
            code = self._fix_syntax_error(
                code,
                syntax_error,
                component.component_type,
                state=state,
            )

        # Quiet-STaR: ITERATIVE self-evaluation loop before execution
        use_quiet_star = run_mode != "mlebench" and not state.get("fast_mode", False)
        MAX_QUIET_STAR_ITERATIONS = 3
        CONFIDENCE_THRESHOLD = 0.7

        if use_quiet_star:
            print("\n🔮 Quiet-STaR: Iterative self-evaluation loop...")

            best_code = code
            best_confidence = 0.0

            for qs_iter in range(MAX_QUIET_STAR_ITERATIONS):
                print(f"   Iteration {qs_iter + 1}/{MAX_QUIET_STAR_ITERATIONS}")

                self_eval = self._self_evaluate_code(code, component, state)
                print(f"   Confidence: {self_eval.confidence:.2f}, Proceed: {self_eval.proceed}")

                if self_eval.concerns:
                    print(f"   Concerns: {', '.join(self_eval.concerns[:3])}")

                # Track best code version
                if self_eval.confidence > best_confidence:
                    best_confidence = self_eval.confidence
                    best_code = code

                # Store for state persistence
                self._last_self_evaluation = self_eval

                # Exit condition: confidence is good enough
                if self_eval.confidence >= CONFIDENCE_THRESHOLD and self_eval.proceed:
                    print(
                        f"   ✓ Confidence threshold reached ({self_eval.confidence:.2f} >= {CONFIDENCE_THRESHOLD})"
                    )
                    break

                # Apply fixes if available
                if self_eval.suggested_fixes:
                    print("   🔧 Applying self-evaluation fixes...")
                    code = self._apply_self_evaluation_fixes(code, self_eval, component)
                else:
                    # No fixes to apply, can't improve further
                    print("   ℹ️ No suggested fixes available, stopping iteration")
                    break

            # Use the best code version we found
            if best_confidence > self_eval.confidence:
                print(f"   Using best code version (confidence: {best_confidence:.2f})")
                code = best_code

            print(
                f"   Final Quiet-STaR confidence: {max(best_confidence, self_eval.confidence):.2f}"
            )

        print("\nExecuting code...")
        max_retries = 3
        meta_feedback: str | None = None

        # Provide runtime knobs to generated code (optional but strongly encouraged).
        run_mode = str(state.get("run_mode", "")).lower()
        objective = str(state.get("objective", ""))
        fast_mode_state = state.get("fast_mode")
        fast_mode_env_raw = os.getenv("KAGGLE_AGENTS_FAST_MODE") or os.getenv("FAST_MODE") or ""
        if fast_mode_env_raw:
            fast_mode = fast_mode_env_raw.lower() in {"1", "true", "yes"}
        elif fast_mode_state is not None:
            fast_mode = bool(fast_mode_state)
        else:
            fast_mode = run_mode == "mlebench"
        cv_folds_override = os.getenv("KAGGLE_AGENTS_CV_FOLDS")
        cv_folds: int
        state_cv_folds = state.get("cv_folds")
        if cv_folds_override:
            try:
                cv_folds = max(2, min(int(cv_folds_override), 10))
            except ValueError:
                cv_folds = (
                    5  # Increased from 2: More folds = more stable OOF predictions
                    if run_mode == "mlebench"
                    else (3 if (fast_mode or getattr(self.executor, "timeout", 0) <= 1200) else 5)
                )
        elif isinstance(state_cv_folds, int) and state_cv_folds >= 2:
            cv_folds = min(state_cv_folds, 10)
        else:
            cv_folds = (
                5  # Increased from 2: More folds = more stable OOF predictions
                if run_mode == "mlebench"
                else (3 if (fast_mode or getattr(self.executor, "timeout", 0) <= 1200) else 5)
            )
        env_overrides = {
            "KAGGLE_AGENTS_COMPONENT_TIMEOUT_S": str(getattr(self.executor, "timeout", "")),
            "KAGGLE_AGENTS_RUN_MODE": run_mode,
            "KAGGLE_AGENTS_OBJECTIVE": objective,
            "KAGGLE_AGENTS_FAST_MODE": "1" if fast_mode else "0",
            "KAGGLE_AGENTS_CV_FOLDS": str(cv_folds),
        }

        def attempt_diagnostic_score(stdout: str) -> float | None:
            """Keep candidate-declared scores out of MLE-bench memory."""
            if run_mode == "mlebench":
                return None
            return self._extract_cv_score(stdout)

        prev_env: dict[str, str | None] = {k: os.getenv(k) for k in env_overrides}
        expected_artifacts = _expected_model_artifacts(component, working_dir, run_mode)
        execution_expected_artifacts = list(expected_artifacts or [])
        if requires_submission_helper(component.component_type):
            execution_expected_artifacts.append("submission.csv")
        execution_problem_type = _model_validation_problem_type(state)
        execution_class_order = _validation_class_order_for_state(
            state,
            execution_problem_type,
        )
        if (
            component.component_type == "model"
            and execution_class_order is not None
            and _requires_class_order_artifact(
                state,
                execution_problem_type,
            )
        ):
            execution_expected_artifacts.append(
                f"models/class_order_{component.name}.npy"
            )

        for attempt in range(max_retries):
            print(f"\nAttempt {attempt + 1}/{max_retries}")

            for k, v in env_overrides.items():
                os.environ[k] = v

            # Check the artifact contract before paying for training: the
            # post-execution check would fail the same program after it has
            # already spent its whole budget, and the regenerated program then
            # trains from scratch.
            unsaved = unsaved_expected_artifacts(
                code, expected_artifacts, component.name
            )
            untrusted_helper_import = untrusted_contract_helper_import(code)
            missing_submission_helper = (
                missing_submission_helper_call(code)
                if requires_submission_helper(component.component_type)
                else False
            )
            missing_class_order = (
                any(
                    Path(path).name.startswith("class_order_")
                    for path in execution_expected_artifacts
                )
                and missing_class_order_helper_argument(code)
            )
            hand_written = (
                handwritten_submission_write(code)
                if requires_submission_helper(component.component_type)
                else None
            )
            # Report every violated contract at once. Reporting them one at a
            # time costs an attempt per round-trip, and a repair that satisfies
            # one contract by breaking another looks like progress until the
            # budget is gone: the observed loop was "omits class_order=" ->
            # "shadows an injected helper" -> success, three attempts for one
            # program.
            contract_violations: list[tuple[str, str]] = []
            if untrusted_helper_import:
                contract_violations.append(
                    (
                        "code shadows an injected contract helper "
                        f"({untrusted_helper_import.splitlines()[0]})",
                        HELPER_IMPORT_CONTRACT_ERROR,
                    )
                )
            if missing_submission_helper:
                contract_violations.append(
                    (
                        "model never calls the injected "
                        f"{_SUBMISSION_HELPER}() helper",
                        MISSING_SUBMISSION_HELPER_ERROR,
                    )
                )
            if missing_class_order:
                contract_violations.append(
                    (
                        "multiclass evidence helper call omits class_order=",
                        MISSING_CLASS_ORDER_ERROR,
                    )
                )
            if unsaved:
                contract_violations.append(
                    (
                        "code never saves " + ", ".join(unsaved),
                        f"Missing expected artifacts: {', '.join(unsaved)}",
                    )
                )
            if hand_written:
                contract_violations.append(
                    (
                        f"writes the submission by hand ({hand_written}) "
                        f"instead of calling {_SUBMISSION_HELPER}()",
                        SUBMISSION_CONTRACT_ERROR,
                    )
                )

            if contract_violations:
                print(
                    "   Skipping execution (contract check before training): "
                    f"{len(contract_violations)} violation(s)"
                )
                for summary, _ in contract_violations:
                    print(f"      - {summary}")
                # One joined string, because every consumer of a failed
                # execution reads errors[0]. A list would silently drop
                # everything after the first entry and restore the ping-pong.
                combined_error = (
                    "\n".join(error for _, error in contract_violations)
                    if len(contract_violations) == 1
                    else (
                        "This code violates "
                        f"{len(contract_violations)} contracts. Fix all of them "
                        "in one revision; fixing one by breaking another wastes "
                        "an attempt.\n"
                        + "\n".join(
                            f"{index}. {error}"
                            for index, (_, error) in enumerate(
                                contract_violations, start=1
                            )
                        )
                    )
                )
                exec_result = ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    execution_time=0.0,
                    exit_code=-1,
                    artifacts_created=[],
                    errors=[combined_error],
                )
            else:
                exec_result = execute_generated_candidate(
                    self.executor,
                    code,
                    working_dir=working_dir,
                    expected_artifacts=execution_expected_artifacts or None,
                    component_type=component.component_type,
                )
            self._write_execution_logs_and_manifest(
                component=component,
                exec_result=exec_result,
                working_dir=working_dir,
                attempt=attempt,
                expected_artifacts=execution_expected_artifacts or None,
            )
            for k, old in prev_env.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old

            if exec_result.retryable is False:
                # The failure is not about this candidate: the injected
                # preamble raised, or an input it was fingerprinted against
                # changed. Return before meta-feedback, the fixer, the
                # debugger, simplification and rollback execution - all of
                # which would rewrite code that was never the problem.
                print(
                    "   Non-retryable execution failure "
                    f"({exec_result.failure_origin}); skipping both retry "
                    "levels for this component"
                )
                attempt_records.append(
                    CodeAttempt(
                        component_name=component.name,
                        component_type=component.component_type,
                        stage="generate" if attempt == 0 else "fix",
                        attempt=attempt + 1,
                        success=False,
                        error=(
                            exec_result.errors[0][:800]
                            if exec_result.errors
                            else (exec_result.stderr or "")[:800]
                        ),
                        code_excerpt="\n".join(code.splitlines()[:140]),
                        stdout_tail=(exec_result.stdout or "")[-2000:],
                        stderr_tail=(exec_result.stderr or "")[-2000:],
                        execution_time=exec_result.execution_time,
                        run_fidelity="full",
                        failure_origin=exec_result.failure_origin,
                        retryable=False,
                        header_sha256=exec_result.header_sha256,
                        contract_fingerprint=exec_result.contract_fingerprint,
                    )
                )
                return execution_failure_to_development_result(
                    code, exec_result, "full"
                ), attempt_records

            if exec_result.success:
                print(f"Execution successful ({exec_result.execution_time:.2f}s)")

                attempt_records.append(
                    CodeAttempt(
                        component_name=component.name,
                        component_type=component.component_type,
                        stage="generate" if attempt == 0 else "fix",
                        attempt=attempt + 1,
                        success=True,
                        cv_score=attempt_diagnostic_score(exec_result.stdout),
                        code_excerpt="\n".join(code.splitlines()[:140]),
                        stdout_tail=(exec_result.stdout or "")[-2000:],
                        stderr_tail=(exec_result.stderr or "")[-2000:],
                        execution_time=exec_result.execution_time,
                        run_fidelity="full",
                        meta_feedback=meta_feedback,
                        header_sha256=exec_result.header_sha256,
                        contract_fingerprint=exec_result.contract_fingerprint,
                    )
                )

                return DevelopmentResult(
                    code=code,
                    success=True,
                    stdout=exec_result.stdout,
                    stderr=exec_result.stderr,
                    execution_time=exec_result.execution_time,
                    artifacts_created=exec_result.artifacts_created,
                    errors=[],
                ), attempt_records

            print(f"Execution failed: {exec_result.errors[0] if exec_result.errors else 'Unknown'}")

            if attempt == 0:
                if run_mode == "mlebench":
                    print(
                        "\nFormal evaluation: using deterministic execution "
                        "diagnostics without recursive meta-feedback."
                    )
                else:
                    print("\nGetting meta-evaluator feedback...")
                    error_msg = (
                        exec_result.errors[0]
                        if exec_result.errors
                        else exec_result.stderr
                    )
                    meta_feedback = self._get_meta_feedback(
                        code,
                        error_msg,
                        component.name,
                    )
                    print(f"Meta-Feedback:\n{meta_feedback}\n")

            error_msg = exec_result.errors[0] if exec_result.errors else exec_result.stderr
            attempt_records.append(
                CodeAttempt(
                    component_name=component.name,
                    component_type=component.component_type,
                    stage="generate" if attempt == 0 else "fix",
                    attempt=attempt + 1,
                    success=False,
                    cv_score=attempt_diagnostic_score(exec_result.stdout),
                    error=error_msg[:800] if error_msg else None,
                    meta_feedback=meta_feedback,
                    code_excerpt="\n".join(code.splitlines()[:140]),
                    stdout_tail=(exec_result.stdout or "")[-2000:],
                    stderr_tail=(exec_result.stderr or "")[-2000:],
                    execution_time=exec_result.execution_time,
                    run_fidelity="full",
                    failure_origin=exec_result.failure_origin,
                    retryable=exec_result.retryable,
                    header_sha256=exec_result.header_sha256,
                    contract_fingerprint=exec_result.contract_fingerprint,
                )
            )

            if attempt < max_retries - 1:
                if error_msg:
                    snippet = error_msg.replace("\n", " ")[:400]
                    print(f"Passing error context to fixer: {snippet}")
                print("Attempting to fix...")
                code = self._fix_code_error(
                    code,
                    error_msg,
                    meta_feedback=meta_feedback,
                    attempt=attempt,
                    component_type=component.component_type,
                    state=state,  # Pass state for Meta-Evaluator guidance
                    paths=getattr(self, "_resolved_paths", None),  # Pass paths for FileNotFoundError fixes
                )

        # If all retries failed, try debug iterations
        print("\nEntering debug mode...")
        debug_error_msg = exec_result.errors[0] if exec_result.errors else exec_result.stderr
        if debug_error_msg:
            snippet = debug_error_msg.replace("\n", " ")[:400]
            print(f"Last error passed to debugger: {snippet}")
        code, exec_result, debug_success = self._debug_code(
            code,
            exec_result,
            working_dir,
            max_iterations=5,
            meta_feedback=meta_feedback,
            component_name=component.name,
            component_type=component.component_type,
            state=state,  # Pass state for Meta-Evaluator guidance injection
            paths=getattr(self, "_resolved_paths", None),  # Pass paths for path-related error fixes
            expected_artifacts=execution_expected_artifacts or None,
        )

        attempt_records.append(
            CodeAttempt(
                component_name=component.name,
                component_type=component.component_type,
                stage="debug",
                attempt=max_retries + 1,
                success=bool(debug_success and exec_result.success),
                cv_score=attempt_diagnostic_score(exec_result.stdout),
                error=(exec_result.errors[0] if exec_result.errors else exec_result.stderr)[:800]
                if (exec_result.errors or exec_result.stderr)
                else None,
                meta_feedback=meta_feedback,
                code_excerpt="\n".join(code.splitlines()[:140]),
                stdout_tail=(exec_result.stdout or "")[-2000:],
                stderr_tail=(exec_result.stderr or "")[-2000:],
                execution_time=exec_result.execution_time,
                run_fidelity="debug",
                failure_origin=exec_result.failure_origin,
                retryable=exec_result.retryable,
                header_sha256=exec_result.header_sha256,
                contract_fingerprint=exec_result.contract_fingerprint,
            )
        )

        return DevelopmentResult(
            code=code,
            success=exec_result.success if debug_success else False,
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
            execution_time=exec_result.execution_time,
            artifacts_created=exec_result.artifacts_created,
            errors=exec_result.errors,
            run_fidelity="debug",
            # A debug iteration can be the one that hits a non-retryable
            # failure; the classification has to survive to the transition.
            failure_origin=exec_result.failure_origin,
            retryable=exec_result.retryable,
            header_sha256=exec_result.header_sha256,
            contract_fingerprint=exec_result.contract_fingerprint,
        ), attempt_records

    # _generate_code is now in CodeGeneratorMixin


def developer_agent_node(state: KaggleState) -> dict[str, Any]:
    """
    LangGraph node function for the developer agent.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    # Keep formal MLE-bench runs independent from globally optimized DSPy
    # pickles produced by any earlier competition in the same process.
    is_mlebench = str(state.get("run_mode", "")).strip().lower() == "mlebench"
    agent = DeveloperAgent(use_dspy=not is_mlebench)
    return agent(state)
