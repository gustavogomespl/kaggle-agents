"""
Developer Agent with Code Generation and Auto-Retry.

This agent generates Python code to implement ablation components,
with automatic retry and debugging capabilities.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...core.config import (
    calculate_score_improvement,
    get_config,
    get_llm_for_role,
    is_metric_minimization,
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
from ...utils.llm_utils import get_text_content
from ...utils.log_parser import format_feedback_for_llm, parse_training_logs

# Re-export temperature utilities for backward compatibility
from .code_generator import (
    CodeGeneratorMixin,
)
from .dspy_modules import CodeFixerModule, CodeGeneratorModule
from .grpo import GRPOMixin
from .mlebench import MLEBenchMixin
from .quiet_star import QuietStarMixin
from .refinement import RefinementMixin
from .retry import RetryMixin
from .utils import DeveloperUtilsMixin
from .validation import ValidationMixin


class DeveloperAgent(
    GRPOMixin,
    QuietStarMixin,
    MLEBenchMixin,
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

        # DPO: Preference collector for learning from code fixes
        self._preference_collector = create_preference_collector()

        # Quiet-STaR: Store last self-evaluation for state persistence
        self._last_self_evaluation: SelfEvaluation | None = None

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
            "execution_time_s": exec_result.execution_time,
            "exit_code": exec_result.exit_code,
            "expected_artifacts": expected,
            "missing_expected_artifacts": missing_expected,
            "artifacts_created": exec_result.artifacts_created,
            "cv_score": self._extract_cv_score(exec_result.stdout),
            "submission_exists": (working_dir / "submission.csv").exists(),
            "oof_exists": (working_dir / "models" / f"oof_{component.name}.npy").exists(),
            "test_preds_exists": (working_dir / "models" / f"test_{component.name}.npy").exists(),
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

        if not ablation_plan:
            print("No ablation plan found. Run Planner Agent first.")
            return {}

        if current_index >= len(ablation_plan):
            print("All components implemented!")
            return {"current_component_index": current_index}

        # If a previous iteration already achieved the objective (e.g., medal in MLE-bench),
        # allow skipping remaining planned components to save time.
        if state.get("skip_remaining_components"):
            print("Skipping remaining components (skip_remaining_components=True)")
            return {"current_component_index": len(ablation_plan)}

        run_mode = str(state.get("run_mode", "")).lower()
        mlebench_grade = state.get("mlebench_grade")
        if run_mode == "mlebench" and isinstance(mlebench_grade, dict):
            if mlebench_grade.get("valid_submission") and mlebench_grade.get("gold_medal"):
                print("Skipping remaining components (GOLD medal achieved)")
                return {
                    "skip_remaining_components": True,
                    "current_component_index": len(ablation_plan),
                }

        component = ablation_plan[current_index]
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

        # Per-type caps (never exceed base_timeout)
        heavy_timeout = base_timeout
        ensemble_timeout = min(base_timeout, 1200)
        feature_timeout = min(base_timeout, 900)
        light_timeout = min(base_timeout, 300)
        name_lower = component.name.lower()

        if component.component_type == "model" or "optuna" in name_lower:
            desired_timeout = heavy_timeout
        elif component.component_type == "ensemble":
            desired_timeout = ensemble_timeout
        elif component.component_type == "feature_engineering":
            desired_timeout = feature_timeout
        else:
            desired_timeout = light_timeout if light_timeout > 0 else base_timeout

        if self.executor.timeout != desired_timeout:
            self.executor.timeout = desired_timeout
            print(f"Component timeout set to: {desired_timeout}s ({desired_timeout / 60:.1f} min)")

        result, attempt_records = self._implement_component(component, state)

        should_keep_component = True
        new_cv_score: float | None = None
        primary_score: float | None = None
        primary_score_source: str | None = None
        grading: dict[str, Any] | None = None
        force_retry = False

        if result.success and component.component_type == "model":
            exec_result = ExecutionResult(
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=result.execution_time,
                exit_code=0 if result.success else -1,
                artifacts_created=result.artifacts_created,
                errors=result.errors,
            )

            if run_mode == "mlebench":
                new_cv_score = self._extract_cv_score(result.stdout)
                should_keep_component = True
                if new_cv_score is None:
                    print(
                        "⚠️ No CV score found; skipping rollback in MLE-bench mode."
                    )
            else:
                should_keep_component, new_cv_score = self._validate_component_improvement(
                    component, exec_result, state
                )

            if not should_keep_component:
                print("\nROLLBACK: Component did not improve score - discarding")
                return {
                    "development_results": [],
                    "current_component_index": current_index + 1,
                    "component_rollback": component.name,
                    "rollback_reason": "No CV improvement detected (Ablation Study)",
                    "code_retry_count": 0,
                    "code_attempts": attempt_records,
                }

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
        skip_due_to_retries = False

        if not result.success:
            code_retry_count = max(0, code_retry_count) + 1
            if code_retry_count >= max_component_retries:
                skip_due_to_retries = True
                print(
                    f"⚠️ Max component retries reached ({code_retry_count}/{max_component_retries}) "
                    f"for {component.name}. Skipping."
                )
                # Track failed component for planner to avoid in future iterations
                # Note: state uses Annotated[list, add] reducer, so we only return the new component
                existing_failed = set(state.get("failed_component_names", []))
                if component.name not in existing_failed:
                    state["_new_failed_component"] = component.name  # Track for state_updates
                    print(f"   📝 Recorded {component.name} as failed component")

        data_not_found = not result.success and "Data files not found" in (result.stderr or "")
        should_advance = result.success or data_not_found or skip_due_to_retries
        state_updates: dict[str, Any] = {
            "development_results": [result] if should_keep_component else [],
            "current_code": result.code,
            "code_retry_count": 0 if should_advance else code_retry_count,
            "current_component_index": current_index + 1 if should_advance else current_index,
            "last_updated": datetime.now(),
            "code_attempts": attempt_records,
        }

        # Save original train row count for data loss detection
        if n_train_original_to_save is not None:
            state_updates["n_train_original"] = n_train_original_to_save

        # Persist new failed component name (add reducer will accumulate)
        if state.get("_new_failed_component"):
            state_updates["failed_component_names"] = [state["_new_failed_component"]]
            del state["_new_failed_component"]  # Clean up temp key

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
                validate_model_artifacts,
                validate_prediction_quality,
                StrictValidationConfig,
            )

            validation_config = StrictValidationConfig.from_env()

            # Get expected values from state
            expected_n_train = state.get("n_train_samples")
            expected_n_test = state.get("n_test_samples")
            expected_class_order = state.get("class_order")
            # Get problem_type from competition_info first, then fallback to state
            competition_info = state.get("competition_info")
            if competition_info and hasattr(competition_info, "problem_type") and competition_info.problem_type:
                problem_type = competition_info.problem_type
            else:
                problem_type = state.get("problem_type", "classification")

            # Run comprehensive validation
            validation_result = validate_model_artifacts(
                working_dir=working_dir,
                component_name=component.name,
                expected_n_train=expected_n_train,
                expected_n_test=expected_n_test,
                expected_class_order=expected_class_order,
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
            oof_file = working_dir / "models" / f"oof_{component.name}.npy"
            if oof_file.exists():
                try:
                    import numpy as np
                    oof_preds = np.load(oof_file)
                    is_quality_ok, quality_issues = validate_prediction_quality(
                        oof_preds, problem_type=problem_type
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

            submission_candidates = [
                Path(state.get("sample_submission_path"))
                if state.get("sample_submission_path")
                else None,
                working_dir / "submission.csv",
                working_dir / "sample_submission.csv",
            ]
            submission_path = next(
                (p for p in submission_candidates if p is not None and p.exists()),
                None,
            )
            if submission_path:
                backup_name = f"submission_{component.name}.csv"
                backup_path = working_dir / backup_name
                shutil.copy(submission_path, backup_path)
                print(f"Backup submission saved: {backup_name}")

                # Validate submission format before MLE-bench grading
                sample_sub_path = (
                    Path(state.get("sample_submission_path"))
                    if state.get("sample_submission_path")
                    else working_dir / "sample_submission.csv"
                )
                if sample_sub_path and sample_sub_path.exists():
                    is_valid, validation_msg = self.executor.validate_submission_format(
                        submission_path=submission_path,
                        sample_submission_path=sample_sub_path,
                        component_type=component.component_type,
                    )
                    if not is_valid:
                        print(f"   ❌ Submission validation FAILED: {validation_msg}")
                        # Continue to MLE-bench anyway - it will also fail but gives more info
                    else:
                        print(f"   {validation_msg}")

                # In MLE-bench mode, grade after each successful model component so we can
                # stop early once the objective is reached (medal/target/above-median).
                if run_mode == "mlebench":
                    grading = self._grade_with_mlebench(
                        competition_name=competition_info.name,
                        submission_path=submission_path,
                    )
                    state_updates["mlebench_grade"] = grading
                    score = (
                        _coerce_score(grading.get("score"))
                        if grading.get("valid_submission")
                        else None
                    )
                    if score is not None:
                        state_updates["current_performance_score"] = score
                        print(
                            f"✅ MLE-bench grade: score={score:.5f} "
                            f"above_median={bool(grading.get('above_median', False))}"
                        )

                        if self._should_stop_on_mlebench_grade(
                            grading=grading,
                            state=state,
                            metric_name=competition_info.evaluation_metric,
                        ):
                            print(
                                "🏁 Objective reached (MLE-bench) - stopping remaining components"
                            )
                            state_updates["skip_remaining_components"] = True
                            state_updates["current_component_index"] = len(ablation_plan)
                    else:
                        print(
                            f"⚠️  MLE-bench grading unavailable/invalid: {grading.get('error', 'unknown error')}"
                        )

                if run_mode == "mlebench" and isinstance(grading, dict):
                    grade_score = (
                        _coerce_score(grading.get("score"))
                        if grading.get("valid_submission")
                        else None
                    )
                    if grade_score is not None:
                        primary_score = grade_score
                        primary_score_source = "mlebench"

                if primary_score is None:
                    cv_score = _coerce_score(new_cv_score)
                    if cv_score is not None:
                        primary_score = cv_score
                        primary_score_source = "cv"

                current_best_score = state.get("best_single_model_score")
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
                        extracted = self._extract_cv_score(result.stdout)
                        score_for_gate = _coerce_score(extracted)
                        if score_for_gate is not None:
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
                        if retry_next >= max_component_retries:
                            print(
                                f"⚠️ Score {score_for_gate} ({score_source}) below threshold "
                                f"{min_component_score:.5f}, but max retries reached "
                                f"({retry_next}/{max_component_retries}). Proceeding."
                            )
                        else:
                            force_retry = True
                            state_updates["code_retry_count"] = retry_next
                            state_updates["current_component_index"] = current_index
                            state_updates["development_results"] = []
                            print(
                                f"🔄 Score {score_for_gate} ({score_source}) below threshold "
                                f"{min_component_score:.5f}; retrying component "
                                f"({retry_next}/{max_component_retries})."
                            )

                is_best = False
                if primary_score is not None:
                    if current_best_score is None:
                        is_best = True
                    else:
                        improvement = calculate_score_improvement(
                            primary_score, current_best_score, metric_name
                        )
                        if improvement > 0:
                            is_best = True

                if is_best:
                    print(f"New Best Single Model! ({primary_score:.4f})")
                    state_updates["best_single_model_score"] = primary_score
                    state_updates["best_single_model_name"] = component.name

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

            if (
                result.success
                and should_keep_component
                and (primary_score is not None or new_cv_score is not None)
                and not force_retry
            ):
                baseline_score = primary_score if primary_score is not None else new_cv_score
                baseline_candidate = _coerce_score(baseline_score)
                if baseline_candidate is not None:
                    if run_mode == "mlebench":
                        baseline_current = _coerce_score(state.get("baseline_cv_score"))
                        if baseline_current is None:
                            should_update = True
                        else:
                            improvement = calculate_score_improvement(
                                baseline_candidate, baseline_current, metric_name
                            )
                            should_update = improvement > 0
                        if should_update:
                            state_updates["baseline_cv_score"] = baseline_candidate
                            if primary_score_source == "mlebench":
                                print(
                                    f"Updated baseline MLE-bench score: {baseline_candidate:.4f}"
                                )
                            else:
                                print(
                                    f"Updated baseline CV score: {baseline_candidate:.4f}"
                                )
                    else:
                        state_updates["baseline_cv_score"] = baseline_candidate
                        print(f"Updated baseline CV score: {baseline_candidate:.4f}")

        if result.success and component.component_type in {"model", "ensemble"}:
            submission_path = working_dir / "submission.csv"
            best_submission = working_dir / "submission_best.csv"
            if run_mode == "mlebench":
                baseline_score = state.get("baseline_cv_score") or state.get(
                    "best_single_model_score"
                )
            else:
                baseline_score = state.get("baseline_cv_score") or state.get(
                    "best_single_model_score"
                )

            if run_mode == "mlebench" and grading is None and submission_path.exists():
                # Validate submission format before MLE-bench grading (fallback path)
                sample_sub_path = (
                    Path(state.get("sample_submission_path"))
                    if state.get("sample_submission_path")
                    else working_dir / "sample_submission.csv"
                )
                if sample_sub_path and sample_sub_path.exists():
                    is_valid, validation_msg = self.executor.validate_submission_format(
                        submission_path=submission_path,
                        sample_submission_path=sample_sub_path,
                        component_type=component.component_type,
                    )
                    if not is_valid:
                        print(f"   ❌ Submission validation FAILED: {validation_msg}")
                    else:
                        print(f"   {validation_msg}")

                grading = self._grade_with_mlebench(
                    competition_name=competition_info.name,
                    submission_path=submission_path,
                )
                state_updates["mlebench_grade"] = grading
                score = (
                    _coerce_score(grading.get("score"))
                    if isinstance(grading, dict) and grading.get("valid_submission")
                    else None
                )
                if score is not None:
                    state_updates["current_performance_score"] = score
                    print(
                        f"✅ MLE-bench grade: score={score:.5f} "
                        f"above_median={bool(grading.get('above_median', False))}"
                    )
                    if self._should_stop_on_mlebench_grade(
                        grading=grading,
                        state=state,
                        metric_name=metric_name,
                    ):
                        print(
                            "🏁 Objective reached (MLE-bench) - stopping remaining components"
                        )
                        state_updates["skip_remaining_components"] = True
                        state_updates["current_component_index"] = len(ablation_plan)
                elif isinstance(grading, dict):
                    print(
                        f"⚠️  MLE-bench grading unavailable/invalid: {grading.get('error', 'unknown error')}"
                    )

            if primary_score is None and run_mode == "mlebench" and isinstance(grading, dict):
                grade_score = (
                    _coerce_score(grading.get("score"))
                    if grading.get("valid_submission")
                    else None
                )
                if grade_score is not None:
                    primary_score = grade_score
                    primary_score_source = "mlebench"

            score_for_gate = primary_score
            if score_for_gate is None:
                extracted = self._extract_cv_score(result.stdout)
                score_for_gate = _coerce_score(extracted)

            if (
                isinstance(baseline_score, (int, float))
                and isinstance(score_for_gate, (int, float))
                and submission_path.exists()
                and best_submission.exists()
            ):
                is_minimize = is_metric_minimization(metric_name)
                is_worse = (
                    score_for_gate > float(baseline_score)
                    if is_minimize
                    else score_for_gate < float(baseline_score)
                )
                if is_worse:
                    shutil.copy(best_submission, submission_path)
                    print(
                        "Restored submission.csv from submission_best.csv (score worse than baseline)"
                    )
                    state_updates["submission_reverted"] = True
                    state_updates["submission_revert_reason"] = "worse_than_baseline"

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
                    print("Updated submission_best.csv with improved MLE-bench score")

        # Track OOF availability for ensemble (even if ablation study rejected the model)
        if result.success and component.component_type == "model":
            oof_file = working_dir / "models" / f"oof_{component.name}.npy"
            if oof_file.exists():
                oof_key = f"oof_available_{component.name}"
                state_updates[oof_key] = True
                print(f"   OOF file available for ensemble: {component.name}")

        if result.success and should_keep_component and not force_retry:
            cache_key = f"component_result_{component.name}"
            state_updates[cache_key] = result
            print(f"Cached successful result for: {component.name}")

            if component.component_type == "model" and self._should_run_refinement(
                component,
                state,
                new_cv_score,
                execution_time_s=result.execution_time,
                component_timeout_s=desired_timeout,
            ):
                print("\nADK Refinement Loop: Trying to improve score...")
                best_code = result.code
                best_score = new_cv_score if new_cv_score is not None else 0.0
                best_stdout = result.stdout

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

                    refine_prompt = f"""
## Current Performance
- CV Score: {best_score:.6f}

{formatted_feedback if formatted_feedback else "No structured training logs available."}

## Improvement Task
Based on the training results above, improve the model to achieve a HIGHER CV score.

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

                    system_prompt = f"{DEVELOPER_CORE_IDENTITY}\n\n{HARD_CONSTRAINTS}"
                    refine_messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(
                            content=f"Here is the current working code:\n```python\n{best_code}\n```\n\n{refine_prompt}"
                        ),
                    ]

                    try:
                        refined_response = self.llm.invoke(refine_messages)
                        refined_code = self._extract_code_from_response(
                            get_text_content(refined_response.content)
                        )

                        print("Executing refined code...")
                        refined_exec = self.executor.execute(
                            refined_code, working_dir, component_type=component.component_type
                        )

                        if refined_exec.success:
                            refined_score = self._extract_cv_score(refined_exec.stdout)
                            if refined_score is not None:
                                improvement = calculate_score_improvement(
                                    refined_score,
                                    best_score,
                                    competition_info.evaluation_metric,
                                )
                                if improvement > 0:
                                    print(
                                        f"🚀 Improvement found: {refined_score:.6f} (was {best_score:.6f})"
                                    )
                                    best_score = refined_score
                                    best_code = refined_code
                                    best_stdout = refined_exec.stdout
                                    result.code = best_code
                                    result.stdout = refined_exec.stdout
                                    state_updates["current_code"] = best_code
                                    state_updates["baseline_cv_score"] = best_score
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
        metric_name = competition_info.evaluation_metric
        working_dir = Path(state["working_directory"])
        domain = state.get("domain_detected", "tabular")
        attempt_records: list[CodeAttempt] = []

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

        # Add common non-standard directories (prioritized for audio/image competitions)
        # These are used by MLE-bench competitions like mlsp-2013-birds
        nonstandard_data_dirs = [
            "essential_data",
            "supplemental_data",
            "data",
            "audio",
            "audio_data",
            "raw_data",
        ]
        for subdir_name in nonstandard_data_dirs:
            subdir = working_dir / subdir_name
            if subdir.is_dir():
                train_candidates.append(str(subdir))
                # For audio/image competitions, test might be in same dir as train
                if str(domain).startswith(("image", "audio")):
                    test_candidates.append(str(subdir))

        # Dynamic fallback: scan ALL subdirectories for train/test data
        # This handles non-standard competition structures (e.g., essential_data/, data/)
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
            dataset_info = self._get_dataset_info(working_dir, state)
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
            except Exception as e:
                print(f"⚠️ Could not save code: {e}")

        is_valid, syntax_error = self.executor.validate_syntax(code)
        if not is_valid:
            print(f"Syntax error detected: {syntax_error}")
            code = self._fix_syntax_error(code, syntax_error, component.component_type)

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
                    2
                    if run_mode == "mlebench"
                    else (3 if (fast_mode or getattr(self.executor, "timeout", 0) <= 1200) else 5)
                )
        elif isinstance(state_cv_folds, int) and state_cv_folds >= 2:
            cv_folds = min(state_cv_folds, 10)
        else:
            cv_folds = (
                2
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
        prev_env: dict[str, str | None] = {k: os.getenv(k) for k in env_overrides}
        require_oof_env = os.getenv("KAGGLE_AGENTS_REQUIRE_OOF", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        expected_artifacts = None
        if component.component_type == "model" and require_oof_env:
            expected_artifacts = [
                f"models/oof_{component.name}.npy",
                f"models/test_{component.name}.npy",
            ]

        for attempt in range(max_retries):
            print(f"\nAttempt {attempt + 1}/{max_retries}")

            for k, v in env_overrides.items():
                os.environ[k] = v
            exec_result = self.executor.execute(
                code=code,
                working_dir=working_dir,
                expected_artifacts=expected_artifacts,
                component_type=component.component_type,
            )
            self._write_execution_logs_and_manifest(
                component=component,
                exec_result=exec_result,
                working_dir=working_dir,
                attempt=attempt,
                expected_artifacts=expected_artifacts,
            )
            for k, old in prev_env.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old

            if exec_result.success:
                print(f"Execution successful ({exec_result.execution_time:.2f}s)")

                attempt_records.append(
                    CodeAttempt(
                        component_name=component.name,
                        component_type=component.component_type,
                        stage="generate" if attempt == 0 else "fix",
                        attempt=attempt + 1,
                        success=True,
                        cv_score=self._extract_cv_score(exec_result.stdout),
                        code_excerpt="\n".join(code.splitlines()[:140]),
                        stdout_tail=(exec_result.stdout or "")[-2000:],
                        stderr_tail=(exec_result.stderr or "")[-2000:],
                        execution_time=exec_result.execution_time,
                        run_fidelity="full",
                        meta_feedback=meta_feedback,
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
                print("\nGetting meta-evaluator feedback...")
                error_msg = exec_result.errors[0] if exec_result.errors else exec_result.stderr
                meta_feedback = self._get_meta_feedback(code, error_msg, component.name)
                print(f"Meta-Feedback:\n{meta_feedback}\n")

            error_msg = exec_result.errors[0] if exec_result.errors else exec_result.stderr
            attempt_records.append(
                CodeAttempt(
                    component_name=component.name,
                    component_type=component.component_type,
                    stage="generate" if attempt == 0 else "fix",
                    attempt=attempt + 1,
                    success=False,
                    cv_score=self._extract_cv_score(exec_result.stdout),
                    error=error_msg[:800] if error_msg else None,
                    meta_feedback=meta_feedback,
                    code_excerpt="\n".join(code.splitlines()[:140]),
                    stdout_tail=(exec_result.stdout or "")[-2000:],
                    stderr_tail=(exec_result.stderr or "")[-2000:],
                    execution_time=exec_result.execution_time,
                    run_fidelity="full",
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
        )

        attempt_records.append(
            CodeAttempt(
                component_name=component.name,
                component_type=component.component_type,
                stage="debug",
                attempt=max_retries + 1,
                success=bool(debug_success and exec_result.success),
                cv_score=self._extract_cv_score(exec_result.stdout),
                error=(exec_result.errors[0] if exec_result.errors else exec_result.stderr)[:800]
                if (exec_result.errors or exec_result.stderr)
                else None,
                meta_feedback=meta_feedback,
                code_excerpt="\n".join(code.splitlines()[:140]),
                stdout_tail=(exec_result.stdout or "")[-2000:],
                stderr_tail=(exec_result.stderr or "")[-2000:],
                execution_time=exec_result.execution_time,
                run_fidelity="debug",
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
    agent = DeveloperAgent()
    return agent(state)
