"""
Developer Agent with Code Generation and Auto-Retry.

This agent generates Python code to implement ablation components,
with automatic retry and debugging capabilities.
"""

from typing import Dict, Any, Optional
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import shutil
import os

import dspy
from langchain_core.messages import HumanMessage, SystemMessage
from ..core.state import KaggleState, AblationComponent, DevelopmentResult
from ..core.config import (
    get_config,
    get_llm_for_role,
    calculate_score_improvement,
    is_metric_minimization,
)
from ..tools.code_executor import CodeExecutor, ArtifactValidator, ExecutionResult
from ..prompts.templates.developer_prompts import (
    DEVELOPER_SYSTEM_PROMPT,
    GENERATE_CODE_PROMPT,
    FIX_CODE_PROMPT,
    DEBUG_CODE_PROMPT,
    format_component_details,
    format_error_info,
)
from ..optimization import create_optimizer


# ==================== DSPy Signatures ====================


class CodeGeneratorSignature(dspy.Signature):
    """Signature for code generation."""

    component_details: str = dspy.InputField(desc="Component to implement")
    competition_context: str = dspy.InputField(desc="Competition metadata")
    data_paths: str = dspy.InputField(desc="Paths to data files")
    requirements: str = dspy.InputField(desc="Implementation requirements")

    code: str = dspy.OutputField(desc="Complete Python code")
    explanation: str = dspy.OutputField(desc="Brief explanation of implementation")


class CodeFixerSignature(dspy.Signature):
    """Signature for code fixing."""

    code: str = dspy.InputField(desc="Code with errors")
    error: str = dspy.InputField(desc="Error message")
    error_type: str = dspy.InputField(desc="Type of error")

    fixed_code: str = dspy.OutputField(desc="Fixed Python code")
    changes_made: str = dspy.OutputField(desc="Description of fixes")


# ==================== DSPy Modules ====================


class CodeGeneratorModule(dspy.Module):
    """DSPy module for code generation."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(CodeGeneratorSignature)

    def forward(self, component_details, competition_context, data_paths, requirements):
        """Generate code."""
        result = self.generate(
            component_details=component_details,
            competition_context=competition_context,
            data_paths=data_paths,
            requirements=requirements,
        )
        return result


class CodeFixerModule(dspy.Module):
    """DSPy module for code fixing."""

    def __init__(self):
        super().__init__()
        self.fix = dspy.ChainOfThought(CodeFixerSignature)

    def forward(self, code, error, error_type):
        """Fix code."""
        result = self.fix(code=code, error=error, error_type=error_type)
        return result


# ==================== Developer Agent ====================


class DeveloperAgent:
    """
    Agent responsible for code generation and execution.

    Features:
    - Generate code from ablation components
    - Execute code in sandbox
    - Automatic retry on failure (5 attempts)
    - Debug iterations (10 max)
    - Artifact validation
    - DSPy optimization support
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
            self.generator_module = optimizer.load_optimized_prompt(
                "developer_generator"
            )
            self.fixer_module = optimizer.load_optimized_prompt("developer_fixer")

            if self.generator_module is None:
                print("Using base (unoptimized) generator module")
                self.generator_module = CodeGeneratorModule()

            if self.fixer_module is None:
                print("Using base (unoptimized) fixer module")
                self.fixer_module = CodeFixerModule()

    def __call__(self, state: KaggleState) -> Dict[str, Any]:
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

        if not ablation_plan:
            print("No ablation plan found. Run Planner Agent first.")
            return {}

        if current_index >= len(ablation_plan):
            print("All components implemented!")
            return {"current_component_index": current_index}

        component = ablation_plan[current_index]
        print(f"\n= Implementing: {component.name} ({component.component_type})")
        print(f"Estimated Impact: {component.estimated_impact:.1%}")

        base_timeout = self.config.ablation.testing_timeout
        heavy_timeout = max(base_timeout, 2700)
        ensemble_timeout = min(base_timeout, 1200) if base_timeout else 1200
        ensemble_timeout = max(ensemble_timeout, 1200)
        feature_timeout = min(base_timeout, 900) if base_timeout else 900
        feature_timeout = max(feature_timeout, 900)
        light_timeout = min(base_timeout, 300) if base_timeout else 300
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
            print(
                f"Component timeout set to: {desired_timeout}s ({desired_timeout / 60:.1f} min)"
            )

        result = self._implement_component(component, state)

        should_keep_component = True
        new_cv_score = 0.0

        if result.success and component.component_type == "model":
            from kaggle_agents.tools.code_executor import ExecutionResult

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
                return {
                    "development_results": [],
                    "current_component_index": current_index + 1,
                    "component_rollback": component.name,
                    "rollback_reason": "No CV improvement detected (Ablation Study)",
                }

        should_advance = result.success or (
            not result.success and "Data files not found" in (result.stderr or "")
        )
        state_updates = {
            "development_results": [result] if should_keep_component else [],
            "current_code": result.code,
            "code_retry_count": 0,
            "current_component_index": current_index + 1
            if should_advance
            else current_index,
            "last_updated": datetime.now(),
        }

        if result.success and component.component_type == "model":
            oof_file = working_dir / "models" / f"oof_{component.name}.npy"
            if not oof_file.exists():
                print(f"WARNING: Model {component.name} did NOT save OOF file!")
                print(f"Expected: {oof_file.name}")
                print("Stacking will fail for this model.")

            submission_path = working_dir / "submission.csv"
            if submission_path.exists():
                backup_name = f"submission_{component.name}.csv"
                backup_path = working_dir / backup_name
                shutil.copy(submission_path, backup_path)
                print(f"Backup submission saved: {backup_name}")

                current_best_score = state.get("best_single_model_score")
                metric_name = competition_info.evaluation_metric

                is_best = False
                if new_cv_score is not None:
                    if current_best_score is None:
                        is_best = True
                    else:
                        improvement = calculate_score_improvement(
                            new_cv_score, current_best_score, metric_name
                        )
                        if improvement > 0:
                            is_best = True

                if is_best:
                    print(f"New Best Single Model! ({new_cv_score:.4f})")
                    state_updates["best_single_model_score"] = new_cv_score
                    state_updates["best_single_model_name"] = component.name

                    best_path = working_dir / "submission_best.csv"
                    shutil.copy(submission_path, best_path)
                    print("Saved to submission_best.csv")
            else:
                print("Warning: submission.csv not found after successful execution")

            if result.success and should_keep_component and new_cv_score is not None:
                state_updates["baseline_cv_score"] = new_cv_score
                print(f"Updated baseline CV score: {new_cv_score:.4f}")

        if result.success and should_keep_component:
            cache_key = f"component_result_{component.name}"
            state_updates[cache_key] = result
            print(f"Cached successful result for: {component.name}")

            if (
                component.component_type == "model"
                and self.config.ablation.enable_refinement
            ):
                print("\nADK Refinement Loop: Trying to improve score...")
                best_code = result.code
                best_score = new_cv_score if new_cv_score is not None else 0.0

                for i in range(2):
                    print(f"Refinement Iteration {i + 1}/2")
                    refine_prompt = f"""
                    Current code achieved CV Score: {best_score}.
                    Analyze the hyperparameters and architecture.
                    Suggest a modification to IMPROVE the score.
                    Return the full updated code.
                    """
                    refine_component = component

                    from langchain_core.messages import HumanMessage, SystemMessage

                    refine_messages = [
                        SystemMessage(content=DEVELOPER_SYSTEM_PROMPT),
                        HumanMessage(
                            content=f"Here is the current working code:\n```python\n{best_code}\n```\n\n{refine_prompt}"
                        ),
                    ]

                    try:
                        refined_response = self.llm.invoke(refine_messages)
                        refined_code = self._extract_code_from_response(
                            refined_response.content
                        )

                        print("Executing refined code...")
                        refined_exec = self.executor.execute(refined_code, working_dir)

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
                                        f"🚀 Improvement found: {refined_score} (was {best_score})"
                                    )
                                    best_score = refined_score
                                    best_code = refined_code
                                    result.code = best_code
                                    result.stdout = refined_exec.stdout
                                    state_updates["current_code"] = best_code
                                    state_updates["baseline_cv_score"] = best_score
                                else:
                                    print(
                                        f"No improvement ({refined_score} vs {best_score})"
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
                    state_updates["current_train_path"] = str(eng_train)
                    state_updates["current_test_path"] = str(eng_test)
                    print(
                        "  🔄 Pipeline Update: Pointing subsequent agents to engineered data:"
                    )
                    print(f"     Train: {eng_train.name}")
                    print(f"     Test:  {eng_test.name}")

        return state_updates

    def _implement_component(
        self,
        component: AblationComponent,
        state: KaggleState,
    ) -> DevelopmentResult:
        """
        Implement a single component with retry and debug.

        Args:
            component: Component to implement
            state: Current state

        Returns:
            DevelopmentResult
        """
        competition_info = state["competition_info"]
        working_dir = Path(state["working_directory"])
        domain = state.get("domain_detected", "tabular")

        train_path = working_dir / "train.csv"
        test_path = working_dir / "test.csv"

        if not train_path.exists() or not test_path.exists():
            error_msg = f"Data files not found in {working_dir}\n"
            error_msg += f"Expected: {train_path.name}, {test_path.name}\n"

            if working_dir.exists():
                existing_files = [f.name for f in working_dir.iterdir() if f.is_file()]
                error_msg += (
                    f"Found: {existing_files if existing_files else 'No files'}\n"
                )
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
            )

        skip_result = self._should_skip_component(component, state)
        if skip_result is not None:
            return skip_result

        print("\nGenerating code...")
        code = self._generate_code(
            component, competition_info, working_dir, domain, state
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
            code = self._fix_syntax_error(code, syntax_error)

        print("\nExecuting code...")
        max_retries = 3
        for attempt in range(max_retries):
            print(f"\nAttempt {attempt + 1}/{max_retries}")

            exec_result = self.executor.execute(
                code=code,
                working_dir=working_dir,
            )

            if exec_result.success:
                print(f"Execution successful ({exec_result.execution_time:.2f}s)")

                return DevelopmentResult(
                    code=code,
                    success=True,
                    stdout=exec_result.stdout,
                    stderr=exec_result.stderr,
                    execution_time=exec_result.execution_time,
                    artifacts_created=exec_result.artifacts_created,
                    errors=[],
                )

            print(
                f"Execution failed: {exec_result.errors[0] if exec_result.errors else 'Unknown'}"
            )

            if attempt == 0:
                print("\nGetting meta-evaluator feedback...")
                error_msg = (
                    exec_result.errors[0] if exec_result.errors else exec_result.stderr
                )
                feedback = self._get_meta_feedback(code, error_msg, component.name)
                print(f"Meta-Feedback:\n{feedback}\n")

            if attempt < max_retries - 1:
                error_msg = (
                    exec_result.errors[0] if exec_result.errors else exec_result.stderr
                )
                if error_msg:
                    snippet = error_msg.replace("\n", " ")[:400]
                    print(f"Passing error context to fixer: {snippet}")
                print("Attempting to fix...")
                code = self._fix_code_error(code, error_msg)

        # If all retries failed, try debug iterations
        print("\nEntering debug mode...")
        debug_error_msg = (
            exec_result.errors[0] if exec_result.errors else exec_result.stderr
        )
        if debug_error_msg:
            snippet = debug_error_msg.replace("\n", " ")[:400]
            print(f"Last error passed to debugger: {snippet}")
        code, exec_result, debug_success = self._debug_code(
            code, exec_result, working_dir, max_iterations=5
        )

        return DevelopmentResult(
            code=code,
            success=exec_result.success if debug_success else False,
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
            execution_time=exec_result.execution_time,
            artifacts_created=exec_result.artifacts_created,
            errors=exec_result.errors,
        )

    def _extract_cv_score(self, stdout: str) -> Optional[float]:
        """
        Extract cross-validation score from stdout using regex patterns.

        Args:
            stdout: Standard output from code execution

        Returns:
            Extracted CV score, or None if not found
        """
        import re

        # Try multiple patterns to extract CV score
        patterns = [
            r"CV Score.*?(\d+\.\d+)",
            r"Final Validation Performance:\s*(\d+\.\d+)",
            r"ROC-AUC.*?(\d+\.\d+)",
            r"Accuracy.*?(\d+\.\d+)",
            r"RMSE.*?(\d+\.\d+)",
            r"Mean.*?(\d+\.\d+)\s*\(",  # Mean score with std
        ]

        for pattern in patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return score
                except ValueError:
                    continue

        return None

    def _validate_component_improvement(
        self,
        component: AblationComponent,
        exec_result: ExecutionResult,
        state: KaggleState,
    ) -> tuple[bool, Optional[float]]:
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

        is_minimize = is_metric_minimization(metric_name)

        cv_score = self._extract_cv_score(exec_result.stdout)

        if cv_score is None:
            print("\n   📊 Ablation Study (Hill Climbing):")
            print(
                f"      Metric:         {metric_name} ({'↓' if is_minimize else '↑'} {'minimize' if is_minimize else 'maximize'})"
            )
            print(
                "      ⚠️  No CV score found in stdout; skipping rollback and keeping component."
            )
            return True, None

        baseline_score = state.get("baseline_cv_score")
        if baseline_score is None:
            baseline_score = float("inf") if is_minimize else float("-inf")
        improvement = calculate_score_improvement(cv_score, baseline_score, metric_name)
        direction_symbol = "↓" if is_minimize else "↑"
        direction_text = "minimize" if is_minimize else "maximize"

        print("\n   📊 Ablation Study (Hill Climbing):")
        print(
            f"      Metric:         {metric_name} ({direction_symbol} {direction_text})"
        )
        print(f"      Baseline CV:    {baseline_score:.4f}")
        print(f"      Component CV:   {cv_score:.4f}")
        print(f"      Improvement:    {improvement:+.4f}")

        min_improvement = 0.001
        should_keep = improvement >= min_improvement

        if not should_keep:
            print("      ❌ Component REJECTED (no improvement or negative impact)")
            print(
                f"      Reason: Delta ({improvement:+.4f}) < threshold ({min_improvement})"
            )
        else:
            print("      ✅ Component ACCEPTED (positive improvement)")
            if baseline_score not in [float("inf"), float("-inf"), 0]:
                relative_gain = abs(improvement / baseline_score * 100)
                print(f"      Impact: {relative_gain:.2f}% relative improvement")

        return should_keep, cv_score

    def _execute_with_multi_level_retry_v2(
        self,
        component: AblationComponent,
        initial_code: str,
        working_dir: Path,
        competition_info,
        domain: str,
        state: KaggleState,
    ) -> tuple[str, bool]:
        """
        Multi-level retry with rollback (MLE-STAR pattern).

        This wraps the existing retry logic and adds Level 3: simplified rollback.
        Returns (code, success) tuple.
        """

        print("\nAttempting simplified version...")
        simplified_component = self._create_simplified_component(component)
        print(f"Simplified: {simplified_component.name}")
        simplified_code = self._generate_code(
            simplified_component,
            competition_info,
            working_dir,
            domain,
            state,
        )

        is_valid, syntax_error = self.executor.validate_syntax(simplified_code)
        if not is_valid:
            print(f"Syntax error in simplified code: {syntax_error}")
            simplified_code = self._fix_syntax_error(simplified_code, syntax_error)
        print("Executing simplified version...")
        for attempt in range(3):
            print(f"Simplified attempt {attempt + 1}/3")

            exec_result = self.executor.execute(
                code=simplified_code,
                working_dir=working_dir,
            )

            if exec_result.success:
                print("Simplified version successful!")
                return simplified_code, True

            print(
                f"Simplified attempt failed: {exec_result.errors[0] if exec_result.errors else 'Unknown'}"
            )

            if attempt < 2:
                simplified_code = self._fix_code_error(
                    simplified_code,
                    exec_result.errors[0] if exec_result.errors else exec_result.stderr,
                )

        print("❌ All retry levels exhausted (original + debug + simplified)")
        return simplified_code, False

    def _should_skip_component(
        self,
        component: AblationComponent,
        state: KaggleState,
    ) -> Optional[DevelopmentResult]:
        """
        Check if component should be skipped (MLE-STAR pattern).

        This implements callback-based skip logic to avoid redundant work:
        - Skip if code already generated and successfully executed
        - Skip if this is a refinement iteration and component worked before

        Args:
            component: Component to check
            state: Current workflow state

        Returns:
            DevelopmentResult if should skip (reuse previous result), None otherwise
        """
        dev_results = state.get("development_results", [])

        for result in dev_results:
            if result.success and component.name in result.code:
                print(f"Skipping {component.name} - already implemented successfully")
                print(f"Reusing previous execution ({result.execution_time:.2f}s)")
                return result

        cached_result_key = f"component_result_{component.name}"
        if cached_result_key in state:
            cached_result = state[cached_result_key]
            if cached_result.success:
                print(f"Skipping {component.name} - found in cache")
                print(f"Reusing cached execution ({cached_result.execution_time:.2f}s)")
                return cached_result

        return None

    def _create_simplified_component(
        self,
        component: AblationComponent,
    ) -> AblationComponent:
        """
        Create a simplified version of component for rollback (MLE-STAR pattern).

        Simplification strategies:
        - Model: Use simpler hyperparameters, fewer estimators
        - Feature engineering: Reduce complexity of features
        - Ensemble: Use simple averaging instead of stacking

        Args:
            component: Original component

        Returns:
            Simplified component
        """
        simplified_desc = ""

        if component.component_type == "model":
            model_name = component.name.split("_")[0]
            simplified_desc = f"Simple {model_name} model with basic hyperparameters: n_estimators=100, max_depth=5, learning_rate=0.1. Use default class_weight='balanced' and 5-fold StratifiedKFold."

        elif component.component_type == "feature_engineering":
            simplified_desc = "Basic feature engineering: simple polynomial features (degree 2) and basic statistical aggregations (mean, std, min, max). Avoid complex transformations."

        elif component.component_type == "ensemble":
            simplified_desc = "Simple ensemble: weighted average of model predictions with equal weights. Load predictions from submission files and average them."

        else:
            simplified_desc = f"Simplified version of {component.name}"

        simplified_component = replace(
            component,
            name=f"{component.name}_simplified",
            code=simplified_desc,
            estimated_impact=component.estimated_impact * 0.7,  # Lower expected impact
        )

        return simplified_component

    def _build_dynamic_instructions(
        self,
        component: AblationComponent,
        state: KaggleState,
    ) -> str:
        """
        Build dynamic instructions based on current state (MLE-STAR pattern).

        Creates context-aware guidance by analyzing:
        - Previous component results (what worked/failed)
        - Current iteration number (more specific in later iterations)
        - Performance trends
        - Common error patterns

        Args:
            component: Component being implemented
            state: Current workflow state

        Returns:
            Dynamic instructions string
        """
        instructions = []

        instructions.append(f"Implement {component.component_type}: {component.name}")

        current_iteration = state.get("current_iteration", 0)
        if current_iteration > 0:
            instructions.append(f"\n⚡ REFINEMENT ITERATION {current_iteration}")
            instructions.append(
                "Focus on improvements that address previous shortcomings."
            )

        refinement_guidance = state.get("refinement_guidance", {})
        if refinement_guidance and refinement_guidance.get("developer_guidance"):
            instructions.append("\nMETA-EVALUATOR GUIDANCE:")
            instructions.append(f"  {refinement_guidance['developer_guidance']}")

        if refinement_guidance and "component_type_guidance" in refinement_guidance:
            comp_guidance = refinement_guidance["component_type_guidance"].get(
                component.component_type
            )
            if comp_guidance:
                instructions.append(
                    f"\n🎯 {component.component_type.upper()} SPECIFIC GUIDANCE:"
                )
                instructions.append(f"  {comp_guidance}")

        if refinement_guidance and refinement_guidance.get("priority_fixes"):
            instructions.append("\nAVOID THESE ERROR PATTERNS:")
            for error in refinement_guidance["priority_fixes"][:3]:
                instructions.append(f"  - {error}")
        dev_results = state.get("development_results", [])
        if dev_results:
            successful_components = [r for r in dev_results if r.success]
            failed_components = [r for r in dev_results if not r.success]

            if successful_components:
                instructions.append(
                    "\n✅ SUCCESSFUL PATTERNS FROM PREVIOUS COMPONENTS:"
                )
                for i, result in enumerate(successful_components[-2:], 1):
                    if "LightGBM" in result.code:
                        instructions.append("  - LightGBM implementation worked well")
                    if "StratifiedKFold" in result.code:
                        instructions.append(
                            "  - StratifiedKFold cross-validation successful"
                        )
                    if "predict_proba" in result.code:
                        instructions.append(
                            "  - predict_proba() for probabilities confirmed working"
                        )

            if failed_components:
                instructions.append("\nAVOID THESE ERRORS FROM PREVIOUS ATTEMPTS:")
                for i, result in enumerate(failed_components[-2:], 1):
                    if result.errors:
                        error_msg = result.errors[0][:300]
                        instructions.append(f"  - {error_msg}")

        current_score = state.get("current_performance_score", 0.0)
        target_score = os.getenv("TARGET_SCORE", 0.9268)
        if current_score > 0:
            gap = target_score - current_score
            instructions.append(
                f"\nPERFORMANCE GAP: {gap:.4f} to reach target ({target_score:.4f})"
            )
            if gap < 0.01:
                instructions.append(
                    "  - Small gap: Focus on fine-tuning hyperparameters"
                )
            elif gap < 0.05:
                instructions.append(
                    "  - Medium gap: Consider feature engineering or ensemble methods"
                )
            else:
                instructions.append(
                    "  - Large gap: May need different model architecture or approach"
                )

        if component.component_type == "model":
            instructions.append("\nMODEL COMPONENT REQUIREMENTS:")
            instructions.append("  - MUST train a model and generate predictions")
            instructions.append(
                "  - MUST create submission.csv with probability predictions (0.0-1.0)"
            )
            instructions.append(
                f"  - CRITICAL: Use target_col from dataset info (target_col='{state.get('target_col', 'target')}' if available)"
            )
            instructions.append(
                "  - CRITICAL: submission column name MUST match sample_submission.columns[1] (DO NOT hardcode 'target')"
            )
            instructions.append(
                "  - CRITICAL: MUST encode categorical features (object/category dtypes) using ColumnTransformer + OneHotEncoder"
            )
            instructions.append(
                "  - CRITICAL: Never pass raw categorical strings to LightGBM/XGBoost/sklearn (will fail with 'could not convert string to float')"
            )
            instructions.append(
                "  - CatBoost is the ONLY exception that handles categorical features natively"
            )
            instructions.append(
                "  - Use OneHotEncoder(handle_unknown='ignore', sparse_output=False) (NOT sparse=...)"
            )

            instructions.append("\n🔄 CONSISTENT CROSS-VALIDATION (CRITICAL):")
            instructions.append(
                f"  - Check if '{state.get('working_directory')}/folds.csv' exists."
            )
            instructions.append(
                "  - IF EXISTS: Load it and use the 'fold' column for splitting."
            )
            instructions.append("    ```python")
            instructions.append("    folds = pd.read_csv('folds.csv')")
            instructions.append(
                "    # Assuming X is aligned with folds (reset_index if needed)"
            )
            instructions.append("    for fold in sorted(folds['fold'].unique()):")
            instructions.append("        val_idx = folds[folds['fold'] == fold].index")
            instructions.append(
                "        train_idx = folds[folds['fold'] != fold].index"
            )
            instructions.append("        # ... train/val split ...")
            instructions.append("    ```")
            instructions.append(
                "  - IF NOT EXISTS: Use StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"
            )

            instructions.append(
                "  - CRITICAL: MUST save Out-of-Fold (OOF) predictions during CV to models/oof_{component_name}.npy"
            )
            instructions.append(
                "  - OOF predictions enable proper stacking ensemble (meta-model trained on OOF)"
            )
            instructions.append(
                "  - MUST print 'Final Validation Performance: {score}'"
            )
            instructions.append(
                "  - MUST handle class imbalance with class_weight='balanced'"
            )

            instructions.append("\nSTACKING & OOF REQUIREMENTS (CRITICAL):")
            instructions.append(
                "  1. Initialize `oof_preds` array of zeros with length of train set."
            )
            instructions.append(
                "  2. Initialize `test_preds` array of zeros with length of test set."
            )
            instructions.append("  3. During CV loop:")
            instructions.append(
                "     - Fill `oof_preds[val_idx]` with predictions for validation fold."
            )
            instructions.append(
                "     - Predict on test set and accumulate: `test_preds += model.predict_proba(X_test)[:, 1] / n_folds`"
            )
            instructions.append(
                f"  4. Save OOF predictions: `np.save(str(Path('{state.get('working_directory')}') / 'models' / 'oof_{component.name}.npy'), oof_preds)`"
            )
            instructions.append(
                f"  5. Save Test predictions: `np.save(str(Path('{state.get('working_directory')}') / 'models' / 'test_{component.name}.npy'), test_preds)`"
            )
            instructions.append(
                "  6. This enables the Ensemble Agent to use Stacking later."
            )

            if (
                "optuna" in component.name.lower()
                or "tuned" in component.name.lower()
                or "optimized" in component.name.lower()
            ):
                n_trials = self.config.ablation.optuna_trials
                timeout = self.config.ablation.testing_timeout - 60

                instructions.append("\nHYPERPARAMETER OPTIMIZATION (OPTUNA) REQUIRED:")
                instructions.append(
                    "  - MUST use 'optuna' library for hyperparameter search"
                )
                instructions.append(
                    f"  - Run AT MOST {n_trials} trials (n_trials={n_trials}) and timeout={timeout}s to prevent timeouts"
                )
                instructions.append(
                    "  - CRITICAL: Check if 'optuna-integration' is available with try/except:"
                )
                instructions.append("    try:")
                instructions.append(
                    "        from optuna.integration import OptunaSearchCV"
                )
                instructions.append("    except ImportError:")
                instructions.append(
                    "        # Use manual Optuna with study.optimize() instead"
                )
                instructions.append(
                    "  - If optuna-integration is missing, use manual Optuna tuning with study.optimize()"
                )
                instructions.append("  - Use 'TPESampler' for efficient sampling")
                instructions.append(
                    "  - CRITICAL: Do NOT pass 'callbacks' or 'early_stopping_rounds' to .fit() for XGBoost/LightGBM/CatBoost sklearn API; use fixed n_estimators"
                )
                instructions.append(
                    "  - Optimize for the competition metric (minimize RMSE/LogLoss or maximize AUC/Accuracy)"
                )
                instructions.append("  - Print the best parameters found")
                instructions.append("  - Train final model with best parameters")

                instructions.append(
                    "\n⚡ SPEED OPTIMIZATION (CRITICAL TO AVOID TIMEOUT):"
                )
                instructions.append(
                    "  - **SUBSAMPLE FOR TUNING**: If train dataset > 10,000 rows:"
                )
                instructions.append("    1. Create tuning subset with train_test_split")
                instructions.append(
                    "    2. For CLASSIFICATION only: pass stratify=y when sampling (y discrete: y.nunique() < 20 or dtype category/object)"
                )
                instructions.append(
                    "    3. For REGRESSION (continuous y): DO NOT use stratify parameter"
                )
                instructions.append(
                    "    4. Run Optuna study on 25% sample (reduce to 15% if memory errors occur)"
                )
                instructions.append(
                    "    5. After finding best_params, retrain on FULL dataset"
                )
                instructions.append("  - **REDUCE ESTIMATORS DURING TUNING**:")
                instructions.append(
                    "    - Inside objective(): Use n_estimators=150-200 (fast convergence)"
                )
                instructions.append(
                    "    - Final model: Use n_estimators=1000 with early_stopping_rounds=50 (if supported)"
                )
                instructions.append(
                    "  - **TIMEOUT BUDGET**: Set study.optimize(n_trials=5, timeout=600) for max 10 min tuning"
                )
                instructions.append("  - **MEMORY SAFETY (PREVENT OOM CRASHES)**:")
                instructions.append(
                    "    - ALWAYS set n_jobs=1 in model __init__ (LGBMClassifier, XGBClassifier, etc.)"
                )
                instructions.append(
                    "    - ALWAYS set n_jobs=1 in cross_val_score (avoid nested parallelism → memory explosion)"
                )
                instructions.append(
                    "    - Add 'import gc; gc.collect()' inside objective() after computing score"
                )
                instructions.append(
                    "    - Delete model object explicitly: 'del model' before gc.collect()"
                )
                instructions.append(
                    "    - If memory errors persist, reduce train_size from 0.25 → 0.15 (15% of data)"
                )
                instructions.append(
                    "  - **ROBUST TRIALS**: Wrap objective logic in try/except; on exception log and return 0.0 so trials finish"
                )
                instructions.append(
                    "  - **NO-COMPLETION GUARD**: After study.optimize, if NO trials completed, fall back to safe default params instead of study.best_params"
                )
                instructions.append("  - **EXAMPLE PATTERN**:")
                instructions.append("    ```python")
                instructions.append("    # STEP 1: Detect GPU (CRITICAL - MANDATORY)")
                instructions.append("    import torch")
                instructions.append("    use_gpu = torch.cuda.is_available()")
                instructions.append("    print(f'GPU Available: {use_gpu}')")
                instructions.append("    if use_gpu:")
                instructions.append("        print('✅ GPU ENABLED for Optuna tuning')")
                instructions.append("    else:")
                instructions.append("        print('⚠️  CPU mode (slower)')")
                instructions.append("    ")
                instructions.append("    # Subsample for fast tuning")
                instructions.append("    if len(X) > 10000:")
                instructions.append(
                    "        # Only stratify for classification (y discrete)"
                )
                instructions.append(
                    "        is_classification = y.nunique() < 20 or y.dtype in ['object', 'category']"
                )
                instructions.append("        if is_classification:")
                instructions.append(
                    "            tune_X, _, tune_y, _ = train_test_split(X, y, train_size=0.25, stratify=y, random_state=42)"
                )
                instructions.append("        else:")
                instructions.append(
                    "            tune_X, _, tune_y, _ = train_test_split(X, y, train_size=0.25, random_state=42)"
                )
                instructions.append("    else:")
                instructions.append("        tune_X, tune_y = X, y")
                instructions.append("    ")
                instructions.append("    def objective(trial):")
                instructions.append("        params = {")
                instructions.append(
                    "            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),"
                )
                instructions.append(
                    "            'n_estimators': 150,  # Fast for tuning"
                )
                instructions.append(
                    "            'max_depth': trial.suggest_int('max_depth', 3, 10),"
                )
                instructions.append(
                    "            'n_jobs': 1,  # CRITICAL: Prevent memory explosion"
                )
                instructions.append("            # ... other params ...")
                instructions.append("        }")
                instructions.append("        ")
                instructions.append("        # STEP 2: Add GPU params (MANDATORY)")
                instructions.append("        if use_gpu:")
                instructions.append("            # For LightGBM")
                instructions.append("            params['device'] = 'gpu'")
                instructions.append("            params['gpu_platform_id'] = 0")
                instructions.append("            params['gpu_device_id'] = 0")
                instructions.append("            # For XGBoost (if using XGBoost)")
                instructions.append("            # params['tree_method'] = 'gpu_hist'")
                instructions.append("            # params['predictor'] = 'gpu_predictor'")
                instructions.append("        else:")
                instructions.append("            params['device'] = 'cpu'")
                instructions.append("            # params['tree_method'] = 'hist'  # XGBoost CPU")
                instructions.append("        ")
                instructions.append(
                    "        model = LGBMClassifier(**params, random_state=42)"
                )
                instructions.append(
                    "        # Use 3-fold CV on subsample (faster, n_jobs=1 for memory)"
                )
                instructions.append(
                    "        score = cross_val_score(model, tune_X, tune_y, cv=3, n_jobs=1, scoring='roc_auc').mean()"
                )
                instructions.append("        ")
                instructions.append("        # Free memory immediately after trial")
                instructions.append("        del model")
                instructions.append("        import gc")
                instructions.append("        gc.collect()")
                instructions.append("        ")
                instructions.append("        return score")
                instructions.append("    ")
                instructions.append(
                    "    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))"
                )
                instructions.append(
                    "    study.optimize(objective, n_trials=5, timeout=600)  # 10 min max"
                )
                instructions.append("    ")
                instructions.append(
                    "    # Train final model on FULL data with best params"
                )
                instructions.append("    best_params = study.best_params.copy()")
                instructions.append(
                    "    best_params['n_estimators'] = 1000  # More estimators for final"
                )
                instructions.append("    ")
                instructions.append("    # STEP 3: Add GPU params to final model (MANDATORY)")
                instructions.append("    if use_gpu:")
                instructions.append("        best_params['device'] = 'gpu'")
                instructions.append("        best_params['gpu_platform_id'] = 0")
                instructions.append("        best_params['gpu_device_id'] = 0")
                instructions.append("        # best_params['tree_method'] = 'gpu_hist'  # XGBoost")
                instructions.append("        # best_params['predictor'] = 'gpu_predictor'  # XGBoost")
                instructions.append("    else:")
                instructions.append("        best_params['device'] = 'cpu'")
                instructions.append("        # best_params['tree_method'] = 'hist'  # XGBoost")
                instructions.append("    ")
                instructions.append(
                    "    final_model = LGBMClassifier(**best_params, random_state=42)"
                )
                instructions.append(
                    "    # If early_stopping supported (XGBoost/LightGBM native API):"
                )
                instructions.append(
                    "    # final_model.fit(X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=50)"
                )
                instructions.append(
                    "    final_model.fit(X, y)  # Or just train on full data"
                )
                instructions.append("    ```")

        elif component.component_type == "feature_engineering":
            instructions.append("\n🔧 FEATURE ENGINEERING REQUIREMENTS:")
            instructions.append("  - Create NEW features from existing ones")
            instructions.append("  - IMPLEMENT SOTA TECHNIQUES:")
            instructions.append(
                "    - Target Encoding: MUST be done inside Cross-Validation (fit on train folds, transform val fold) to prevent leakage."
            )
            instructions.append(
                "    - Frequency Encoding: Map categorical features to their frequency/count."
            )
            instructions.append(
                "    - Aggregations: Mean/Count of numeric features grouped by categorical features."
            )
            instructions.append(
                "  - Save engineered features to file for model components"
            )
            instructions.append("  - NO model training in this component")
            instructions.append("  - Print feature importance or correlation metrics")

            instructions.append("\nFEATURE SELECTION (CRITICAL):")
            instructions.append(
                "  - After creating new features, perform selection to remove noise:"
            )
            instructions.append(
                "  1. Train a quick LightGBM/XGBoost on the new feature set."
            )
            instructions.append("  2. Calculate feature importance (gain/split).")
            instructions.append(
                "  3. Drop features with 0 importance or very low importance (< 1e-4)."
            )
            instructions.append(
                "  4. Save ONLY the selected features to 'train_engineered.csv' and 'test_engineered.csv'."
            )
            instructions.append("  5. Print list of dropped features.")
        elif component.component_type == "ensemble":
            instructions.append("\nENSEMBLE REQUIREMENTS:")
            instructions.append("  - Combine predictions from multiple models")
            instructions.append(
                "  - PREFERRED STRATEGY: Stacking Ensemble (best performance)"
            )
            instructions.append(
                "    - Load OOF predictions from models/oof_*.npy files"
            )
            instructions.append(
                "    - Stack OOF predictions: oof_stack = np.column_stack([oof1, oof2, ...])"
            )
            instructions.append(
                "    - Train meta-model (LogisticRegression/Ridge) on stacked OOF"
            )
            instructions.append(
                "    - Load test predictions from each model and stack them"
            )
            instructions.append(
                "    - Use meta-model to predict on stacked test predictions"
            )
            instructions.append("  - FALLBACK: Weighted average if OOF files missing")
            instructions.append("    - Load submission files from each model")
            instructions.append(
                "    - Combine with weights: final = w1*pred1 + w2*pred2 + ..."
            )
            instructions.append("  - Generate final submission.csv")
            instructions.append(
                f"  - CRITICAL: Use target_col from dataset info (target_col='{state.get('target_col', 'target')}' if available)"
            )
            instructions.append(
                "  - CRITICAL: submission column name MUST match sample_submission.columns[1] (DO NOT hardcode 'target' or 'prediction')"
            )
            instructions.append(
                "  - Print which models were used and their contribution/weights"
            )

        instructions.append("\nSTANDARD REQUIREMENTS:")
        instructions.append("  - Save models to models/ directory")
        instructions.append("  - Print progress and metrics throughout execution")
        instructions.append("  - NO sys.exit() or exit() calls")
        instructions.append(
            "  - CRITICAL: Do NOT use deprecated 'pandas.append()'. Use 'pd.concat()' instead."
        )
        instructions.append("  - Complete, executable single-file Python program")

        return "\n".join(instructions)

    def _get_dataset_info(self, working_dir: Path, state: KaggleState = None) -> str:
        """
        Read dataset columns and basic info to provide to LLM.

        Args:
            working_dir: Working directory containing train.csv
            state: Current state (optional)

        Returns:
            Formatted string with dataset information
        """
        try:
            import pandas as pd

            train_path = working_dir / "train.csv"

            if not train_path.exists():
                return "Dataset info not available (file not found)"

            df = pd.read_csv(train_path, nrows=5)

            columns = df.columns.tolist()
            dtypes = df.dtypes.to_dict()
            target_col = "UNKNOWN"

            if state and state.get("target_col"):
                target_col = state["target_col"]
            else:
                target_candidates = [
                    c
                    for c in columns
                    if c.lower()
                    in [
                        "target",
                        "label",
                        "y",
                        "class",
                        "loan_paid_back",
                        "survived",
                        "price",
                        "sales",
                    ]
                ]
                target_col = target_candidates[0] if target_candidates else "UNKNOWN"

            numeric_cols = [
                c for c, dtype in dtypes.items() if dtype in ["int64", "float64"]
            ]
            categorical_cols = [c for c, dtype in dtypes.items() if dtype == "object"]

            info = f"""
            **CRITICAL**: Use these EXACT column names from the dataset:

            Target Column: {target_col}
            Total Columns: {len(columns)}

            Numeric Columns ({len(numeric_cols)}): {", ".join(numeric_cols[:10])}{"..." if len(numeric_cols) > 10 else ""}
            Categorical Columns ({len(categorical_cols)}): {", ".join(categorical_cols[:10])}{"..." if len(categorical_cols) > 10 else ""}

            All Columns: {", ".join(columns)}

            IMPORTANT: Always use target_col='{target_col}' in your code!
            """
            return info

        except Exception as e:
            return f"Dataset info not available (error: {str(e)})"

    def _generate_code(
        self,
        component: AblationComponent,
        competition_info,
        working_dir: Path,
        domain: str,
        state: KaggleState = None,
    ) -> str:
        """Generate code for a component."""
        component_details = format_component_details(component)

        dataset_info = self._get_dataset_info(working_dir, state)

        competition_context = f"""
        Name: {competition_info.name}
        Domain: {domain}
        Problem Type: {competition_info.problem_type}
        Metric: {competition_info.evaluation_metric}
        """

        train_path = (
            state.get("current_train_path")
            if state and state.get("current_train_path")
            else working_dir / "train.csv"
        )
        test_path = (
            state.get("current_test_path")
            if state and state.get("current_test_path")
            else working_dir / "test.csv"
        )

        data_paths = f"""
        Train: {train_path}
        Test: {test_path}
        Models: {working_dir / "models"}
        Submission: {working_dir / "submission.csv"}
        """

        if state is not None:
            requirements = self._build_dynamic_instructions(component, state)
        else:
            requirements = f"""
            1. Implement {component.component_type}: {component.name}
            2. Save models to models/ directory
            3. Print progress and metrics
            4. Handle errors gracefully
            """

        if self.use_dspy:
            result = self.generator_module(
                component_details=component_details,
                competition_context=competition_context,
                data_paths=data_paths,
                requirements=requirements,
            )
            code = self._extract_code_from_response(result.code)
        else:
            prompt = GENERATE_CODE_PROMPT.format(
                component_details=component_details,
                competition_name=competition_info.name,
                domain=domain,
                problem_type=competition_info.problem_type,
                metric=competition_info.evaluation_metric,
                train_data_path=str(working_dir / "train.csv"),
                test_data_path=str(working_dir / "test.csv"),
                models_dir=str(working_dir / "models"),
                submission_path=str(working_dir / "submission.csv"),
                dataset_info=dataset_info,
                component_name=component.name,
            )

            messages = [
                SystemMessage(content=DEVELOPER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            code = self._extract_code_from_response(response.content)

        return code

    def _fix_syntax_error(self, code: str, error: str) -> str:
        """Fix syntax error in code."""
        return self._fix_code_error(code, f"SyntaxError: {error}")

    def _get_meta_feedback(self, code: str, error: str, component_name: str) -> str:
        """
        Get quick meta-evaluator feedback on failure (Phase 4: Mini Meta-Evaluator).

        Provides immediate strategic guidance to improve code quality.

        Args:
            code: Failed code
            error: Error message
            component_name: Name of component

        Returns:
            Strategic feedback string
        """
        # Quick analysis prompt
        prompt = f"""You are a Meta-Evaluator analyzing code failure.

        Component: {component_name}
        Error: {error[:500]}

        Code Summary (first 500 lines):
        ```python
        {chr(10).join(code.split(chr(10))[:500])}
        ```

        Provide 2-3 specific, actionable suggestions to fix this error.
        Focus on:
        1. Root cause of the error
        2. Specific code changes needed
        3. Best practices to avoid similar errors

        Keep response under 150 words."""

        try:
            messages = [
                SystemMessage(
                    content="You are an expert code reviewer and meta-evaluator."
                ),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"Meta-feedback unavailable: {str(e)}"

    def _fix_code_error(self, code: str, error: str) -> str:
        """Fix code based on error."""
        error_info = format_error_info(error)

        if self.use_dspy:
            result = self.fixer_module(
                code=code,
                error=error_info["error"],
                error_type=error_info["error_type"],
            )
            fixed_code = self._extract_code_from_response(result.fixed_code)
        else:
            prompt = FIX_CODE_PROMPT.format(
                code=code,
                error=error_info["error"],
                error_type=error_info["error_type"],
            )

            messages = [
                SystemMessage(content=DEVELOPER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            fixed_code = self._extract_code_from_response(response.content)

        return fixed_code

    def _debug_code(
        self,
        code: str,
        exec_result: ExecutionResult,
        working_dir: Path,
        max_iterations: int = 10,
    ) -> tuple[str, ExecutionResult, bool]:
        """Debug code iteratively with loop-safety and configurable timeouts."""
        original_timeout = getattr(self.executor, "timeout", None)
        # Use configurable debug_timeout (default 600s = 10 min) for Optuna tuning
        debug_timeout = self.config.ablation.debug_timeout
        if original_timeout is not None:
            self.executor.timeout = min(original_timeout, debug_timeout)
            print(f"   Debug timeout set to: {self.executor.timeout}s ({self.executor.timeout / 60:.1f} min)")

        last_error_sig = None

        for iteration in range(max_iterations):
            print(f"   Debug iteration {iteration + 1}/{max_iterations}")

            issue = f"Code failed after {iteration + 1} attempts. Errors: {', '.join(exec_result.errors)}"

            prompt = DEBUG_CODE_PROMPT.format(
                code=code,
                issue=issue,
                stdout=exec_result.stdout[-2000:] if exec_result.stdout else "",
                stderr=exec_result.stderr[-2000:] if exec_result.stderr else "",
            )

            messages = [
                SystemMessage(
                    content=DEVELOPER_SYSTEM_PROMPT
                    + "\n\nYou are in DEBUG MODE. Fix the code carefully."
                ),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            debugged_code = self._extract_code_from_response(response.content)

            test_result = self.executor.execute(debugged_code, working_dir)

            if test_result.success:
                print("Debug successful!")
                if original_timeout is not None:
                    self.executor.timeout = original_timeout
                return debugged_code, test_result, True

            error_sig = (
                "|".join(test_result.errors)
                if test_result.errors
                else test_result.stderr.strip()
            )
            if error_sig and error_sig == last_error_sig:
                print(
                    "Debug halted: same error persists; stopping to avoid infinite loop"
                )
                if original_timeout is not None:
                    self.executor.timeout = original_timeout
                return debugged_code, test_result, False

            if any("Timeout" in e for e in test_result.errors):
                print("Debug halted: repeated timeout during debug")
                if original_timeout is not None:
                    self.executor.timeout = original_timeout
                return debugged_code, test_result, False

            code = debugged_code
            exec_result = test_result
            last_error_sig = error_sig

        print("Debug failed after max iterations")
        if original_timeout is not None:
            self.executor.timeout = original_timeout
        return code, exec_result, False

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response

        return code.strip()


# ==================== LangGraph Node Function ====================


def developer_agent_node(state: KaggleState) -> Dict[str, Any]:
    """
    LangGraph node function for the developer agent.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = DeveloperAgent()
    return agent(state)
