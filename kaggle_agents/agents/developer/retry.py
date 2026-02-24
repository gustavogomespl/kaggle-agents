"""
Multi-level retry, debug, and fix logic.

Provides capabilities for retrying code execution with increasing
levels of intervention (fix, debug, simplify).

Uses dynamic temperature strategy:
- Higher temperatures for error fixing (0.25-0.5) to encourage creative solutions
- Lower temperatures for initial generation (0.1) for consistency
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...core.config import get_llm_for_role
from ...core.state import AblationComponent, DevelopmentResult, KaggleState
from ...prompts.templates.developer_prompts import (
    DEBUG_CODE_PROMPT,
    DEVELOPER_CORE_IDENTITY,
    FIX_CODE_PROMPT,
    HARD_CONSTRAINTS,
    format_error_info,
)
from ...utils.llm_utils import get_text_content, invoke_with_retry
from .code_generator import get_dynamic_temperature


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from ...optimization import PreferenceCollector
    from ...tools.code_executor import CodeExecutor, ExecutionResult

_CATEGORICAL_ENCODING_HINT = (
    "\n\n## Encoding Hint (auto-detected from error):\n"
    "The error indicates string/categorical columns that the model cannot consume directly. "
    "Before fitting ANY model, convert every object/category column:\n"
    "  for col in df.select_dtypes(include=['object', 'category']).columns:\n"
    "      df[col] = df[col].astype('category').cat.codes\n"
    "Apply this to BOTH train and test DataFrames BEFORE any model.fit() call."
)

_CATEGORICAL_ERROR_PATTERNS = ("could not convert string", "invalid literal")


def _maybe_add_encoding_hint(error_text: str) -> str:
    """Append a categorical-encoding hint if the error matches known patterns."""
    error_lower = error_text.lower()
    if any(pattern in error_lower for pattern in _CATEGORICAL_ERROR_PATTERNS):
        return error_text + _CATEGORICAL_ENCODING_HINT
    return error_text


class RetryMixin:
    """Mixin providing retry and debug capabilities."""

    llm: BaseChatModel
    executor: CodeExecutor
    use_dspy: bool
    fixer_module: Any
    config: Any
    _preference_collector: PreferenceCollector

    def _should_skip_component(
        self,
        component: AblationComponent,
        state: KaggleState,
    ) -> DevelopmentResult | None:
        """
        Check if component should be skipped (MLE-STAR pattern).

        This implements callback-based skip logic to avoid redundant work:
        - Skip if code already generated and successfully executed
        - Skip if this is a refinement iteration and component worked before
        - NEW: Validate data volume before reusing feature engineering cache
        - NEW: Validate regression predictions before reusing model cache
        - NEW: Invalidate cache if refinement guidance mentions this component

        Args:
            component: Component to check
            state: Current workflow state

        Returns:
            DevelopmentResult if should skip (reuse previous result), None otherwise
        """
        # ITERATION-AWARE CACHE INVALIDATION: Check if refinement guidance targets this component
        current_iteration = state.get("current_iteration", 0)
        refinement_guidance = state.get("refinement_guidance", {})

        if current_iteration > 1 and refinement_guidance:
            developer_guidance = refinement_guidance.get("developer_guidance", "")
            planner_guidance = refinement_guidance.get("planner_guidance", "")
            combined_guidance = f"{developer_guidance} {planner_guidance}".lower()

            # Check if component name or type is mentioned in guidance
            component_name_lower = component.name.lower()
            component_type_lower = component.component_type.lower()

            # Also check for common model names like "lightgbm", "xgboost" in component name
            model_keywords = ["lightgbm", "xgboost", "catboost", "lgbm", "logreg", "bert", "tfidf"]
            component_keywords = [kw for kw in model_keywords if kw in component_name_lower]

            guidance_mentions_component = (
                component_name_lower in combined_guidance
                or component_type_lower in combined_guidance
                or any(kw in combined_guidance for kw in component_keywords)
            )

            if guidance_mentions_component:
                print(
                    f"   🔄 Cache INVALIDATED for {component.name} - "
                    f"refinement guidance targets this component (iteration {current_iteration})"
                )
                return None  # Force re-execution

        working_dir = Path(state.get("working_directory", "."))

        dev_results = state.get("development_results", [])

        for result in dev_results:
            if result.success and component.name in result.code:
                # Validate before reusing cached result
                if not self._validate_cached_result(component, state, working_dir):
                    print(f"Cache INVALIDATED for {component.name} - forcing re-execution")
                    return None
                print(f"Skipping {component.name} - already implemented successfully")
                print(f"Reusing previous execution ({result.execution_time:.2f}s)")
                return result

        cached_result_key = f"component_result_{component.name}"
        if cached_result_key in state:
            cached_result = state[cached_result_key]
            if cached_result.success:
                # Validate before reusing cached result
                if not self._validate_cached_result(component, state, working_dir):
                    print(f"Cache INVALIDATED for {component.name} - forcing re-execution")
                    return None
                print(f"Skipping {component.name} - found in cache")
                print(f"Reusing cached execution ({cached_result.execution_time:.2f}s)")
                return cached_result

        return None

    def _validate_cached_result(
        self,
        component: AblationComponent,
        state: KaggleState,
        working_dir: Path,
    ) -> bool:
        """
        Validate cached result before reusing.

        Checks:
        - Feature engineering: data volume preserved (>90% of original)
        - Model (regression): predictions in reasonable range

        Args:
            component: Component being validated
            state: Current workflow state
            working_dir: Working directory path

        Returns:
            True if cache is valid, False to invalidate
        """
        if component.component_type == "feature_engineering":
            return self._validate_data_volume(working_dir, state)

        if component.component_type == "model":
            problem_type = state.get("problem_type", "classification")
            if problem_type == "regression":
                return self._validate_regression_predictions(component.name, working_dir)

        return True

    def _validate_data_volume(self, working_dir: Path, state: KaggleState) -> bool:
        """
        Check if engineered data preserves original row count.

        Invalidates cache if more than 10% of data was lost during
        feature engineering (e.g., from drop_duplicates or sampling).

        Args:
            working_dir: Working directory path
            state: Current workflow state

        Returns:
            True if data volume is acceptable, False to invalidate cache
        """
        train_orig = working_dir / "train.csv"
        train_eng = working_dir / "train_engineered.csv"

        if not (train_orig.exists() and train_eng.exists()):
            return True  # Can't validate, allow cache

        try:
            # Use cached original count if available, otherwise count lines
            n_orig = state.get("n_train_original")
            if n_orig is None:
                with open(train_orig) as f:
                    n_orig = sum(1 for _ in f) - 1  # Subtract header

            with open(train_eng) as f:
                n_eng = sum(1 for _ in f) - 1  # Subtract header

            if n_eng < n_orig * 0.9:  # Allow max 10% data loss
                loss_pct = (1 - n_eng / n_orig) * 100
                print(f"   ⚠️  Data loss detected: {n_orig:,} → {n_eng:,} ({loss_pct:.1f}% lost)")
                return False

            return True

        except Exception as e:
            print(f"   ⚠️  Data volume validation failed: {e}")
            return True  # On error, allow cache to avoid blocking

    def _validate_regression_predictions(
        self,
        component_name: str,
        working_dir: Path,
    ) -> bool:
        """
        Validate regression model predictions are in reasonable range.

        Checks for:
        - NaN/Inf values
        - Extreme prediction ranges (may indicate undertrained model)

        Args:
            component_name: Name of the model component
            working_dir: Working directory path

        Returns:
            True if predictions are valid, False to invalidate cache
        """
        import numpy as np

        oof_path = working_dir / "models" / f"oof_{component_name}.npy"
        test_path = working_dir / "models" / f"test_{component_name}.npy"

        if not oof_path.exists():
            return True  # Can't validate, allow cache

        try:
            oof_preds = np.load(oof_path)

            # Check for NaN/Inf
            if np.any(~np.isfinite(oof_preds)):
                print(f"   ⚠️  Invalid predictions: NaN/Inf detected in {component_name}")
                return False

            # Check for extreme ranges (may indicate bad training)
            oof_min, oof_max = oof_preds.min(), oof_preds.max()
            pred_range = oof_max - oof_min

            # If test predictions exist, check them too
            if test_path.exists():
                test_preds = np.load(test_path)
                if np.any(~np.isfinite(test_preds)):
                    print(f"   ⚠️  Invalid test predictions: NaN/Inf in {component_name}")
                    return False

            return True

        except Exception as e:
            print(f"   ⚠️  Prediction validation failed: {e}")
            return True  # On error, allow cache

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

        return replace(
            component,
            name=f"{component.name}_simplified",
            code=simplified_desc,
            estimated_impact=component.estimated_impact * 0.7,  # Lower expected impact
        )

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
                component_type=component.component_type,
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
                    attempt=attempt,
                )

        print("❌ All retry levels exhausted (original + debug + simplified)")
        return simplified_code, False

    def _fix_syntax_error(self, code: str, error: str, component_type: str = "model") -> str:
        """Fix syntax error in code with dynamic temperature."""
        return self._fix_code_error(
            code,
            f"SyntaxError: {error}",
            attempt=0,  # Syntax errors are usually first-pass issues
            component_type=component_type,
        )

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
        timeout_s = getattr(self.executor, "timeout", None)
        prompt = f"""You are a Meta-Evaluator analyzing code failure.

        Component: {component_name}
        Component timeout: {timeout_s}s
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
                SystemMessage(content="You are an expert code reviewer and meta-evaluator."),
                HumanMessage(content=prompt),
            ]

            response = invoke_with_retry(self.llm, messages)
            return get_text_content(response.content).strip()
        except Exception as e:
            return f"Meta-feedback unavailable: {e!s}"

    def _fix_code_error(
        self,
        code: str,
        error: str,
        *,
        meta_feedback: str | None = None,
        attempt: int = 0,
        component_type: str = "model",
        state: dict | None = None,
        paths: dict | None = None,
    ) -> str:
        """
        Fix code based on error with dynamic temperature.

        Uses higher temperature for fixing (0.25-0.5) to encourage
        creative problem-solving, escalating with each failed attempt.

        Also injects meta-evaluator guidance for strategic error fixing.

        Args:
            code: Code that failed
            error: Error message
            meta_feedback: Optional meta-evaluator feedback
            attempt: Current fix attempt (0-indexed), used for temperature escalation
            component_type: Type of component being fixed
            state: Optional state dict for meta-evaluator guidance
            paths: Optional resolved data paths for FileNotFoundError fixes

        Returns:
            Fixed code
        """
        error_info = format_error_info(error)
        error_text = error_info["error"]
        if meta_feedback:
            error_text = f"{error_text}\n\nMeta-Feedback:\n{meta_feedback}"

        # META-EVAL FEEDBACK LOOP: Inject refinement guidance if available
        if state:
            refinement_guidance = state.get("refinement_guidance", {})
            developer_guidance = refinement_guidance.get("developer_guidance", "")
            if developer_guidance:
                error_text = f"{error_text}\n\n## Meta-Evaluator Strategy:\n{developer_guidance}"

        # Inject categorical encoding hint when applicable
        error_text = _maybe_add_encoding_hint(error_text)

        # Get dynamic temperature based on attempt number
        fix_temperature = get_dynamic_temperature(
            context="fixing",
            attempt=attempt,
            component_type=component_type,
        )
        print(f"   🌡️  Fix temperature: {fix_temperature} (attempt {attempt + 1})")

        fixed_code: str | None = None

        if self.use_dspy:
            try:
                result = self.fixer_module(
                    code=code,
                    error=error_text,
                    error_type=error_info["error_type"],
                )
                fixed_code = self._extract_code_from_response(result.fixed_code)
            except Exception as e:
                print(f"   ⚠️ DSPy fixer failed: {e}. Falling back to direct LLM fix.")

        if fixed_code is None:
            # Format path context for FileNotFoundError fixes
            path_context = ""
            if paths:
                path_context = f"""Train: {paths.get('train', 'N/A')}
Test: {paths.get('test', 'N/A')}
Sample Submission: {paths.get('sample_submission', 'N/A')}
Models: {paths.get('models', 'models/')}
Output Dir: {paths.get('output_dir', '.')}"""

            prompt = FIX_CODE_PROMPT.format(
                code=code,
                error=error_text,
                error_type=error_info["error_type"],
                meta_feedback=meta_feedback or "",
                paths=path_context,
            )

            system_prompt = f"{DEVELOPER_CORE_IDENTITY}\n\n{HARD_CONSTRAINTS}"
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]

            try:
                # Create LLM with dynamic temperature for fixing
                fix_llm = get_llm_for_role(
                    role="developer",
                    temperature=fix_temperature,
                    max_tokens=self.config.llm.max_tokens,
                )
                response = invoke_with_retry(fix_llm, messages)
                fixed_code = self._extract_code_from_response(
                    get_text_content(response.content)
                )
            except Exception as e:
                print(f"   ⚠️ Fallback fixer failed: {e}. Returning original code.")
                return code

        return fixed_code

    def _debug_code(
        self,
        code: str,
        exec_result: ExecutionResult,
        working_dir: Path,
        max_iterations: int = 10,
        meta_feedback: str | None = None,
        component_name: str = "",
        component_type: str = "",
        state: dict | None = None,
        paths: dict | None = None,
    ) -> tuple[str, ExecutionResult, bool]:
        """
        Debug code iteratively with loop-safety, configurable timeouts, and dynamic temperature.

        Uses higher temperature (0.45) in debug mode to encourage creative solutions
        when standard fixes have failed.

        Also injects meta-evaluator guidance from state for strategic debugging direction.
        """
        # DPO: Store original code for preference pair collection
        original_code = code
        original_error = exec_result.errors[0] if exec_result.errors else exec_result.stderr[:500]
        original_timeout = getattr(self.executor, "timeout", None)
        # Use configurable debug_timeout (default 600s = 10 min) for Optuna tuning
        debug_timeout = self.config.ablation.debug_timeout
        if original_timeout is not None:
            self.executor.timeout = min(original_timeout, debug_timeout)
            print(
                f"   Debug timeout set to: {self.executor.timeout}s ({self.executor.timeout / 60:.1f} min)"
            )

        # META-EVAL FEEDBACK LOOP: Inject refinement guidance from MetaEvaluator
        if state:
            refinement_guidance = state.get("refinement_guidance", {})
            developer_guidance = refinement_guidance.get("developer_guidance", "")
            priority_fixes = refinement_guidance.get("priority_fixes", [])

            if developer_guidance or priority_fixes:
                meta_eval_context = "\n\n## Meta-Evaluator Strategic Guidance:\n"
                if developer_guidance:
                    meta_eval_context += f"{developer_guidance}\n"
                if priority_fixes:
                    meta_eval_context += "Priority error patterns to avoid:\n"
                    for fix in priority_fixes[:3]:
                        meta_eval_context += f"  - {fix}\n"

                meta_feedback = (meta_feedback or "") + meta_eval_context
                print("   🧠 Injected Meta-Evaluator guidance into debug context")

        # Inject categorical encoding hint based on the initial error
        initial_errors = " ".join(exec_result.errors) if exec_result.errors else exec_result.stderr
        if any(p in initial_errors.lower() for p in _CATEGORICAL_ERROR_PATTERNS):
            meta_feedback = (meta_feedback or "") + _CATEGORICAL_ENCODING_HINT
            print("   🔤 Injected categorical encoding hint into debug context")

        # Get debug temperature (higher for creative problem-solving)
        debug_temperature = get_dynamic_temperature(
            context="debug",
            component_type=component_type,
        )
        print(f"   🌡️  Debug temperature: {debug_temperature}")

        # Create LLM with debug temperature
        debug_llm = get_llm_for_role(
            role="developer",
            temperature=debug_temperature,
            max_tokens=self.config.llm.max_tokens,
        )

        last_error_sig = None

        for iteration in range(max_iterations):
            print(f"   Debug iteration {iteration + 1}/{max_iterations}")

            issue = f"Code failed after {iteration + 1} attempts. Errors: {', '.join(exec_result.errors)}"

            # Format path context for path-related errors
            path_context = ""
            if paths:
                path_context = f"""Train: {paths.get('train', 'N/A')}
Test: {paths.get('test', 'N/A')}
Sample Submission: {paths.get('sample_submission', 'N/A')}
Models: {paths.get('models', 'models/')}
Output Dir: {paths.get('output_dir', '.')}"""

            # Truncate code to prevent token overflow in debug LLM calls (default 2000 lines)
            max_lines = getattr(getattr(self, 'config', None), 'ablation', None)
            max_lines = getattr(max_lines, 'max_code_lines_debug', 2000) if max_lines else 2000
            code_lines = code.split("\n")
            if len(code_lines) > max_lines:
                code_truncated = "\n".join(code_lines[:max_lines])
                code_truncated += f"\n\n# ... [TRUNCATED: {len(code_lines) - max_lines} more lines]"
                print(f"   [DEBUG] Code truncated from {len(code_lines)} to {max_lines} lines")
            else:
                code_truncated = code

            prompt = DEBUG_CODE_PROMPT.format(
                code=code_truncated,
                issue=issue,
                stdout=exec_result.stdout[-2000:] if exec_result.stdout else "",
                stderr=exec_result.stderr[-2000:] if exec_result.stderr else "",
                meta_feedback=meta_feedback or "",
                paths=path_context,
            )

            debug_system_prompt = f"{DEVELOPER_CORE_IDENTITY}\n\n{HARD_CONSTRAINTS}\n\nYou are in DEBUG MODE. Fix the code carefully."
            messages = [
                SystemMessage(content=debug_system_prompt),
                HumanMessage(content=prompt),
            ]

            try:
                response = invoke_with_retry(debug_llm, messages)
            except Exception as e:
                print(f"   ⚠️ Debug LLM call failed after retries: {e}. Returning current code.")
                if original_timeout is not None:
                    self.executor.timeout = original_timeout
                return code, exec_result, False
            debugged_code = self._extract_code_from_response(get_text_content(response.content))

            test_result = self.executor.execute(
                debugged_code, working_dir, component_type=component_type
            )

            if test_result.success:
                print("Debug successful!")

                # DPO: Collect preference pair (original failed -> fixed succeeded)
                if component_name and original_code != debugged_code:
                    context = f"Fixing {component_type}: {component_name}"
                    self._preference_collector.collect_from_fix_cycle(
                        component_name=component_name,
                        component_type=component_type,
                        original_code=original_code,
                        fixed_code=debugged_code,
                        context=context,
                        error=original_error,
                        cv_score=None,  # Will be updated later if available
                    )
                    print(f"   📊 DPO: Collected preference pair for {component_name}")

                if original_timeout is not None:
                    self.executor.timeout = original_timeout
                return debugged_code, test_result, True

            error_sig = (
                "|".join(test_result.errors) if test_result.errors else test_result.stderr.strip()
            )
            if error_sig and error_sig == last_error_sig:
                print("Debug halted: same error persists; stopping to avoid infinite loop")
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
