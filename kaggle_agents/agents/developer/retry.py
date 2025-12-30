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
from ...utils.llm_utils import get_text_content
from .code_generator import get_dynamic_temperature


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from ...optimization import PreferenceCollector
    from ...tools.code_executor import CodeExecutor, ExecutionResult


class RetryMixin:
    """Mixin providing retry and debug capabilities."""

    llm: "BaseChatModel"
    executor: "CodeExecutor"
    use_dspy: bool
    fixer_module: Any
    config: Any
    _preference_collector: "PreferenceCollector"

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

            response = self.llm.invoke(messages)
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

        # Get dynamic temperature based on attempt number
        fix_temperature = get_dynamic_temperature(
            context="fixing",
            attempt=attempt,
            component_type=component_type,
        )
        print(f"   🌡️  Fix temperature: {fix_temperature} (attempt {attempt + 1})")

        if self.use_dspy:
            result = self.fixer_module(
                code=code,
                error=error_text,
                error_type=error_info["error_type"],
            )
            fixed_code = self._extract_code_from_response(result.fixed_code)
        else:
            prompt = FIX_CODE_PROMPT.format(
                code=code,
                error=error_text,
                error_type=error_info["error_type"],
                meta_feedback=meta_feedback or "",
            )

            system_prompt = f"{DEVELOPER_CORE_IDENTITY}\n\n{HARD_CONSTRAINTS}"
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]

            # Create LLM with dynamic temperature for fixing
            fix_llm = get_llm_for_role(
                role="developer",
                temperature=fix_temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            response = fix_llm.invoke(messages)
            fixed_code = self._extract_code_from_response(get_text_content(response.content))

        return fixed_code

    def _debug_code(
        self,
        code: str,
        exec_result: "ExecutionResult",
        working_dir: Path,
        max_iterations: int = 10,
        meta_feedback: str | None = None,
        component_name: str = "",
        component_type: str = "",
        state: dict | None = None,
    ) -> tuple[str, "ExecutionResult", bool]:
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

            prompt = DEBUG_CODE_PROMPT.format(
                code=code,
                issue=issue,
                stdout=exec_result.stdout[-2000:] if exec_result.stdout else "",
                stderr=exec_result.stderr[-2000:] if exec_result.stderr else "",
                meta_feedback=meta_feedback or "",
            )

            debug_system_prompt = f"{DEVELOPER_CORE_IDENTITY}\n\n{HARD_CONSTRAINTS}\n\nYou are in DEBUG MODE. Fix the code carefully."
            messages = [
                SystemMessage(content=debug_system_prompt),
                HumanMessage(content=prompt),
            ]

            response = debug_llm.invoke(messages)
            debugged_code = self._extract_code_from_response(get_text_content(response.content))

            test_result = self.executor.execute(debugged_code, working_dir)

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

    def _generate_code(self, *args, **kwargs) -> str:
        """Placeholder - implemented in main agent class."""
        raise NotImplementedError("_generate_code must be implemented in main agent class")
