"""
Refinement guidance generation for Meta-Evaluator.

Contains methods for generating strategic guidance for prompt optimization (PREFACE pattern).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...utils.llm_utils import get_text_content
from .prompts import META_EVALUATOR_SYSTEM_PROMPT


if TYPE_CHECKING:
    from ...core.state import KaggleState


class GuidanceMixin:
    """Mixin providing refinement guidance generation methods."""

    def _generate_refinement_guidance(
        self,
        state: KaggleState,
        failure_analysis: dict[str, Any],
        reward_signals: dict[str, float],
    ) -> dict[str, str]:
        """
        Generate refinement guidance for prompt optimization (PREFACE pattern).

        Uses LLM to analyze failures and generate strategic guidance
        for improving prompts in next iteration. Integrates semantic
        log analysis for deeper insights.

        Args:
            state: Current workflow state
            failure_analysis: Failure analysis results
            reward_signals: Calculated rewards

        Returns:
            Refinement guidance dictionary
        """
        print("\n   🎯 Generating refinement guidance...")

        # Analyze execution logs for semantic errors (LLM-driven)
        log_analysis = self._analyze_execution_logs(state)

        # Detect undertrained models (critical for image/multiclass problems)
        undertrained_info = self._detect_undertrained_models(state)

        # Build context for LLM
        context = self._build_evaluation_context(state, failure_analysis, reward_signals)

        # Inject undertrained model detection into context (highest priority)
        if undertrained_info:
            context += "\n\n## ⚠️ CRITICAL: UNDERTRAINED MODEL DETECTED\n"
            context += f"**Severity**: {undertrained_info.get('severity', 'critical')}\n"
            context += f"**Message**: {undertrained_info.get('message', '')}\n"
            context += f"**CV Score**: {undertrained_info.get('cv_score', 0):.4f}\n"
            context += f"**Random Baseline**: {undertrained_info.get('random_baseline', 0):.4f}\n"
            context += f"**Classes**: {undertrained_info.get('n_classes', 2)}\n\n"
            context += "**Suggestions**:\n"
            for sugg in undertrained_info.get("suggestions", []):
                context += f"  - {sugg}\n"

        # Inject semantic analysis into context
        if log_analysis.get("has_semantic_errors"):
            context += "\n\n## Semantic Log Analysis (from LLM)\n"
            context += f"**Severity**: {log_analysis.get('severity', 'unknown')}\n"
            context += f"**Summary**: {log_analysis.get('summary', '')}\n\n"

            for issue in log_analysis.get("detected_issues", [])[:5]:
                context += f"### Issue: `{issue.get('pattern', 'Unknown')}`\n"
                context += f"- **Root Cause**: {issue.get('root_cause', '')}\n"
                context += f"- **Diagnosis**: {issue.get('diagnosis', '')}\n"
                context += "- **Solutions**:\n"
                for sol in issue.get("solutions", []):
                    context += f"  - {sol}\n"
                context += "\n"

            if log_analysis.get("planner_directives"):
                context += "**Planner Directives (from log analysis)**:\n"
                for directive in log_analysis["planner_directives"]:
                    context += f"- {directive}\n"

            if log_analysis.get("developer_directives"):
                context += "\n**Developer Directives (from log analysis)**:\n"
                for directive in log_analysis["developer_directives"]:
                    context += f"- {directive}\n"

        # Generate guidance using LLM
        prompt = self._build_refinement_prompt(context)

        messages = [
            SystemMessage(content=META_EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

        # Parse guidance from response
        try:
            guidance = json.loads(get_text_content(response.content))
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            guidance = {
                "planner_guidance": "Focus on high-impact components with proven track record.",
                "developer_guidance": "Ensure code follows all requirements and outputs correct format.",
                "priority_fixes": failure_analysis["error_patterns"],
            }

        # Inject semantic directives into guidance (high priority)
        if log_analysis.get("planner_directives"):
            semantic_guidance = " | ".join(log_analysis["planner_directives"])
            existing = guidance.get("planner_guidance", "")
            guidance["planner_guidance"] = f"PRIORITY: {semantic_guidance}. {existing}"

        if log_analysis.get("developer_directives"):
            existing_dev = guidance.get("developer_guidance", "")
            dev_directives = " | ".join(log_analysis["developer_directives"])
            guidance["developer_guidance"] = f"PRIORITY: {dev_directives}. {existing_dev}"

        # Inject undertrained model directives (HIGHEST priority - overrides everything)
        if undertrained_info:
            existing_planner = guidance.get("planner_guidance", "")
            guidance["planner_guidance"] = f"🔴 {undertrained_info.get('planner_directive', '')} {existing_planner}"

            existing_dev = guidance.get("developer_guidance", "")
            guidance["developer_guidance"] = f"🔴 {undertrained_info.get('developer_directive', '')} {existing_dev}"

            guidance["undertrained_analysis"] = undertrained_info

        # Store full analysis for downstream use
        guidance["semantic_analysis"] = log_analysis

        print("   ✓ Generated guidance for Planner and Developer")

        return guidance

    def _build_evaluation_context(
        self,
        state: KaggleState,
        failure_analysis: dict[str, Any],
        reward_signals: dict[str, float],
    ) -> str:
        """Build context string for LLM evaluation."""
        current_iteration = state.get("current_iteration", 0)
        run_mode = str(state.get("run_mode", "")).lower()
        objective = str(state.get("objective", "")).lower()
        mlebench_grade = state.get("mlebench_grade")

        current_score = state.get("current_performance_score", 0.0)
        if isinstance(mlebench_grade, dict) and mlebench_grade.get("valid_submission"):
            score = mlebench_grade.get("score")
            if isinstance(score, (int, float)):
                current_score = float(score)

        target_score = state.get("target_score")
        if target_score is None:
            target_score = 1.0
        elif isinstance(target_score, str):
            try:
                target_score = float(target_score)
            except (ValueError, TypeError):
                target_score = 1.0
        elif not isinstance(target_score, (int, float)):
            target_score = 1.0

        context = f"""# Iteration {current_iteration} Evaluation

## Objective
- run_mode: {run_mode or "kaggle"}
- objective: {objective or "top20"}

## Current Performance
- Score: {current_score:.4f}
- Target: {target_score:.4f}
- Gap: {target_score - current_score:.4f}

## Component Results
- Total: {len(state.get("development_results", []))}
- Successful: {len(failure_analysis["success_components"])}
- Failed: {len(failure_analysis["failed_components"])}

## Success Patterns
{chr(10).join("- " + p for p in failure_analysis["success_patterns"])}

## Error Patterns
{chr(10).join("- " + p for p in failure_analysis["error_patterns"])}
"""

        if isinstance(mlebench_grade, dict):
            medals = []
            if mlebench_grade.get("gold_medal"):
                medals.append("Gold")
            if mlebench_grade.get("silver_medal"):
                medals.append("Silver")
            if mlebench_grade.get("bronze_medal"):
                medals.append("Bronze")
            context += f"""
## MLE-bench Grading
- valid_submission: {bool(mlebench_grade.get("valid_submission", False))}
- score: {mlebench_grade.get("score")}
- above_median: {bool(mlebench_grade.get("above_median", False))}
- medals: {", ".join(medals) if medals else "None"}
"""

        # FULL CODE AND PERFORMANCE ANALYSIS
        context += "\n## Component Code and Performance Analysis\n"

        dev_results = state.get("development_results", [])

        # Limit to 5 most recent components to reduce token usage
        recent_results = dev_results[-5:] if len(dev_results) > 5 else dev_results
        if len(dev_results) > 5:
            context += f"*(Showing 5 most recent components out of {len(dev_results)} total)*\n\n"

        for i, res in enumerate(recent_results):
            # Extract score from stdout if possible
            score_match = re.search(r"Final Validation Performance: (0\.\d+)", res.stdout)
            score = float(score_match.group(1)) if score_match else "N/A"

            # Determine component name (heuristic)
            comp_name = f"Component_{i + 1}"
            if "class " in res.code:
                # Try to find class name
                class_match = re.search(r"class (\w+)", res.code)
                if class_match:
                    comp_name = class_match.group(1)

            status = "✅ Success" if res.success else "❌ Failed"

            context += f"\n### {comp_name}\n"
            context += f"**Status**: {status}\n"
            context += f"**Score**: {score}\n"
            context += f"**Execution Time**: {res.execution_time:.2f}s\n"

            if not res.success:
                context += f"**Error**: {res.stderr[-200:] if res.stderr else 'Unknown Error'}\n"

            # Send code summary instead of full code to reduce tokens
            code_lines = res.code.split("\n")
            context += "**Code Summary**:\n```python\n"
            context += "\n".join(code_lines[:20])  # First 20 lines
            if len(code_lines) > 30:
                context += "\n# ... (middle section omitted) ...\n"
                context += "\n".join(code_lines[-10:])  # Last 10 lines
            context += "\n```\n"
            context += f"**Total Lines**: {len(code_lines)}\n"
            stdout_tail = res.stdout[-3000:] if res.stdout else ""
            stderr_tail = res.stderr[-1500:] if res.stderr else ""
            if stdout_tail:
                context += "**STDOUT (tail)**:\n```text\n"
                context += stdout_tail
                context += "\n```\n"
            if stderr_tail:
                context += "**STDERR (tail)**:\n```text\n"
                context += stderr_tail
                context += "\n```\n"
            context += "-" * 40 + "\n"

        context += "\n## Reward Signals\n"
        for key, value in reward_signals.items():
            context += f"- {key}: {value:.3f}\n"

        return context

    def _build_refinement_prompt(self, context: str) -> str:
        """Build prompt for refinement guidance generation."""
        return f"""{context}

## Your Task
Analyze the above results and provide strategic guidance for improving prompts in the next iteration.

If the objective is `mlebench_medal`, explicitly prioritize:
- Achieving at least a Bronze medal (then Silver/Gold)
- Reducing wall-clock execution time (avoid expensive CV / over-sized models)
- Robustness and deterministic outputs (no flaky dependencies)

Return a JSON object with:
{{
  "planner_guidance": "Specific guidance for Planner agent on how to improve component selection",
  "developer_guidance": "Specific guidance for Developer agent on how to avoid errors",
  "priority_fixes": ["error_type_1", "error_type_2"],
  "success_amplification": ["what worked that should be emphasized"],
  "component_type_guidance": {{
    "model": "guidance for model components",
    "feature_engineering": "guidance for feature engineering",
    "ensemble": "guidance for ensemble components"
  }}
}}

Focus on actionable, specific improvements based on error patterns and performance gaps.
"""
