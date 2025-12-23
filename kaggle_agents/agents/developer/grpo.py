"""
GRPO: Reasoning-First Code Generation.

Implements Group Relative Policy Optimization style reasoning traces
that are generated before code generation to improve code quality.
"""

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...core.state import AblationComponent, KaggleState, ReasoningTrace
from ...utils.llm_utils import get_text_content

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class GRPOMixin:
    """Mixin providing GRPO reasoning capabilities."""

    llm: "BaseChatModel"

    def _generate_reasoning_trace(
        self,
        component: AblationComponent,
        state: KaggleState,
    ) -> ReasoningTrace:
        """
        GRPO-style: Generate structured reasoning trace before code generation.

        Creates a chain-of-thought analysis that:
        1. Analyzes requirements
        2. Identifies potential issues
        3. Plans implementation approach
        4. Defines validation checklist

        Args:
            component: Component to implement
            state: Current workflow state

        Returns:
            ReasoningTrace with structured analysis
        """
        competition_info = state.get("competition_info")
        domain = state.get("domain_detected", "tabular")
        failure_analysis = state.get("failure_analysis", {})
        known_errors = failure_analysis.get("error_patterns", [])

        prompt = f"""Before implementing {component.name} ({component.component_type}), analyze the task:

COMPETITION: {competition_info.name if competition_info else 'Unknown'}
DOMAIN: {domain}
METRIC: {competition_info.evaluation_metric if competition_info else 'Unknown'}
COMPONENT: {component.name}
TYPE: {component.component_type}
DESCRIPTION: {component.code[:500] if component.code else 'No description'}

KNOWN ERROR PATTERNS TO AVOID: {', '.join(known_errors) if known_errors else 'None'}

Provide a structured analysis in JSON format:
{{
    "requirements_analysis": "What exactly this component needs to do (2-3 sentences)",
    "potential_issues": ["Issue 1 that could cause failure", "Issue 2", "Issue 3"],
    "solution_approach": "High-level technical approach (2-3 sentences)",
    "implementation_plan": "Step 1: ...\\nStep 2: ...\\nStep 3: ...",
    "validation_checklist": ["Check 1", "Check 2", "Check 3", "Check 4"]
}}

Be specific and actionable. Focus on preventing common failures."""

        messages = [
            SystemMessage(content="You are an expert ML engineer analyzing implementation requirements."),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = get_text_content(response.content).strip()

            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {}

            return ReasoningTrace(
                component_name=component.name,
                requirements_analysis=result.get("requirements_analysis", ""),
                potential_issues=result.get("potential_issues", []),
                solution_approach=result.get("solution_approach", ""),
                implementation_plan=result.get("implementation_plan", ""),
                validation_checklist=result.get("validation_checklist", []),
                step_scores={},
                final_score=0.0,
                timestamp=datetime.now(),
            )

        except Exception as e:
            print(f"   ⚠️ Reasoning trace generation failed: {e}")
            return ReasoningTrace(
                component_name=component.name,
                requirements_analysis=f"Implement {component.component_type}: {component.name}",
                potential_issues=known_errors[:3] if known_errors else [],
                solution_approach="Standard implementation approach",
                implementation_plan="",
                validation_checklist=["Verify output format", "Check for errors"],
                step_scores={},
                final_score=0.0,
                timestamp=datetime.now(),
            )

    def _validate_reasoning(
        self,
        trace: ReasoningTrace,
        state: KaggleState,
    ) -> dict[str, float]:
        """
        GRPO-style: Validate reasoning quality with process rewards.

        Evaluates:
        - Issue coverage: Does reasoning address known error patterns?
        - Plan specificity: Is the implementation plan concrete?
        - Validation quality: Are validation checks comprehensive?

        Args:
            trace: Reasoning trace to validate
            state: Current workflow state

        Returns:
            Dictionary of step scores
        """
        scores = {}

        failure_analysis = state.get("failure_analysis", {})
        known_errors = failure_analysis.get("error_patterns", [])

        # 1. Issue Coverage Score (0-1)
        if known_errors and trace.potential_issues:
            issues_text = " ".join(trace.potential_issues).lower()
            covered = sum(1 for e in known_errors if e.lower().replace("_", " ") in issues_text)
            scores["issue_coverage"] = min(covered / max(len(known_errors), 1), 1.0)
        else:
            scores["issue_coverage"] = 0.5 if trace.potential_issues else 0.2

        # 2. Plan Specificity Score (0-1)
        if trace.implementation_plan:
            steps = trace.implementation_plan.split("\n")
            concrete_steps = [s for s in steps if len(s.strip()) > 10]
            scores["plan_specificity"] = min(len(concrete_steps) / 5, 1.0)
        else:
            scores["plan_specificity"] = 0.0

        # 3. Validation Quality Score (0-1)
        if trace.validation_checklist:
            scores["validation_quality"] = min(len(trace.validation_checklist) / 4, 1.0)
        else:
            scores["validation_quality"] = 0.0

        # 4. Requirements Clarity Score (0-1)
        if trace.requirements_analysis:
            scores["requirements_clarity"] = min(len(trace.requirements_analysis) / 200, 1.0)
        else:
            scores["requirements_clarity"] = 0.0

        # 5. Solution Approach Score (0-1)
        if trace.solution_approach:
            scores["solution_approach"] = min(len(trace.solution_approach) / 150, 1.0)
        else:
            scores["solution_approach"] = 0.0

        return scores

    def _refine_reasoning(
        self,
        trace: ReasoningTrace,
        step_scores: dict[str, float],
        state: KaggleState,
    ) -> ReasoningTrace:
        """
        GRPO-style: Refine reasoning trace based on process reward scores.

        Args:
            trace: Original reasoning trace
            step_scores: Scores for each reasoning step
            state: Current workflow state

        Returns:
            Refined reasoning trace
        """
        weak_areas = [k for k, v in step_scores.items() if v < 0.5]

        if not weak_areas:
            return trace

        print(f"   🔄 Refining reasoning (weak areas: {weak_areas})")

        refinement_prompt = f"""The following reasoning trace needs improvement in: {', '.join(weak_areas)}

ORIGINAL TRACE:
- Requirements: {trace.requirements_analysis}
- Issues: {trace.potential_issues}
- Approach: {trace.solution_approach}
- Plan: {trace.implementation_plan}
- Validation: {trace.validation_checklist}

WEAK AREAS TO IMPROVE:
{chr(10).join(f'- {area}: Score {step_scores.get(area, 0):.2f}/1.0' for area in weak_areas)}

Provide an IMPROVED version focusing on the weak areas. Return JSON:
{{
    "requirements_analysis": "Improved requirements (more specific)",
    "potential_issues": ["More specific issue 1", "Issue 2", "Issue 3", "Issue 4"],
    "solution_approach": "More detailed approach",
    "implementation_plan": "Step 1: Specific action\\nStep 2: ...\\nStep 3: ...\\nStep 4: ...\\nStep 5: ...",
    "validation_checklist": ["Specific check 1", "Check 2", "Check 3", "Check 4", "Check 5"]
}}"""

        messages = [
            SystemMessage(content="You are refining an implementation plan to address weak areas."),
            HumanMessage(content=refinement_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = get_text_content(response.content).strip()

            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())

                from dataclasses import replace
                return replace(
                    trace,
                    requirements_analysis=result.get("requirements_analysis", trace.requirements_analysis),
                    potential_issues=result.get("potential_issues", trace.potential_issues),
                    solution_approach=result.get("solution_approach", trace.solution_approach),
                    implementation_plan=result.get("implementation_plan", trace.implementation_plan),
                    validation_checklist=result.get("validation_checklist", trace.validation_checklist),
                )

        except Exception as e:
            print(f"   ⚠️ Reasoning refinement failed: {e}")

        return trace

    def _format_reasoning_for_prompt(self, trace: ReasoningTrace) -> str:
        """Format reasoning trace for injection into code generation prompt."""
        if not trace:
            return ""

        issues_str = "\n".join(f"  - {issue}" for issue in trace.potential_issues[:5])
        validation_str = "\n".join(f"  - {check}" for check in trace.validation_checklist[:5])

        return f"""
## GRPO Reasoning Trace (Pre-implementation Analysis)

### Requirements Analysis
{trace.requirements_analysis}

### Potential Issues to Prevent
{issues_str}

### Solution Approach
{trace.solution_approach}

### Implementation Plan
{trace.implementation_plan}

### Validation Checklist (MUST verify before submission)
{validation_str}

**IMPORTANT**: Follow this reasoning trace. Address ALL identified issues in your implementation.
"""
