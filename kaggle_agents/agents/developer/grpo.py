"""
GRPO: Reasoning-First Code Generation.

Implements Group Relative Policy Optimization style reasoning traces
that are generated before code generation to improve code quality.

Also provides Chain-of-Thought (CoT) reasoning for complex code generation tasks.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...core.state import AblationComponent, KaggleState, ReasoningTrace
from ...utils.llm_utils import get_text_content

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@dataclass
class ChainOfThoughtResult:
    """Result of Chain-of-Thought reasoning before code generation."""
    data_analysis: str          # Analysis of input data format and characteristics
    transformation_plan: str    # Planned data transformations
    model_architecture: str     # Model/algorithm selection reasoning
    validation_strategy: str    # Cross-validation and evaluation strategy
    output_format: str          # Submission format requirements
    thinking_summary: str       # Overall reasoning summary
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


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

    def _verify_code_alignment(
        self,
        code: str,
        trace: ReasoningTrace,
        state: KaggleState,
    ) -> tuple[float, list[str]]:
        """
        GRPO Enforcement: Verify that generated code aligns with reasoning trace.

        Checks:
        1. Validation checklist items are addressed in code
        2. Identified issues have corresponding safeguards
        3. Implementation follows planned approach

        Args:
            code: Generated code
            trace: Reasoning trace that guided generation
            state: Current workflow state

        Returns:
            Tuple of (alignment_score, list of missing items)
        """
        if not trace or not code:
            return 0.5, []

        code_lower = code.lower()
        missing_items = []
        checks_passed = 0
        total_checks = 0

        # 1. Check validation checklist items (most important)
        for check in trace.validation_checklist:
            total_checks += 1
            # Extract key concepts from checklist item
            concepts = [w.strip().lower() for w in check.split() if len(w) > 3]
            if any(c in code_lower for c in concepts[:3]):
                checks_passed += 1
            else:
                missing_items.append(f"Missing validation: {check}")

        # 2. Check if potential issues are addressed
        issue_keywords = {
            "timeout": ["timeout", "time_limit", "deadline", "max_time"],
            "memory": ["memory", "gc.collect", "del ", "chunk"],
            "overflow": ["clip", "nan", "inf", "overflow"],
            "cross-validation": ["kfold", "stratified", "cv", "fold"],
            "prediction": ["predict", "predict_proba", "submission"],
            "save": ["save", "dump", "pickle", "np.save", "to_csv"],
        }

        for issue in trace.potential_issues:
            issue_lower = issue.lower()
            for key, patterns in issue_keywords.items():
                if key in issue_lower:
                    total_checks += 1
                    if any(p in code_lower for p in patterns):
                        checks_passed += 1
                    else:
                        missing_items.append(f"Issue not addressed: {issue[:50]}")
                    break

        # 3. Check implementation plan steps
        if trace.implementation_plan:
            steps = [s.strip() for s in trace.implementation_plan.split("\n") if s.strip()]
            for step in steps[:5]:  # Check first 5 steps
                # Extract action verbs and nouns
                keywords = [w.lower() for w in step.split() if len(w) > 4][:3]
                if keywords:
                    total_checks += 1
                    if any(k in code_lower for k in keywords):
                        checks_passed += 1
                    else:
                        missing_items.append(f"Step not found: {step[:40]}")

        # Calculate alignment score
        alignment_score = checks_passed / max(total_checks, 1)

        return alignment_score, missing_items

    def _regenerate_with_strict_enforcement(
        self,
        original_code: str,
        trace: ReasoningTrace,
        missing_items: list[str],
        component: AblationComponent,
        state: KaggleState,
    ) -> str:
        """
        GRPO Enforcement: Regenerate code with strict alignment to reasoning trace.

        When initial code doesn't align well with the reasoning trace, this
        method regenerates with explicit requirements for each missing item.

        Args:
            original_code: Original generated code
            trace: Reasoning trace
            missing_items: Items that were not addressed
            component: Component being implemented
            state: Current workflow state

        Returns:
            Regenerated code with better alignment
        """
        print(f"   🔄 GRPO Enforcement: Regenerating with strict alignment...")
        print(f"   Missing items to address: {len(missing_items)}")

        missing_str = "\n".join(f"  - {item}" for item in missing_items[:10])

        enforcement_prompt = f"""You generated code that does NOT fully align with the reasoning trace.

## MISSING REQUIREMENTS (MUST address each one):
{missing_str}

## ORIGINAL REASONING TRACE TO FOLLOW:

### Requirements
{trace.requirements_analysis}

### Potential Issues (MUST handle)
{chr(10).join('- ' + issue for issue in trace.potential_issues)}

### Validation Checklist (MUST implement)
{chr(10).join('- ' + check for check in trace.validation_checklist)}

## ORIGINAL CODE (incomplete):
```python
{original_code[:3000]}
```

## YOUR TASK:
Modify the code above to address ALL missing requirements.
For each missing item, add explicit handling:
1. If a validation check is missing, add it
2. If an issue isn't handled, add safeguards
3. If a step is missing, implement it

Return the COMPLETE, corrected Python code.
Wrap your code in ```python ... ```"""

        messages = [
            SystemMessage(content="You are an expert ML engineer enforcing code quality requirements. Address ALL missing items explicitly."),
            HumanMessage(content=enforcement_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = get_text_content(response.content)

            # Extract code from response
            if "```python" in content:
                code = content.split("```python")[1].split("```")[0]
            elif "```" in content:
                code = content.split("```")[1].split("```")[0]
            else:
                code = content

            return code.strip()

        except Exception as e:
            print(f"   ⚠️ GRPO enforcement failed: {e}")
            return original_code

    def _generate_chain_of_thought(
        self,
        component: AblationComponent,
        state: KaggleState,
        data_info: str = "",
    ) -> ChainOfThoughtResult:
        """
        Generate explicit Chain-of-Thought reasoning before code generation.

        This provides step-by-step reasoning about:
        1. What data format will be received
        2. What transformations are needed
        3. What model architecture fits the task
        4. What validation strategy prevents overfitting
        5. What output format the submission needs

        Args:
            component: Component to implement
            state: Current workflow state
            data_info: Optional dataset information

        Returns:
            ChainOfThoughtResult with structured reasoning
        """
        competition_info = state.get("competition_info")
        domain = state.get("domain_detected", "tabular")
        metric = competition_info.evaluation_metric if competition_info else "unknown"
        problem_type = competition_info.problem_type if competition_info else "unknown"

        prompt = f"""Before writing code, think step-by-step about the implementation:

## TASK CONTEXT
Component: {component.name} ({component.component_type})
Competition: {competition_info.name if competition_info else 'Unknown'}
Domain: {domain}
Problem Type: {problem_type}
Metric: {metric}
Description: {component.code[:300] if component.code else 'No description'}

{f'## DATA INFO{chr(10)}{data_info}' if data_info else ''}

## THINK STEP BY STEP

Answer each question carefully:

1. DATA ANALYSIS: What is the expected input data format? What are the key columns/features? What data types are involved? What preprocessing might be needed?

2. TRANSFORMATION PLAN: What specific transformations will you apply? Feature engineering steps? Encoding strategies? Handling missing values?

3. MODEL ARCHITECTURE: What model/algorithm is most appropriate for this task and domain? What hyperparameters should be considered? Why is this model suitable?

4. VALIDATION STRATEGY: How will you implement cross-validation? How many folds? StratifiedKFold for classification? What prevents overfitting?

5. OUTPUT FORMAT: What is the exact submission format required? How will predictions be saved? What columns are expected?

Return your reasoning in JSON format:
{{
    "data_analysis": "Your analysis of the input data...",
    "transformation_plan": "Specific transformations to apply...",
    "model_architecture": "Model choice and reasoning...",
    "validation_strategy": "CV strategy and overfitting prevention...",
    "output_format": "Exact submission requirements...",
    "thinking_summary": "Overall approach in 2-3 sentences"
}}"""

        messages = [
            SystemMessage(content="You are an expert ML engineer planning an implementation. Think carefully before coding."),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = get_text_content(response.content).strip()

            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())

                return ChainOfThoughtResult(
                    data_analysis=result.get("data_analysis", ""),
                    transformation_plan=result.get("transformation_plan", ""),
                    model_architecture=result.get("model_architecture", ""),
                    validation_strategy=result.get("validation_strategy", ""),
                    output_format=result.get("output_format", ""),
                    thinking_summary=result.get("thinking_summary", ""),
                )

        except Exception as e:
            print(f"   ⚠️ Chain-of-Thought generation failed: {e}")

        # Default fallback
        return ChainOfThoughtResult(
            data_analysis=f"Implement {component.component_type} for {domain} domain",
            transformation_plan="Apply standard preprocessing",
            model_architecture="Use appropriate model for task",
            validation_strategy="Use StratifiedKFold cross-validation",
            output_format="Standard submission.csv format",
            thinking_summary="Standard implementation approach",
        )

    def _format_cot_for_prompt(self, cot: ChainOfThoughtResult) -> str:
        """Format Chain-of-Thought result for injection into code generation prompt."""
        if not cot:
            return ""

        return f"""
## Chain-of-Thought Analysis (FOLLOW THIS PLAN)

### 1. Data Analysis
{cot.data_analysis}

### 2. Transformation Plan
{cot.transformation_plan}

### 3. Model Architecture
{cot.model_architecture}

### 4. Validation Strategy
{cot.validation_strategy}

### 5. Output Format
{cot.output_format}

### Summary
{cot.thinking_summary}

**IMPORTANT**: Follow this reasoning in your implementation. Each section above should be reflected in your code.
"""
