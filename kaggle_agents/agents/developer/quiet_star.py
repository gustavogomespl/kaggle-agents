"""
Quiet-STaR: Self-Evaluation before execution.

Implements internal reflection on generated code before running it,
allowing the agent to identify and fix issues proactively.
"""

import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from ...core.state import AblationComponent, KaggleState, SelfEvaluation
from ...utils.llm_utils import get_text_content


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class QuietStarMixin:
    """Mixin providing Quiet-STaR self-evaluation capabilities."""

    llm: "BaseChatModel"

    def _self_evaluate_code(
        self,
        code: str,
        component: AblationComponent,
        state: KaggleState,
    ) -> SelfEvaluation:
        """
        Quiet-STaR style: Internal reflection before finalizing code.

        Performs self-evaluation asking:
        - Will this code execute without errors?
        - Will it improve the metric?
        - Are there obvious issues to fix?

        Args:
            code: Generated code to evaluate
            component: Component being implemented
            state: Current workflow state

        Returns:
            SelfEvaluation with confidence, concerns, and suggested fixes
        """
        competition_info = state.get("competition_info")
        metric = competition_info.evaluation_metric if competition_info else "unknown"
        domain = state.get("domain_detected", "tabular")

        # Truncate code for LLM
        code_preview = code[:3000] + "..." if len(code) > 3000 else code

        prompt = f"""Self-evaluate this code before execution. Be critical and honest.

COMPONENT: {component.name} ({component.component_type})
METRIC: {metric}
DOMAIN: {domain}

CODE:
```python
{code_preview}
```

Analyze and return JSON:
{{
    "confidence": 0.0-1.0 (how likely this will execute successfully AND improve the metric),
    "concerns": ["Concern 1", "Concern 2", ...] (up to 5 specific issues),
    "suggested_fixes": ["Fix 1", "Fix 2", ...] (concrete fixes for the concerns),
    "proceed": true/false (should we execute this code or regenerate?),
    "reflection": "Brief summary of overall assessment"
}}

Be specific about potential failure points (imports, data types, output format, etc.)."""

        messages = [
            SystemMessage(
                content="You are a critical code reviewer evaluating ML code before execution."
            ),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = get_text_content(response.content).strip()

            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                result = json.loads(json_match.group())

                return SelfEvaluation(
                    confidence=float(result.get("confidence", 0.5)),
                    concerns=result.get("concerns", [])[:5],
                    suggested_fixes=result.get("suggested_fixes", [])[:5],
                    proceed=bool(result.get("proceed", True)),
                    reflection_summary=result.get("reflection", ""),
                )

        except Exception as e:
            print(f"   ⚠️ Self-evaluation failed: {e}")

        # Default: proceed with moderate confidence
        return SelfEvaluation(
            confidence=0.5,
            concerns=[],
            suggested_fixes=[],
            proceed=True,
            reflection_summary="Self-evaluation could not be performed",
        )

    def _apply_self_evaluation_fixes(
        self,
        code: str,
        evaluation: SelfEvaluation,
        component: AblationComponent,
    ) -> str:
        """
        Apply suggested fixes from self-evaluation.

        Args:
            code: Original code
            evaluation: Self-evaluation with suggested fixes
            component: Component being implemented

        Returns:
            Fixed code
        """
        if not evaluation.suggested_fixes:
            return code

        fixes_text = "\n".join(f"- {fix}" for fix in evaluation.suggested_fixes)
        concerns_text = "\n".join(f"- {concern}" for concern in evaluation.concerns)

        prompt = f"""Apply these fixes to the code:

CONCERNS:
{concerns_text}

SUGGESTED FIXES:
{fixes_text}

ORIGINAL CODE:
```python
{code}
```

Return the COMPLETE fixed code with all fixes applied. Include ALL imports and functionality."""

        messages = [
            SystemMessage(
                content="You are fixing code based on self-evaluation feedback. Apply all suggested fixes."
            ),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            fixed_code = self._extract_code_from_response(get_text_content(response.content))

            if len(fixed_code) > 100:  # Basic sanity check
                print(f"   ✓ Applied {len(evaluation.suggested_fixes)} self-evaluation fixes")
                return fixed_code

        except Exception as e:
            print(f"   ⚠️ Failed to apply self-evaluation fixes: {e}")

        return code

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response

        return code.strip()
