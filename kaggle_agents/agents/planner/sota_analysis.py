"""SOTA solution analysis functions for the planner agent."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ...core.state import SOTASolution


def analyze_sota_solutions(
    state: dict[str, Any],
    llm,
    use_dspy: bool,
    sota_analyzer=None,
    planner_system_prompt: str = "",
    analyze_sota_prompt: str = "",
) -> dict[str, Any]:
    """
    Analyze SOTA solutions to extract patterns.

    Args:
        state: Current state with SOTA solutions
        llm: LLM instance for analysis
        use_dspy: Whether to use DSPy modules
        sota_analyzer: DSPy SOTA analyzer module
        planner_system_prompt: System prompt for LLM
        analyze_sota_prompt: Prompt template for SOTA analysis

    Returns:
        Dictionary with analysis results
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from ...utils.llm_utils import get_text_content

    sota_solutions = state.get("sota_solutions", [])

    if not sota_solutions:
        return {
            "common_models": [],
            "feature_patterns": [],
            "ensemble_strategies": [],
            "unique_tricks": [],
            "success_factors": [],
        }

    # Format SOTA solutions for analysis
    sota_summary = format_sota_solutions(sota_solutions)

    if use_dspy and sota_analyzer is not None:
        # Use DSPy module
        result = sota_analyzer(sota_solutions=sota_summary)

        analysis = {
            "common_models": result.common_models.split(", ") if result.common_models else [],
            "feature_patterns": result.feature_patterns.split(", ")
            if result.feature_patterns
            else [],
            "ensemble_strategies": result.ensemble_strategies
            if result.ensemble_strategies
            else "",
            "unique_tricks": result.unique_tricks.split(", ") if result.unique_tricks else [],
            "success_factors": result.success_factors.split(", ")
            if result.success_factors
            else [],
        }
    else:
        # Use direct LLM call
        prompt = analyze_sota_prompt.format(sota_solutions=sota_summary)
        messages = [
            SystemMessage(content=planner_system_prompt),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)

        # Parse JSON from response
        try:
            content = get_text_content(response.content)
            # Strip optional markdown fences
            if isinstance(content, str):
                content = content.strip()
                if "```json" in content:
                    content = content.split("```json", 1)[1].split("```", 1)[0].strip()
                elif content.startswith("```") and content.endswith("```"):
                    content = content.strip("` \n")
            analysis = json.loads(content)
        except Exception:
            # Fallback to empty analysis
            analysis = {
                "common_models": [],
                "feature_patterns": [],
                "ensemble_strategies": "",
                "unique_tricks": [],
                "success_factors": [],
            }

    print(f"   Found {len(analysis.get('common_models', []))} common models")
    print(f"   Found {len(analysis.get('feature_patterns', []))} feature patterns")

    return analysis


def format_sota_solutions(solutions: list[SOTASolution]) -> str:
    """Format SOTA solutions for prompts (summary version without code)."""
    formatted = []
    for sol in solutions[:5]:  # Top 5
        formatted.append(f"""
Title: {sol.title}
Votes: {sol.votes}
Models: {", ".join(sol.models_used) if sol.models_used else "N/A"}
Features: {", ".join(sol.feature_engineering) if sol.feature_engineering else "N/A"}
Ensemble: {sol.ensemble_approach or "N/A"}
""")
    return "\n---\n".join(formatted)


def estimate_complexity(sol: SOTASolution) -> str:
    """
    Estimate time complexity based on code patterns.

    Args:
        sol: SOTA solution to analyze

    Returns:
        Complexity level: "Low", "Medium", or "High" with explanation
    """
    high_complexity_signals = [
        "Ensemble",
        "Stacking",
        "stacking",
        "VotingClassifier",
        "BaggingClassifier",
        "StackingClassifier",
        "StackingRegressor",
        "neural",
        "deep",
        "LSTM",
        "Transformer",
        "BERT",
        "CNN",
        "optuna",
        "hyperopt",
        "GridSearchCV",
        "RandomizedSearchCV",
        "n_estimators=5000",
        "n_estimators=10000",
        "epochs=100",
    ]

    medium_complexity_signals = [
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "RandomForest",
        "n_estimators=1000",
        "n_estimators=2000",
        "cross_val",
        "KFold",
        "StratifiedKFold",
    ]

    # Build text to check from all solution fields
    text_to_check = " ".join(sol.models_used or [])
    text_to_check += " " + (sol.ensemble_approach or "")
    text_to_check += " " + " ".join(sol.strategies or [])
    if sol.code_snippets:
        text_to_check += " " + " ".join(sol.code_snippets[:2])

    text_lower = text_to_check.lower()

    # Count signals
    high_count = sum(1 for signal in high_complexity_signals if signal.lower() in text_lower)
    medium_count = sum(
        1 for signal in medium_complexity_signals if signal.lower() in text_lower
    )

    if high_count >= 3:
        return "High (likely slow - heavy ensembles/optimization/deep learning)"
    if high_count >= 1 or medium_count >= 2:
        return "Medium (moderate training time - standard ML pipeline)"
    return "Low (fast - simple models, quick iteration)"


def format_sota_details(solutions: list[SOTASolution]) -> str:
    """
    Format SOTA solutions with code snippets, votes, and complexity estimation.

    This provides detailed information for the "Adopt & Improve" strategy,
    allowing the planner to directly copy successful approaches.

    Args:
        solutions: List of SOTA solutions from search

    Returns:
        Formatted string with detailed solution info including code snippets
    """
    if not solutions:
        return "No SOTA solutions found. Create a baseline plan using domain best practices."

    details = []
    for i, sol in enumerate(solutions[:3], 1):  # Top 3 to save tokens
        # Estimate complexity based on code patterns
        complexity = estimate_complexity(sol)

        # Get code snippet (truncated to 1500 chars as per user preference)
        code_snippet = ""
        if sol.code_snippets:
            code_snippet = sol.code_snippets[0][:1500]
            if len(sol.code_snippets[0]) > 1500:
                code_snippet += "\n... (truncated)"

        details.append(f"""
### Candidate {i}: {sol.title}
- **Votes**: {sol.votes} (Quality Signal - higher is better)
- **Estimated Complexity**: {complexity}
- **Models Used**: {", ".join(sol.models_used) if sol.models_used else "N/A"}
- **Feature Engineering**: {", ".join(sol.feature_engineering) if sol.feature_engineering else "N/A"}
- **Ensemble Approach**: {sol.ensemble_approach or "N/A"}
- **Key Strategies**: {", ".join(sol.strategies[:3]) if sol.strategies else "N/A"}

**Code Snippet** (use this as reference for your implementation):
```python
{code_snippet if code_snippet else "# No code available"}
```
""")

    return "\n".join(details)
