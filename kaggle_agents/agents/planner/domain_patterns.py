"""Domain-specific pattern extraction for the planner agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ...core.state import SOTASolution


def extract_domain_specific_patterns(
    sota_solutions: list[SOTASolution], domain: str
) -> dict[str, Any]:
    """Extract domain-specific patterns from SOTA solutions.

    Uses structured SOTASolution fields (strategies, models_used) for reliability,
    with code snippet scanning as fallback for additional signals.

    Args:
        sota_solutions: List of SOTA solutions from search
        domain: Detected domain (e.g., 'seq_to_seq', 'tabular')

    Returns:
        Dictionary with extracted patterns for the domain
    """
    if domain != "seq_to_seq":
        return {}

    patterns: dict[str, Any] = {
        "uses_hybrid_lookup": False,
        "uses_lookup_baseline": False,
        "lookup_coverage_estimate": None,
        "neural_models": set(),
        "recommended_utilities": set(),
    }

    for sol in sota_solutions:
        # PRIMARY: Use structured fields from SOTASolution dataclass
        for strategy in sol.strategies:
            strategy_lower = strategy.lower()
            if any(kw in strategy_lower for kw in ["lookup", "dictionary", "hybrid"]):
                patterns["uses_hybrid_lookup"] = True

        for model in sol.models_used:
            model_lower = model.lower()
            if "t5" in model_lower:
                patterns["neural_models"].add("T5")
            if "seq2seq" in model_lower or "transformer" in model_lower:
                patterns["neural_models"].add("Seq2Seq")
            if "lookup" in model_lower:
                patterns["uses_lookup_baseline"] = True
                patterns["recommended_utilities"].add("LookupBaseline")

        # FALLBACK: Scan code snippets for additional signals
        for snippet in sol.code_snippets:
            snippet_lower = snippet.lower()
            if "lookupbaseline" in snippet_lower:
                patterns["uses_lookup_baseline"] = True
                patterns["recommended_utilities"].add("LookupBaseline")
            if "create_hybrid_pipeline" in snippet_lower:
                patterns["recommended_utilities"].add("create_hybrid_pipeline")

    # Add utility recommendations based on detected patterns
    if patterns["uses_hybrid_lookup"]:
        patterns["recommended_utilities"].update([
            "create_hybrid_pipeline",
            "get_neural_training_config",
        ])

    # Convert sets to lists for JSON serialization
    patterns["neural_models"] = list(patterns["neural_models"])
    patterns["recommended_utilities"] = list(patterns["recommended_utilities"])

    return patterns


def format_domain_insights(domain: str, domain_patterns: dict[str, Any]) -> str:
    """Format domain-specific insights for the planner prompt.

    Returns empty string for domains without specific insights.

    Args:
        domain: Detected domain
        domain_patterns: Patterns extracted from SOTA solutions

    Returns:
        Formatted string with domain insights (or empty string)
    """
    if domain == "seq_to_seq":
        return format_seq2seq_insights(domain_patterns)
    # Add more domains as needed:
    # elif domain == "tabular":
    #     return format_tabular_insights(domain_patterns)
    return ""


def format_seq2seq_insights(patterns: dict[str, Any]) -> str:
    """Format validation-gated insights for generic seq2seq tasks."""
    insights = """## DOMAIN-SPECIFIC INSIGHTS (CRITICAL FOR seq_to_seq)

### SEQ2SEQ / TEXT NORMALIZATION PATTERNS

**VALIDATE A HYBRID LOOKUP-FIRST CANDIDATE**
Repeated transformations can make lookup efficient, but neither coverage nor
class behavior may be assumed. Learn routing rules from the training split and
measure them out of fold.

**RECOMMENDED ARCHITECTURE:**
1. **Component 1 (Validated Lookup Baseline)**
   - Use `LookupBaseline` from `kaggle_agents/utils/text_normalization.py`
   - Infer repeated mappings and simple transformations from supplied rows
   - Report OOF confident coverage and accuracy

2. **Component 2 (Neural Seq2Seq)**
   - Train on rows rejected by the OOF confidence router
   - Use T5-small with `get_neural_training_config()` for time-aware training

3. **Component 3 (Hybrid Pipeline)**: Combine lookup + neural
   - Use `create_hybrid_pipeline()` utility
   - Accept lookup predictions only when their learned confidence passes

**AVAILABLE UTILITIES (USE THESE!):**
```python
from kaggle_agents.utils.text_normalization import (
    LookupBaseline,              # Frequency-based lookup table
    create_hybrid_pipeline,      # Returns lookup + ambiguous_df + neural_config
    get_neural_training_config,  # Time-aware training config with max_steps guard
)
```

**MANDATORY FOR SEQ2SEQ:**
- Evaluate lookup against a pure-neural baseline; keep it only if OOF metrics justify it
- At least ONE component must generate the detected target text (not just predict context/class)
- Neural training MUST use `max_steps` guard to prevent timeout
"""

    # Add detected patterns if any
    if patterns.get("uses_hybrid_lookup"):
        insights += "\n**DETECTED IN ALLOWED REFERENCES:** A hybrid lookup strategy was used; revalidate its coverage locally"

    if patterns.get("neural_models"):
        models = ", ".join(patterns["neural_models"])
        insights += f"\n**DETECTED NEURAL MODELS:** {models}"

    return insights
