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
    from ...utils.text_normalization import AMBIGUOUS_CLASSES, DETERMINISTIC_CLASSES

    if domain != "seq_to_seq":
        return {}

    patterns: dict[str, Any] = {
        "uses_hybrid_lookup": False,
        "uses_lookup_baseline": False,
        "lookup_coverage_estimate": 0.0,
        "deterministic_classes": list(DETERMINISTIC_CLASSES),
        "ambiguous_classes": list(AMBIGUOUS_CLASSES),
        "neural_models": set(),
        "recommended_utilities": set(),
    }

    for sol in sota_solutions:
        # PRIMARY: Use structured fields from SOTASolution dataclass
        for strategy in sol.strategies:
            strategy_lower = strategy.lower()
            if any(kw in strategy_lower for kw in ["lookup", "dictionary", "hybrid"]):
                patterns["uses_hybrid_lookup"] = True
                patterns["lookup_coverage_estimate"] = 0.80

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
    """Format seq2seq-specific insights for text normalization competitions."""
    from ...utils.text_normalization import AMBIGUOUS_CLASSES, DETERMINISTIC_CLASSES

    # Use imported constants for complete class lists
    deterministic_str = ", ".join(sorted(DETERMINISTIC_CLASSES))
    ambiguous_str = ", ".join(sorted(AMBIGUOUS_CLASSES))

    insights = f"""## DOMAIN-SPECIFIC INSIGHTS (CRITICAL FOR seq_to_seq)

### SEQ2SEQ / TEXT NORMALIZATION PATTERNS

**CRITICAL: HYBRID LOOKUP-FIRST STRATEGY**
SOTA solutions show that lookup-based approaches handle 80%+ of tokens deterministically.
This is a PROVEN pattern - you MUST include a lookup-first component.

**Class Categories (from text_normalization.py):**
- DETERMINISTIC (use rules/lookup): {deterministic_str}
- AMBIGUOUS (use neural): {ambiguous_str}

**RECOMMENDED ARCHITECTURE:**
1. **Component 1 (Lookup Baseline)**: Handle deterministic classes with O(1) lookup
   - Use `LookupBaseline` from `kaggle_agents/utils/text_normalization.py`
   - Expected coverage: 80%+ of tokens
   - Impact: 0.30-0.40

2. **Component 2 (Neural Seq2Seq)**: Handle ambiguous classes only
   - Train ONLY on ~20% of data (ambiguous tokens)
   - Use T5-small with `get_neural_training_config()` for time-aware training
   - Impact: 0.20-0.30

3. **Component 3 (Hybrid Pipeline)**: Combine lookup + neural
   - Use `create_hybrid_pipeline()` utility
   - Lookup first, neural fallback for ambiguous
   - Impact: 0.10-0.15

**AVAILABLE UTILITIES (USE THESE!):**
```python
from kaggle_agents.utils.text_normalization import (
    LookupBaseline,              # Frequency-based lookup table
    create_hybrid_pipeline,      # Returns lookup + ambiguous_df + neural_config
    get_neural_training_config,  # Time-aware training config with max_steps guard
    DETERMINISTIC_CLASSES,       # Complete set of deterministic classes
    AMBIGUOUS_CLASSES,           # Complete set of ambiguous classes
)
```

**MANDATORY FOR SEQ2SEQ:**
- At least ONE component must use `LookupBaseline` or equivalent lookup approach
- At least ONE component must generate the actual "after" text (not just predict class)
- Neural training MUST use `max_steps` guard to prevent timeout
- Component impacts must be REALISTIC (sum should be ≤ 0.70 for a 3-component plan)
"""

    # Add detected patterns if any
    if patterns.get("uses_hybrid_lookup"):
        coverage = patterns.get("lookup_coverage_estimate", 0.8)
        insights += f"\n**DETECTED FROM SOTA:** Hybrid lookup strategy confirmed (est. {coverage:.0%} coverage)"

    if patterns.get("neural_models"):
        models = ", ".join(patterns["neural_models"])
        insights += f"\n**DETECTED NEURAL MODELS:** {models}"

    return insights
