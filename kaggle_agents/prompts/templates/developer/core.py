"""
Core prompt constants for the Developer Agent.

Contains the core identity and logging format.
The HARD_CONSTRAINTS constant is kept for backward compatibility but
prefer using the modular constraints from .constraints module.
"""

# Re-export builders for backward compatibility


# ==================== Core Identity ====================

DEVELOPER_CORE_IDENTITY = """You are a Kaggle Grandmaster implementing ML components.

Style:
- Write minimal, working code - no unnecessary abstractions
- No comments unless logic is non-obvious
- Use proven patterns from SOTA solutions when provided
- Print structured logs for the feedback loop

Output: A single Python code block. No explanations outside the code."""


# ==================== Logging Format ====================

LOGGING_FORMAT = """## Structured Logs (required for feedback loop):
[LOG:FOLD] fold={n} score={score:.6f} time={time:.2f}
[LOG:CV_SUMMARY] mean={mean:.6f} std={std:.6f} scores={list}
[LOG:OPTUNA] trial={n} score={score:.6f} time={time:.2f} params={dict}
[LOG:TIMING] step={name} time={time:.2f} cumulative={cumulative:.2f}
[LOG:FEATURES] top={list[:20]} importances={list[:20]}
[LOG:WARNING] message={str}
[LOG:ERROR] message={str}"""


# ==================== Hard Constraints (Legacy - Prefer Modular) ====================

# DEPRECATED: Use get_constraints_for_domain() from .constraints for domain-specific
# constraints. This constant is kept for backward compatibility only.
# The modular system reduces token usage by 40-60% by loading only relevant constraints.

def get_hard_constraints() -> str:
    """
    Get full HARD_CONSTRAINTS by combining all domain constraints.

    DEPRECATED: Prefer using get_constraints_for_domain(domain) for
    domain-specific constraints to reduce token usage.
    """
    from ..constraints.audio import AUDIO_CONSTRAINTS
    from ..constraints.base import BASE_CONSTRAINTS
    from ..constraints.image import IMAGE_CONSTRAINTS
    from ..constraints.image_to_image import IMAGE_TO_IMAGE_CONSTRAINTS
    from ..constraints.nlp import NLP_CONSTRAINTS
    from ..constraints.tabular import TABULAR_CONSTRAINTS

    return "\n\n".join([
        BASE_CONSTRAINTS,
        "## TABULAR-SPECIFIC:",
        TABULAR_CONSTRAINTS,
        "## IMAGE-SPECIFIC:",
        IMAGE_CONSTRAINTS,
        "## IMAGE-TO-IMAGE SPECIFIC:",
        IMAGE_TO_IMAGE_CONSTRAINTS,
        "## NLP-SPECIFIC:",
        NLP_CONSTRAINTS,
        "## AUDIO-SPECIFIC:",
        AUDIO_CONSTRAINTS,
    ])


# Lazy-loaded for backward compatibility
_HARD_CONSTRAINTS = None


def _get_hard_constraints_lazy() -> str:
    """Lazy load HARD_CONSTRAINTS on first access."""
    global _HARD_CONSTRAINTS
    if _HARD_CONSTRAINTS is None:
        _HARD_CONSTRAINTS = get_hard_constraints()
    return _HARD_CONSTRAINTS


# For backward compatibility - accessing HARD_CONSTRAINTS returns the full constraints
class _HardConstraintsProxy:
    """Proxy class for lazy loading HARD_CONSTRAINTS."""

    def __str__(self) -> str:
        return _get_hard_constraints_lazy()

    def __repr__(self) -> str:
        return f"HARD_CONSTRAINTS({len(str(self))} chars)"

    def __add__(self, other: str) -> str:
        return str(self) + other

    def __radd__(self, other: str) -> str:
        return other + str(self)


HARD_CONSTRAINTS = _HardConstraintsProxy()
