"""
Meta-Evaluator Agent - Backward Compatibility Shim.

This module now re-exports from the meta_evaluator submodule.
See kaggle_agents/agents/meta_evaluator/ for the implementation.
"""

# Re-export for backward compatibility
from .meta_evaluator import (
    META_EVALUATOR_SYSTEM_PROMPT,
    SEMANTIC_LOG_ANALYSIS_PROMPT,
    MetaEvaluatorAgent,
    meta_evaluator_node,
)


__all__ = [
    "META_EVALUATOR_SYSTEM_PROMPT",
    "SEMANTIC_LOG_ANALYSIS_PROMPT",
    "MetaEvaluatorAgent",
    "meta_evaluator_node",
]
