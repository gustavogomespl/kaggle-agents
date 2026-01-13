"""
Planner Agent module with ablation-driven optimization.

This module implements the planning strategy from Google ADK for
identifying high-impact components for systematic improvement.
"""

from .agent import PlannerAgent, planner_agent_node

# Re-export DSPy modules for backward compatibility
from .dspy_modules import (
    AblationPlannerModule,
    AblationPlannerSignature,
    SOTAAnalysisSignature,
    SOTAAnalyzerModule,
)

# Re-export fallback plan functions
from .fallback_plans import (
    create_audio_fallback_plan,
    create_fallback_plan,
    create_image_fallback_plan,
    create_image_to_image_fallback_plan,
    create_tabular_fallback_plan,
    create_text_fallback_plan,
)

# Re-export strategies for backward compatibility
from .strategies import (
    EXTENDED_STRATEGIES_CV,
    EXTENDED_STRATEGIES_NLP,
    EXTENDED_STRATEGIES_TABULAR,
)

# Re-export validation functions
from .validation import (
    detect_multimodal_competition,
    is_image_competition_without_features,
    validate_plan,
)


__all__ = [
    # Main exports
    "PlannerAgent",
    "planner_agent_node",
    # DSPy modules
    "AblationPlannerModule",
    "AblationPlannerSignature",
    "SOTAAnalyzerModule",
    "SOTAAnalysisSignature",
    # Strategy constants
    "EXTENDED_STRATEGIES_CV",
    "EXTENDED_STRATEGIES_NLP",
    "EXTENDED_STRATEGIES_TABULAR",
    # Fallback plan functions
    "create_fallback_plan",
    "create_tabular_fallback_plan",
    "create_image_fallback_plan",
    "create_image_to_image_fallback_plan",
    "create_text_fallback_plan",
    "create_audio_fallback_plan",
    # Validation functions
    "is_image_competition_without_features",
    "detect_multimodal_competition",
    "validate_plan",
]
