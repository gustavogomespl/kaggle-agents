"""
Developer Agent Prompt Templates.

Refactored to be agentic, feedback-driven, and RL-friendly.
Inspired by Claude Code's concise style.

Builder functions are in the builders/ submodule for better organization.
Domain-specific constraints are in the constraints/ submodule for lazy loading.
"""

# Re-export builders for backward compatibility
from ..builders import (
    DynamicContext,
    build_context,
    build_dynamic_instructions,
)

# Ablation study prompts
from .ablation import (
    ABLATION_STUDY_PROMPT,
    ABLATION_STUDY_SEQUENTIAL_PROMPT,
    EXTRACT_IMPROVEMENT_PLAN_PROMPT,
    EXTRACT_IMPROVEMENT_PLAN_SEQUENTIAL_PROMPT,
    IMPLEMENT_PLAN_PROMPT,
    PLAN_REFINEMENT_PROMPT,
    SUMMARIZE_ABLATION_PROMPT,
)

# Core constants
from .core import (
    DEVELOPER_CORE_IDENTITY,
    HARD_CONSTRAINTS,
    LOGGING_FORMAT,
)

# Fix and debug prompts
from .fix_debug import (
    DEBUG_CODE_PROMPT,
    FIX_CODE_PROMPT,
    REFINEMENT_WITH_FEEDBACK_PROMPT,
)

# Prompt composition
from .prompt_composition import (
    compose_generate_prompt,
)

# Utility functions
from .utils import (
    format_component_details,
    format_error_info,
)


__all__ = [
    # Builders (from ../builders)
    "DynamicContext",
    "build_context",
    "build_dynamic_instructions",
    # Core
    "DEVELOPER_CORE_IDENTITY",
    "LOGGING_FORMAT",
    "HARD_CONSTRAINTS",
    # Composition
    "compose_generate_prompt",
    # Fix/Debug
    "FIX_CODE_PROMPT",
    "DEBUG_CODE_PROMPT",
    "REFINEMENT_WITH_FEEDBACK_PROMPT",
    # Ablation
    "ABLATION_STUDY_PROMPT",
    "ABLATION_STUDY_SEQUENTIAL_PROMPT",
    "SUMMARIZE_ABLATION_PROMPT",
    "EXTRACT_IMPROVEMENT_PLAN_PROMPT",
    "EXTRACT_IMPROVEMENT_PLAN_SEQUENTIAL_PROMPT",
    "PLAN_REFINEMENT_PROMPT",
    "IMPLEMENT_PLAN_PROMPT",
    # Utils
    "format_component_details",
    "format_error_info",
]
