"""
Prompt templates for the Developer Agent.

Refactored to be agentic, feedback-driven, and RL-friendly.
Inspired by Claude Code's concise style.

Builder functions are now in the builders/ submodule for better organization.

NOTE: This module is kept for backward compatibility.
The actual implementation is in the developer/ submodule.
"""

# Re-export everything from the modular developer package
from .developer import (
    # Ablation
    ABLATION_STUDY_PROMPT,
    ABLATION_STUDY_SEQUENTIAL_PROMPT,
    DEBUG_CODE_PROMPT,
    # Core
    DEVELOPER_CORE_IDENTITY,
    EXTRACT_IMPROVEMENT_PLAN_PROMPT,
    EXTRACT_IMPROVEMENT_PLAN_SEQUENTIAL_PROMPT,
    # Fix/Debug
    FIX_CODE_PROMPT,
    HARD_CONSTRAINTS,
    IMPLEMENT_PLAN_PROMPT,
    LOGGING_FORMAT,
    PLAN_REFINEMENT_PROMPT,
    REFINEMENT_WITH_FEEDBACK_PROMPT,
    SUMMARIZE_ABLATION_PROMPT,
    # Builders
    DynamicContext,
    build_context,
    build_dynamic_instructions,
    # Composition
    compose_generate_prompt,
    # Utils
    format_component_details,
    format_error_info,
)


__all__ = [
    # Builders
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
