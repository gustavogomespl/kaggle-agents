"""
Domain-specific fallback plans for the Planner Agent.

When LLM plan generation fails, these provide robust fallback plans
based on domain-specific best practices.
"""

from .base import create_fallback_plan, is_image_competition_without_features
from .tabular import create_tabular_fallback_plan
from .image import create_image_fallback_plan, create_image_to_image_fallback_plan
from .text import create_text_fallback_plan
from .audio import create_audio_fallback_plan

__all__ = [
    "create_fallback_plan",
    "is_image_competition_without_features",
    "create_tabular_fallback_plan",
    "create_image_fallback_plan",
    "create_image_to_image_fallback_plan",
    "create_text_fallback_plan",
    "create_audio_fallback_plan",
]
