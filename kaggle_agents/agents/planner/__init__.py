"""
Planner Agent module.

Provides ablation-driven optimization planning for Kaggle competitions.
"""

from .fallback_plans import (
    create_fallback_plan,
    is_image_competition_without_features,
    create_tabular_fallback_plan,
    create_image_fallback_plan,
    create_image_to_image_fallback_plan,
    create_text_fallback_plan,
    create_audio_fallback_plan,
)

__all__ = [
    # Fallback plan functions
    "create_fallback_plan",
    "is_image_competition_without_features",
    "create_tabular_fallback_plan",
    "create_image_fallback_plan",
    "create_image_to_image_fallback_plan",
    "create_text_fallback_plan",
    "create_audio_fallback_plan",
]
