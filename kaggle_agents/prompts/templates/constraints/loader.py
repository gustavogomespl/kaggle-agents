"""
Lazy loader for domain-specific constraints.

Reduces token usage by loading only relevant constraints for each domain.
"""

from functools import lru_cache
from typing import Optional

# Domain categories for constraint loading
DOMAIN_CATEGORIES = {
    # Image domains
    "image": ["cv", "computer_vision", "image_classification"],
    "image_to_image": ["image_to_image", "image_denoising", "super_resolution", "image_segmentation"],

    # Text domains
    "nlp": ["nlp", "text", "text_classification", "sentiment", "translation"],

    # Tabular domains
    "tabular": ["tabular", "structured", "time_series"],
}


def _detect_domain_category(domain: str) -> str:
    """Map specific domain to its category."""
    # Guard against None or empty domain
    if not domain:
        return "tabular"

    domain_lower = str(domain).lower().strip()

    for category, keywords in DOMAIN_CATEGORIES.items():
        if domain_lower in keywords or any(kw in domain_lower for kw in keywords):
            return category

    # Default to tabular for unknown domains
    return "tabular"


@lru_cache(maxsize=16)
def get_constraints_for_domain(domain: str) -> str:
    """
    Get domain-specific constraints, loading only what's needed.

    Args:
        domain: Detected domain (e.g., "tabular", "image_to_image", "nlp")

    Returns:
        Combined constraints string (base + domain-specific)
    """
    from .base import BASE_CONSTRAINTS

    parts = [BASE_CONSTRAINTS]

    category = _detect_domain_category(domain)

    if category == "image_to_image":
        from .image import IMAGE_CONSTRAINTS
        from .image_to_image import IMAGE_TO_IMAGE_CONSTRAINTS
        parts.append(IMAGE_CONSTRAINTS)
        parts.append(IMAGE_TO_IMAGE_CONSTRAINTS)

    elif category == "image":
        from .image import IMAGE_CONSTRAINTS
        parts.append(IMAGE_CONSTRAINTS)

    elif category == "nlp":
        from .nlp import NLP_CONSTRAINTS
        parts.append(NLP_CONSTRAINTS)

    elif category == "tabular":
        from .tabular import TABULAR_CONSTRAINTS
        parts.append(TABULAR_CONSTRAINTS)

    return "\n\n".join(parts)


def get_constraint_token_estimate(domain: str) -> int:
    """Estimate token count for constraints (rough: 4 chars = 1 token)."""
    constraints = get_constraints_for_domain(domain)
    return len(constraints) // 4
