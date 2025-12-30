"""
Modular Constraints System.

Domain-specific constraints are loaded lazily to reduce token usage.
Only the relevant constraints for the detected domain are included in prompts.

Usage:
    from .constraints import get_constraints_for_domain

    constraints = get_constraints_for_domain("image_to_image")
    # Returns: BASE_CONSTRAINTS + IMAGE_CONSTRAINTS + IMAGE_TO_IMAGE_CONSTRAINTS
"""

from .loader import get_constraints_for_domain


__all__ = ["get_constraints_for_domain"]
