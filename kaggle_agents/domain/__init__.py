"""
Domain-specific detection and pipelines for different competition types.
"""

from .detector import DomainDetector, detect_competition_domain

__all__ = [
    "DomainDetector",
    "detect_competition_domain",
]
