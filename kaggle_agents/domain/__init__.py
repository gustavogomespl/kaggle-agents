"""
Domain-specific detection and pipelines for different competition types.
"""

from .detector import DomainDetector, detect_competition_domain, detect_submission_format


__all__ = [
    "DomainDetector",
    "detect_competition_domain",
    "detect_submission_format",
]
