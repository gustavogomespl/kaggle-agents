"""
Data classes for domain detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DetectionSignal:
    """A single detection signal with source and confidence."""

    domain: str  # DomainType or partial (e.g., "classification", "regression")
    confidence: float
    source: Literal["structural", "submission", "metric", "llm", "sota", "fallback"]
    metadata: dict[str, Any] = field(default_factory=dict)
