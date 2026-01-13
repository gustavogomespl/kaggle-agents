"""
MLE-bench Data Adapter Module.

Provides utilities to adapt MLE-bench prepared data to kaggle-agents expected format.
"""

from .adapter import MLEBenchDataAdapter
from .dataclasses import MLEBenchDataInfo


__all__ = [
    "MLEBenchDataAdapter",
    "MLEBenchDataInfo",
]
