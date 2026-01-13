"""
MLE-bench Data Adapter - Backward Compatibility Shim.

This file provides backward compatibility for imports from the old location.
The implementation has been split into modular files in the `data_adapter` subpackage.

Use:
    from kaggle_agents.mlebench.data_adapter import MLEBenchDataAdapter
    # or
    from kaggle_agents.mlebench.data_adapter.adapter import MLEBenchDataAdapter
"""

from __future__ import annotations

# Re-export all public interfaces from the new modular package
from .data_adapter import MLEBenchDataAdapter, MLEBenchDataInfo


__all__ = [
    "MLEBenchDataAdapter",
    "MLEBenchDataInfo",
]
