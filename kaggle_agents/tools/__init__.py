"""
Tools for Kaggle competition analysis and retrieval.
"""

from __future__ import annotations

import importlib

from .code_executor import (
    ArtifactValidator,
    CodeExecutor,
    ExecutionResult,
    execute_code,
    validate_code_syntax,
)


__all__ = [
    "ArtifactValidator",
    "CodeExecutor",
    "ExecutionResult",
    "execute_code",
    "validate_code_syntax",
    # Lazy imports (avoid importing Kaggle SDK at module import time)
    "CompetitionAnalyzer",
    "auto_detect_competition_config",
    "DiscussionMetadata",
    "KaggleSearcher",
    "NotebookMetadata",
    "search_competition_notebooks",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CompetitionAnalyzer": ("competition_analyzer", "CompetitionAnalyzer"),
    "auto_detect_competition_config": ("competition_analyzer", "auto_detect_competition_config"),
    "DiscussionMetadata": ("kaggle_search", "DiscussionMetadata"),
    "KaggleSearcher": ("kaggle_search", "KaggleSearcher"),
    "NotebookMetadata": ("kaggle_search", "NotebookMetadata"),
    "search_competition_notebooks": ("kaggle_search", "search_competition_notebooks"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_LAZY_IMPORTS.keys())))
