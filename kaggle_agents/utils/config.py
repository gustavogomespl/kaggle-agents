"""
Lightweight configuration helpers (legacy utils layer).

Note: The actively maintained configuration lives in `kaggle_agents/core/config.py`.
This module exists to keep older agent/workflow modules importable and to provide
optional LangSmith tracing configuration.
"""

from __future__ import annotations

import os


class Config:
    """Simple env-backed configuration for legacy workflows."""

    # Model defaults (legacy agents use ChatOpenAI directly)
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5-mini")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0.0")))

    # Local artifact directories (legacy workflows)
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
    SUBMISSIONS_DIR: str = os.getenv("SUBMISSIONS_DIR", "submissions")

    @staticmethod
    def validate() -> None:
        """
        Validate required configuration for legacy workflows.

        Raises:
            ValueError: If required environment variables are missing.
        """
        # Legacy agents rely on ChatOpenAI -> OPENAI_API_KEY.
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Missing OPENAI_API_KEY")

    @staticmethod
    def configure_tracing(
        langsmith_api_key: str | None = None,
        project_name: str = "kaggle-agents",
    ) -> None:
        """Configure LangSmith tracing for observability (no-op if no API key)."""
        api_key = (
            langsmith_api_key or os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        )
        if not api_key:
            return

        # Prefer LANGSMITH_* env vars, but also set LANGCHAIN_* for compatibility.
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", project_name)

        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", project_name)
