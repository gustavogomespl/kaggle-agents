"""Configuration management for Kaggle agents."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration for Kaggle agents.

    Environment variables are loaded from .env file.
    LangSmith tracing is automatically enabled when API key is set.
    """

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
    KAGGLE_KEY = os.getenv("KAGGLE_KEY")

    # LangSmith configuration for tracing
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "kaggle-agents")

    # Model settings
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

    # Workflow settings
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
    CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./checkpoints")

    # Data paths
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    MODELS_DIR = os.getenv("MODELS_DIR", "./models")
    SUBMISSIONS_DIR = os.getenv("SUBMISSIONS_DIR", "./submissions")

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If required configuration is missing
        """
        required = ["OPENAI_API_KEY"]
        missing = [key for key in required if not getattr(cls, key)]

        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")

        return True

    @classmethod
    def configure_tracing(cls) -> None:
        """Configure LangSmith tracing if credentials are available."""
        if cls.LANGSMITH_API_KEY and cls.LANGSMITH_TRACING:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = cls.LANGSMITH_API_KEY
            os.environ["LANGSMITH_PROJECT"] = cls.LANGSMITH_PROJECT
            print(f"LangSmith tracing enabled. Project: {cls.LANGSMITH_PROJECT}")
        else:
            os.environ["LANGSMITH_TRACING"] = "false"
