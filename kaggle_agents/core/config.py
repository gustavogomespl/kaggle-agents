"""
Centralized Configuration Management for Kaggle Agents.

This module provides a clean, type-safe configuration system following
best practices with environment variable support and validation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """LLM provider and model configuration."""

    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120  # seconds


@dataclass
class SearchConfig:
    """Configuration for SOTA search and retrieval."""

    max_notebooks: int = 10  # max notebooks to retrieve
    min_votes: int = 5  # minimum votes for consideration
    embedding_model: str = "text-embedding-ada-002"
    vector_store_path: str = ".chromadb"
    search_depth: Literal["quick", "moderate", "thorough"] = "moderate"


@dataclass
class AblationConfig:
    """Configuration for ablation-driven optimization."""

    max_components: int = 10  # max components to test
    impact_threshold: float = 0.01  # minimum impact to consider (1%)
    parallel_testing: bool = False  # test components in parallel
    testing_timeout: int = 600  # seconds per component test


@dataclass
class ValidationConfig:
    """Configuration for robustness validation."""

    enable_debugging: bool = True
    enable_leakage_check: bool = True
    enable_data_usage_check: bool = True
    enable_format_check: bool = True
    min_validation_score: float = 0.7  # 70% pass rate


@dataclass
class DSPyConfig:
    """Configuration for DSPy prompt optimization."""

    enabled: bool = True
    optimizer: Literal["MIPROv2", "BootstrapFewShot", "SignatureOptimizer"] = "MIPROv2"
    training_examples: int = 20
    max_iterations: int = 50
    metric: Literal["kaggle_score", "cv_score", "combined"] = "combined"


@dataclass
class IterationConfig:
    """Configuration for iteration and convergence."""

    max_iterations: int = 10
    target_percentile: float = 20.0  # top 20%
    early_stopping: bool = True
    patience: int = 3  # iterations without improvement
    min_score_improvement: float = 0.001  # 0.1% minimum improvement


@dataclass
class PathConfig:
    """File system paths configuration."""

    base_dir: Path = field(default_factory=lambda: Path.cwd())
    competitions_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    submissions_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    def __post_init__(self):
        """Initialize derived paths."""
        self.competitions_dir = self.base_dir / "competitions"
        self.models_dir = self.base_dir / "models"
        self.submissions_dir = self.base_dir / "submissions"
        self.logs_dir = self.base_dir / "logs"
        self.cache_dir = self.base_dir / ".cache"

        # Create directories
        for dir_path in [
            self.competitions_dir,
            self.models_dir,
            self.submissions_dir,
            self.logs_dir,
            self.cache_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: str = field(default_factory=lambda: os.getenv("LOG_DIR", "./logs"))
    enable_console: bool = field(default_factory=lambda: os.getenv("LOG_CONSOLE", "true").lower() == "true")
    enable_file: bool = field(default_factory=lambda: os.getenv("LOG_FILE", "true").lower() == "true")
    max_file_size_mb: int = 10  # Max size per log file
    backup_count: int = 5  # Number of backup files to keep


@dataclass
class KaggleConfig:
    """Kaggle API configuration."""

    username: str = field(default_factory=lambda: os.getenv("KAGGLE_USERNAME", ""))
    key: str = field(default_factory=lambda: os.getenv("KAGGLE_KEY", ""))
    auto_submit: bool = field(default_factory=lambda: os.getenv("KAGGLE_AUTO_SUBMIT", "false").lower() == "true")
    submission_message_template: str = "AutoKaggle Agent - Iteration {iteration} (CV: {cv_score:.4f})"

    def is_configured(self) -> bool:
        """Check if Kaggle API credentials are configured."""
        return bool(self.username and self.key)


@dataclass
class AgentConfig:
    """Main configuration aggregating all sub-configs."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    dspy: DSPyConfig = field(default_factory=DSPyConfig)
    iteration: IterationConfig = field(default_factory=IterationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    kaggle: KaggleConfig = field(default_factory=KaggleConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Global settings
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    verbose: bool = field(default_factory=lambda: os.getenv("VERBOSE", "true").lower() == "true")
    save_intermediate: bool = True  # save intermediate results

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        # Check LLM API keys
        if self.llm.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            issues.append("OPENAI_API_KEY environment variable not set")
        elif self.llm.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            issues.append("ANTHROPIC_API_KEY environment variable not set")

        # Check Kaggle credentials if auto-submit enabled
        if self.kaggle.auto_submit and not self.kaggle.is_configured():
            issues.append("Kaggle auto-submit enabled but credentials not configured")

        # Validate ranges
        if not 0 <= self.llm.temperature <= 2:
            issues.append(f"LLM temperature {self.llm.temperature} outside valid range [0, 2]")

        if not 0 < self.iteration.target_percentile <= 100:
            issues.append(f"Target percentile {self.iteration.target_percentile} must be between 0 and 100")

        if self.validation.min_validation_score < 0 or self.validation.min_validation_score > 1:
            issues.append(f"Validation score {self.validation.min_validation_score} must be between 0 and 1")

        return issues

    @classmethod
    def from_env(cls, overrides: Optional[dict] = None) -> "AgentConfig":
        """
        Create configuration from environment variables with optional overrides.

        Args:
            overrides: Dictionary of config overrides

        Returns:
            AgentConfig instance
        """
        config = cls()

        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return config


# ==================== Global Config Instance ====================

_global_config: Optional[AgentConfig] = None


def get_config() -> AgentConfig:
    """
    Get the global configuration instance.

    Returns:
        Global AgentConfig instance
    """
    global _global_config

    if _global_config is None:
        _global_config = AgentConfig.from_env()

        # Validate and raise if critical issues
        issues = _global_config.validate()
        if issues:
            print("�  Configuration Issues:")
            for issue in issues:
                print(f"  - {issue}")

    return _global_config


def set_config(config: AgentConfig) -> None:
    """
    Set the global configuration instance.

    Args:
        config: AgentConfig to set as global
    """
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration to None."""
    global _global_config
    _global_config = None


# ==================== Convenience Functions ====================

def get_competition_dir(competition_name: str) -> Path:
    """
    Get the directory path for a specific competition.

    Args:
        competition_name: Name of the competition

    Returns:
        Path to competition directory
    """
    config = get_config()
    comp_dir = config.paths.competitions_dir / competition_name
    comp_dir.mkdir(parents=True, exist_ok=True)
    return comp_dir


def get_model_save_path(competition_name: str, model_name: str, iteration: int) -> Path:
    """
    Get the save path for a trained model.

    Args:
        competition_name: Name of the competition
        model_name: Name of the model
        iteration: Current iteration number

    Returns:
        Path to save the model
    """
    config = get_config()
    model_dir = config.paths.models_dir / competition_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{model_name}_iter{iteration}.pkl"


def get_submission_path(competition_name: str, iteration: int) -> Path:
    """
    Get the path for a submission file.

    Args:
        competition_name: Name of the competition
        iteration: Current iteration number

    Returns:
        Path to submission file
    """
    config = get_config()
    sub_dir = config.paths.submissions_dir / competition_name
    sub_dir.mkdir(parents=True, exist_ok=True)
    return sub_dir / f"submission_iter{iteration}.csv"
