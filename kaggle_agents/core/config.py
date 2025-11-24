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


load_dotenv()


@dataclass
class LLMConfig:
    """LLM provider and model configuration."""

    provider: Literal["openai", "anthropic", "gemini"] = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai")
    )
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))

    planner_model: Optional[str] = field(
        default_factory=lambda: os.getenv("PLANNER_MODEL")
    )
    planner_provider: Optional[Literal["openai", "anthropic", "gemini"]] = field(
        default_factory=lambda: os.getenv("PLANNER_PROVIDER")
    )
    developer_model: Optional[str] = field(
        default_factory=lambda: os.getenv("DEVELOPER_MODEL")
    )
    developer_provider: Optional[Literal["openai", "anthropic", "gemini"]] = field(
        default_factory=lambda: os.getenv("DEVELOPER_PROVIDER")
    )
    evaluator_model: Optional[str] = field(
        default_factory=lambda: os.getenv("EVALUATOR_MODEL")
    )
    evaluator_provider: Optional[Literal["openai", "anthropic", "gemini"]] = field(
        default_factory=lambda: os.getenv("EVALUATOR_PROVIDER")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "8192"))
    )
    timeout: int = 120


@dataclass
class SearchConfig:
    """Configuration for SOTA search and retrieval."""

    max_notebooks: int = 10
    min_votes: int = 5
    embedding_model: str = "text-embedding-ada-002"
    vector_store_path: str = ".chromadb"
    search_depth: Literal["quick", "moderate", "thorough"] = "moderate"


@dataclass
class AblationConfig:
    """Configuration for ablation-driven optimization."""

    max_components: int = field(
        default_factory=lambda: int(os.getenv("MAX_COMPONENTS", "3"))
    )
    impact_threshold: float = 0.01
    parallel_testing: bool = False

    testing_timeout: int = 2700
    optuna_trials: int = 5
    enable_code_preview: bool = True
    save_generated_code: bool = True
    code_preview_lines: int = 30
    enable_refinement: bool = (
        True
    )


@dataclass
class ValidationConfig:
    """Configuration for robustness validation."""

    enable_debugging: bool = True
    enable_leakage_check: bool = True
    enable_data_usage_check: bool = True
    enable_format_check: bool = True
    min_validation_score: float = 0.7


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

    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "2"))
    )
    target_percentile: float = 20.0
    early_stopping: bool = True
    patience: int = 3
    min_score_improvement: float = 0.001


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

        base_dir = self._detect_environment()


        if base_dir != self.base_dir:
            self.base_dir = base_dir

        self.competitions_dir = self.base_dir / "competitions"
        self.models_dir = self.base_dir / "models"
        self.submissions_dir = self.base_dir / "submissions"
        self.logs_dir = self.base_dir / "logs"
        self.cache_dir = self.base_dir / ".cache"


        for dir_path in [
            self.competitions_dir,
            self.models_dir,
            self.submissions_dir,
            self.logs_dir,
            self.cache_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _detect_environment(self) -> Path:
        """
        Detect execution environment and return appropriate base directory.

        Returns:
            Path: Best base directory for the environment
        """

        try:
            import google.colab


            colab_base = Path("/content/kaggle_competitions")
            print(f"📍 Colab environment detected, using: {colab_base}")
            return colab_base
        except ImportError:
            pass


        if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):

            kaggle_base = Path("/kaggle/working")
            print(f"📍 Kaggle Kernel detected, using: {kaggle_base}")
            return kaggle_base


        return Path.cwd()


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: str = field(default_factory=lambda: os.getenv("LOG_DIR", "./logs"))
    enable_console: bool = field(
        default_factory=lambda: os.getenv("LOG_CONSOLE", "true").lower() == "true"
    )
    enable_file: bool = field(
        default_factory=lambda: os.getenv("LOG_FILE", "true").lower() == "true"
    )
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class KaggleConfig:
    """Kaggle API configuration."""

    username: str = field(default_factory=lambda: os.getenv("KAGGLE_USERNAME", ""))
    key: str = field(default_factory=lambda: os.getenv("KAGGLE_KEY", ""))
    auto_submit: bool = field(
        default_factory=lambda: os.getenv("KAGGLE_AUTO_SUBMIT", "false").lower()
        == "true"
    )
    submission_message_template: str = (
        "AutoKaggle Agent - Iteration {iteration} (CV: {cv_score:.4f})"
    )

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


    debug_mode: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    verbose: bool = field(
        default_factory=lambda: os.getenv("VERBOSE", "true").lower() == "true"
    )
    save_intermediate: bool = True

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []


        if self.llm.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            issues.append("OPENAI_API_KEY environment variable not set")
        elif self.llm.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            issues.append("ANTHROPIC_API_KEY environment variable not set")
        elif self.llm.provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            issues.append("GOOGLE_API_KEY environment variable not set")


        if self.kaggle.auto_submit and not self.kaggle.is_configured():
            issues.append("Kaggle auto-submit enabled but credentials not configured")


        if not 0 <= self.llm.temperature <= 2:
            issues.append(
                f"LLM temperature {self.llm.temperature} outside valid range [0, 2]"
            )

        if not 0 < self.iteration.target_percentile <= 100:
            issues.append(
                f"Target percentile {self.iteration.target_percentile} must be between 0 and 100"
            )

        if (
            self.validation.min_validation_score < 0
            or self.validation.min_validation_score > 1
        ):
            issues.append(
                f"Validation score {self.validation.min_validation_score} must be between 0 and 1"
            )

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


        issues = _global_config.validate()
        if issues:
            print("  Configuration Issues:")
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





def get_llm(temperature: float = None, max_tokens: int = None):
    """
    Get the configured LLM instance (OpenAI or Anthropic).

    This centralizes LLM creation to support provider switching.

    Args:
        temperature: Override default temperature (optional)
        max_tokens: Override default max_tokens (optional)

    Returns:
        ChatOpenAI or ChatAnthropic instance
    """
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic

    config = get_config()

    temp = temperature if temperature is not None else config.llm.temperature
    tokens = max_tokens if max_tokens is not None else config.llm.max_tokens

    if config.llm.provider == "anthropic":
        return ChatAnthropic(
            model=config.llm.model,
            temperature=temp,
            max_tokens=tokens,
        )
    elif config.llm.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.llm.model,
            temperature=temp,
            max_output_tokens=tokens,
        )
    else:

        return ChatOpenAI(
            model=config.llm.model,
            temperature=temp,
            max_tokens=tokens,
        )


def get_llm_for_role(
    role: str,
    temperature: float = None,
    max_tokens: int = None,
):
    """
    Return an LLM configured for a specific agent role (planner/developer/evaluator).

    Falls back to the global provider/model when a role-specific override is not set.
    """
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic

    config = get_config()
    temp = temperature if temperature is not None else config.llm.temperature
    tokens = max_tokens if max_tokens is not None else config.llm.max_tokens

    role_lower = role.lower()
    provider_override = None
    model_override = None

    if role_lower == "planner":
        provider_override = config.llm.planner_provider
        model_override = config.llm.planner_model
    elif role_lower == "developer":
        provider_override = config.llm.developer_provider
        model_override = config.llm.developer_model
    elif role_lower in {"evaluator", "meta_evaluator", "critic"}:
        provider_override = config.llm.evaluator_provider
        model_override = config.llm.evaluator_model

    provider = provider_override or config.llm.provider
    model = model_override or config.llm.model

    if provider == "anthropic":
        return ChatAnthropic(
            model=model,
            temperature=temp,
            max_tokens=tokens,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temp,
            max_output_tokens=tokens,
        )

    return ChatOpenAI(
        model=model,
        temperature=temp,
        max_tokens=tokens,
    )


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





def is_metric_minimization(metric_name: str) -> bool:
    """
    Determine if a metric should be minimized or maximized.

    Args:
        metric_name: Name of the evaluation metric (e.g., 'rmse', 'accuracy')

    Returns:
        True if metric should be minimized (lower is better), False otherwise

    Examples:
        >>> is_metric_minimization('rmse')
        True
        >>> is_metric_minimization('accuracy')
        False
        >>> is_metric_minimization('log_loss')
        True
    """
    if not metric_name:
        return False

    metric_lower = metric_name.lower()


    minimize_metrics = [
        "rmse",
        "mae",
        "mse",
        "rmsle",
        "logloss",
        "log_loss",
        "log loss",
        "error",
        "loss",
        "cross_entropy",
        "brier",
        "mean_absolute_error",
        "mean_squared_error",
        "root_mean_squared_error",
        "mean_absolute_percentage_error",
        "mape",
    ]

    return any(metric in metric_lower for metric in minimize_metrics)


def calculate_score_improvement(
    new_score: float, baseline_score: float, metric_name: str
) -> float:
    """
    Calculate score improvement considering metric direction.

    Args:
        new_score: New score achieved
        baseline_score: Baseline score to compare against
        metric_name: Name of the evaluation metric

    Returns:
        Improvement value (positive = better, negative = worse)

    Examples:
        >>> calculate_score_improvement(0.350, 0.400, 'rmse')
        0.050
        >>> calculate_score_improvement(0.85, 0.80, 'accuracy')
        0.050
        >>> calculate_score_improvement(0.450, 0.400, 'rmse')
        -0.050
    """
    is_minimize = is_metric_minimization(metric_name)

    if is_minimize:

        return baseline_score - new_score
    else:

        return new_score - baseline_score


def compare_scores(score1: float, score2: float, metric_name: str) -> float:
    """
    Compare two scores and return the better one.

    Args:
        score1: First score to compare
        score2: Second score to compare
        metric_name: Name of the evaluation metric

    Returns:
        The better score according to metric direction

    Examples:
        >>> compare_scores(0.350, 0.400, 'rmse')
        0.350
        >>> compare_scores(0.85, 0.80, 'accuracy')
        0.85
    """
    is_minimize = is_metric_minimization(metric_name)

    if is_minimize:
        return min(score1, score2)
    else:
        return max(score1, score2)
