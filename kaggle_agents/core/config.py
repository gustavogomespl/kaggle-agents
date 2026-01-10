"""
Centralized Configuration Management for Kaggle Agents.

This module provides a clean, type-safe configuration system following
best practices with environment variable support and validation.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """LLM provider and model configuration."""

    provider: Literal["openai", "anthropic", "gemini"] = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai")
    )
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    # Optional per-role overrides to balance cost/quality across agents
    planner_model: str | None = field(default_factory=lambda: os.getenv("PLANNER_MODEL"))
    planner_provider: Literal["openai", "anthropic", "gemini"] | None = field(
        default_factory=lambda: os.getenv("PLANNER_PROVIDER")
    )
    developer_model: str | None = field(default_factory=lambda: os.getenv("DEVELOPER_MODEL"))
    developer_provider: Literal["openai", "anthropic", "gemini"] | None = field(
        default_factory=lambda: os.getenv("DEVELOPER_PROVIDER")
    )
    evaluator_model: str | None = field(default_factory=lambda: os.getenv("EVALUATOR_MODEL"))
    evaluator_provider: Literal["openai", "anthropic", "gemini"] | None = field(
        default_factory=lambda: os.getenv("EVALUATOR_PROVIDER")
    )
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "8192"))
    )  # Safe default
    timeout: int = 120  # seconds
    # OpenAI Responses API - enables new API features (structured outputs, web search, etc.)
    use_responses_api: bool = True


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

    max_components: int = field(
        default_factory=lambda: int(os.getenv("MAX_COMPONENTS", "3"))
    )  # max components to test
    impact_threshold: float = 0.01  # minimum impact to consider (1%)
    parallel_testing: bool = False  # test components in parallel
    # Default timeout per component (seconds). Increased to 2700s (45 minutes) to avoid premature failures on heavy training.
    testing_timeout: int = field(default_factory=lambda: int(os.getenv("TESTING_TIMEOUT", "3000")))
    # Debug mode timeout (seconds). Default 600s (10 min) for Optuna tuning during debug iterations.
    debug_timeout: int = field(default_factory=lambda: int(os.getenv("DEBUG_TIMEOUT", "600")))
    optuna_trials: int = 5  # default number of trials for hyperparameter tuning
    enable_code_preview: bool = True  # show code before execution
    save_generated_code: bool = True  # save generated code to files
    code_preview_lines: int = 30  # number of lines to show in preview
    enable_refinement: bool = True  # enable iterative refinement of successful components

    # Training adaptation settings (prevent cascade of epoch reductions)
    max_epoch_reductions: int = field(
        default_factory=lambda: int(os.getenv("KAGGLE_AGENTS_MAX_EPOCH_REDUCTIONS", "1"))
    )  # max number of epoch reductions allowed (prevents 600→300→150→75 cascade)
    min_epochs: int = field(
        default_factory=lambda: int(os.getenv("KAGGLE_AGENTS_MIN_EPOCHS", "10"))
    )  # minimum epochs floor for image models
    epoch_reduction_factor: float = field(
        default_factory=lambda: float(os.getenv("KAGGLE_AGENTS_EPOCH_REDUCTION", "0.5"))
    )  # reduction factor when timeout occurs (0.5 = halve epochs)
    undertrained_threshold: float = field(
        default_factory=lambda: float(os.getenv("KAGGLE_AGENTS_UNDERTRAINED_THRESHOLD", "0.85"))
    )  # score threshold relative to random baseline (0.85 = 85% of random is undertrained)


@dataclass
class ComponentTimeoutConfig:
    """Component-specific timeout configuration.

    Different component types have different computational requirements:
    - Preprocessing: Usually fast, 10 minutes max
    - Feature engineering: Can be slow with complex operations, 15 minutes
    - Light models (GBM): Moderate, 30 minutes
    - Heavy models (Neural): Long training, 60 minutes
    - Ensemble: Usually fast if models are pre-trained, 10 minutes
    """

    preprocessing: int = field(
        default_factory=lambda: int(os.getenv("TIMEOUT_PREPROCESSING", "600"))
    )  # 10 min
    feature_engineering: int = field(
        default_factory=lambda: int(os.getenv("TIMEOUT_FEATURE_ENGINEERING", "900"))
    )  # 15 min
    model_light: int = field(
        default_factory=lambda: int(os.getenv("TIMEOUT_MODEL_LIGHT", "1800"))
    )  # 30 min (LightGBM, XGBoost, CatBoost)
    model_heavy: int = field(
        default_factory=lambda: int(os.getenv("TIMEOUT_MODEL_HEAVY", "3600"))
    )  # 60 min (Neural networks, deep models)
    ensemble: int = field(
        default_factory=lambda: int(os.getenv("TIMEOUT_ENSEMBLE", "600"))
    )  # 10 min

    # Models considered "heavy" (longer timeout)
    heavy_model_patterns: tuple[str, ...] = (
        "neural",
        "tabnet",
        "deep",
        "transformer",
        "vit",
        "swin",
        "bert",
        "roberta",
        "deberta",
        "mlp",
        "cnn",
        "resnet",
        "efficientnet",
    )

    def get_timeout(self, component_type: str, model_name: str = "") -> int:
        """
        Get appropriate timeout for a component.

        Args:
            component_type: Type of component (preprocessing, model, ensemble, etc.)
            model_name: Name of the model (used to detect heavy models)

        Returns:
            Timeout in seconds
        """
        component_type_lower = component_type.lower()
        model_name_lower = model_name.lower()

        if component_type_lower == "preprocessing":
            return self.preprocessing
        elif component_type_lower == "feature_engineering":
            return self.feature_engineering
        elif component_type_lower == "model":
            # Check if it's a heavy model
            if any(pattern in model_name_lower for pattern in self.heavy_model_patterns):
                return self.model_heavy
            return self.model_light
        elif component_type_lower == "ensemble":
            return self.ensemble
        else:
            # Default to light model timeout
            return self.model_light


@dataclass
class ParallelismConfig:
    """Configuration for parallelism across different operations."""

    # CV fold parallelism (use all cores for cross-validation)
    cv_parallel: bool = field(
        default_factory=lambda: os.getenv("CV_PARALLEL", "true").lower() == "true"
    )
    cv_n_jobs: int = field(
        default_factory=lambda: int(os.getenv("CV_N_JOBS", "-1"))
    )  # -1 = all cores

    # Individual models should be single-threaded to avoid contention
    model_n_jobs: int = field(
        default_factory=lambda: int(os.getenv("MODEL_N_JOBS", "1"))
    )

    # Ensemble operations can parallelize calibration
    ensemble_parallel: bool = field(
        default_factory=lambda: os.getenv("ENSEMBLE_PARALLEL", "true").lower() == "true"
    )


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

    max_iterations: int = field(default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "2")))
    target_percentile: float = 20.0  # top 20%
    early_stopping: bool = True
    patience: int = 3  # iterations without improvement
    min_score_improvement: float = 0.001  # 0.1% minimum improvement

    # Adaptive iteration settings for aggressive optimization
    adaptive_iterations: bool = field(
        default_factory=lambda: os.getenv("ADAPTIVE_ITERATIONS", "true").lower() == "true"
    )
    min_iterations: int = field(
        default_factory=lambda: int(os.getenv("MIN_ITERATIONS", "3"))
    )  # Minimum iterations before considering early stop
    extended_max_iterations: int = field(
        default_factory=lambda: int(os.getenv("EXTENDED_MAX_ITERATIONS", "15"))
    )  # Max iterations when gap is large
    score_gap_threshold: float = field(
        default_factory=lambda: float(os.getenv("SCORE_GAP_THRESHOLD", "0.3"))
    )  # If gap > 30% of target, extend iterations

    # Stagnation detection for SOTA search trigger
    stagnation_window: int = field(
        default_factory=lambda: int(os.getenv("STAGNATION_WINDOW", "3"))
    )  # Number of iterations to check for stagnation
    stagnation_threshold: float = field(
        default_factory=lambda: float(os.getenv("STAGNATION_THRESHOLD", "0.01"))
    )  # Avg improvement < 1% = stagnation

    # LangGraph recursion limit - controls max node traversals before error
    # Formula: ~20 nodes per iteration * max_iterations * 1.5 safety margin
    # Default 300 supports up to ~10 iterations safely
    langgraph_recursion_limit: int = field(
        default_factory=lambda: int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "300"))
    )


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
        # Detect Colab environment and adjust base_dir
        base_dir = self._detect_environment()

        # Override base_dir if in special environment
        if base_dir != self.base_dir:
            self.base_dir = base_dir

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

    def _detect_environment(self) -> Path:
        """
        Detect execution environment and return appropriate base directory.

        Returns:
            Path: Best base directory for the environment
        """
        # Check if running in Google Colab
        # Note: We avoid importing google.colab directly as it can cause
        # compatibility issues with IPython/coloransi versions
        is_colab = (
            (os.path.exists("/content") and os.getenv("COLAB_RELEASE_TAG") is not None)
            or (
                # Fallback: check if google.colab is in sys.modules (already imported)
                "google.colab" in sys.modules
            )
            or (
                # Another fallback: check /content existence + typical Colab env vars
                os.path.exists("/content") and os.getenv("COLAB_GPU") is not None
            )
            or (
                # Final fallback: just check /content directory exists
                os.path.exists("/content") and os.path.exists("/root/.config/Google")
            )
        )

        if is_colab:
            # In Colab, use /content/kaggle_competitions
            colab_base = Path("/content/kaggle_competitions")
            print(f"Colab environment detected, using: {colab_base}")
            return colab_base

        # Check if in Kaggle Kernels
        if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
            # In Kaggle, use /kaggle/working
            kaggle_base = Path("/kaggle/working")
            print(f"📍 Kaggle Kernel detected, using: {kaggle_base}")
            return kaggle_base

        # Default: use current working directory
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
    max_file_size_mb: int = 10  # Max size per log file
    backup_count: int = 5  # Number of backup files to keep


@dataclass
class KaggleConfig:
    """Kaggle API configuration."""

    username: str = field(default_factory=lambda: os.getenv("KAGGLE_USERNAME", ""))
    key: str = field(default_factory=lambda: os.getenv("KAGGLE_KEY", ""))
    auto_submit: bool = field(
        default_factory=lambda: os.getenv("KAGGLE_AUTO_SUBMIT", "false").lower() == "true"
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
    component_timeout: ComponentTimeoutConfig = field(default_factory=ComponentTimeoutConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)

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
        elif self.llm.provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            issues.append("GOOGLE_API_KEY environment variable not set")

        # Check Kaggle credentials if auto-submit enabled
        if self.kaggle.auto_submit and not self.kaggle.is_configured():
            issues.append("Kaggle auto-submit enabled but credentials not configured")

        # Validate ranges
        if not 0 <= self.llm.temperature <= 2:
            issues.append(f"LLM temperature {self.llm.temperature} outside valid range [0, 2]")

        if not 0 < self.iteration.target_percentile <= 100:
            issues.append(
                f"Target percentile {self.iteration.target_percentile} must be between 0 and 100"
            )

        if self.validation.min_validation_score < 0 or self.validation.min_validation_score > 1:
            issues.append(
                f"Validation score {self.validation.min_validation_score} must be between 0 and 1"
            )

        return issues

    @classmethod
    def from_env(cls, overrides: dict | None = None) -> AgentConfig:
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

_global_config: AgentConfig | None = None


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


# ==================== Convenience Functions ====================


def get_llm(temperature: float | None = None, max_tokens: int | None = None):
    """
    Get the configured LLM instance (OpenAI or Anthropic).

    This centralizes LLM creation to support provider switching.

    Args:
        temperature: Override default temperature (optional)
        max_tokens: Override default max_tokens (optional)

    Returns:
        ChatOpenAI or ChatAnthropic instance
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    config = get_config()

    temp = temperature if temperature is not None else config.llm.temperature
    tokens = max_tokens if max_tokens is not None else config.llm.max_tokens

    if config.llm.provider == "anthropic":
        return ChatAnthropic(
            model=config.llm.model,
            temperature=temp,
            max_tokens=tokens,
        )
    if config.llm.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.llm.model,
            temperature=temp,
            max_output_tokens=tokens,
        )
    # Default to OpenAI
    return ChatOpenAI(
        model=config.llm.model,
        temperature=temp,
        max_tokens=tokens,
        use_responses_api=config.llm.use_responses_api,
    )


def get_llm_for_role(
    role: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Return an LLM configured for a specific agent role (planner/developer/evaluator).

    Falls back to the global provider/model when a role-specific override is not set.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

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
    if provider == "gemini":
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
        use_responses_api=config.llm.use_responses_api,
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


# ==================== Metric Direction Utilities ====================


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

    # Metrics where lower values are better
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


def calculate_score_improvement(new_score: float, baseline_score: float, metric_name: str) -> float:
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
        0.050  # RMSE decreased from 0.400 to 0.350 (better)
        >>> calculate_score_improvement(0.85, 0.80, 'accuracy')
        0.050  # Accuracy increased from 0.80 to 0.85 (better)
        >>> calculate_score_improvement(0.450, 0.400, 'rmse')
        -0.050  # RMSE increased from 0.400 to 0.450 (worse)
    """
    is_minimize = is_metric_minimization(metric_name)

    if is_minimize:
        # For minimize metrics: lower new_score is better
        return baseline_score - new_score
    # For maximize metrics: higher new_score is better
    return new_score - baseline_score


def compare_scores(score1: float, score2: float, metric_name: str) -> float:
    """
    Compare two scores and return the better one.

    Args:
        search: Optional[str] = None
        score2: Second score to compare
        metric_name: Name of the evaluation metric

    Returns:
        The better score according to metric direction

    Examples:
        >>> compare_scores(0.350, 0.400, 'rmse')
        0.350  # Lower is better for RMSE
        >>> compare_scores(0.85, 0.80, 'accuracy')
        0.85  # Higher is better for accuracy
    """
    is_minimize = is_metric_minimization(metric_name)

    if is_minimize:
        return min(score1, score2)
    return max(score1, score2)
