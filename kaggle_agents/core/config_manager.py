"""Configuration management for enhanced workflow."""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage configuration for the enhanced Kaggle agents workflow."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to config.json file (default: project root)
        """
        if config_path is None:
            # Look for config.json in project root
            config_path = Path(__file__).parent.parent.parent / "config.json"

        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}

        self.load()

    def load(self):
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            logger.info("Using default configuration")
            self.config = self._get_default_config()
            return

        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.info("Using default configuration")
            self.config = self._get_default_config()

    def save(self, output_path: Optional[Path] = None):
        """Save configuration to file.

        Args:
            output_path: Optional output path (default: use config_path)
        """
        if output_path is None:
            output_path = self.config_path

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(self.config, f, indent=2)

            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key (supports nested keys with dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value.

        Args:
            key: Configuration key (supports nested keys with dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    # Phase configuration
    def get_phases(self) -> List[str]:
        """Get list of workflow phases."""
        return self.get("phases", [])

    def get_phase_directory(self, phase: str) -> str:
        """Get directory name for a phase."""
        return self.get("phase_to_directory", {}).get(phase, "unknown")

    def get_phase_agents(self, phase: str) -> List[str]:
        """Get list of agents for a phase."""
        return self.get("phase_to_agents", {}).get(phase, [])

    def get_phase_tools(self, phase: str) -> List[str]:
        """Get list of ML tools available for a phase."""
        return self.get("phase_to_ml_tools", {}).get(phase, [])

    # Retry settings
    def get_max_phase_retries(self) -> int:
        """Get maximum number of phase retries."""
        return self.get("retry_settings.max_phase_retries", 3)

    def get_max_code_retries(self) -> int:
        """Get maximum number of code generation retries."""
        return self.get("retry_settings.max_code_retries", 5)

    def get_max_debug_iterations(self) -> int:
        """Get maximum number of debugging iterations."""
        return self.get("retry_settings.max_debug_iterations", 10)

    def get_retry_delay(self) -> int:
        """Get delay between retries in seconds."""
        return self.get("retry_settings.retry_delay_seconds", 30)

    # Model settings
    def get_model_for_agent(self, agent_role: str) -> str:
        """Get model name for specific agent.

        Args:
            agent_role: Agent role (e.g., 'planner', 'developer')

        Returns:
            Model name (e.g., 'gpt-5-mini')
        """
        key = f"model_settings.{agent_role}_model"
        return self.get(key, self.get("model_settings.default_model", "gpt-5-mini"))

    def get_temperature(self) -> float:
        """Get LLM temperature setting."""
        model = self.get("model_settings.default_model", "gpt-5-mini")

        # gpt-5-mini only supports default temperature (1.0)
        if model == "gpt-5-mini":
            return 1.0

        return self.get("model_settings.temperature", 0.7)

    def get_max_tokens(self) -> int:
        """Get maximum tokens for generation."""
        return self.get("model_settings.max_tokens", 16000)

    # Execution settings
    def get_code_timeout(self) -> int:
        """Get code execution timeout in seconds."""
        return self.get("execution_settings.code_timeout_seconds", 300)

    def is_sandboxing_enabled(self) -> bool:
        """Check if code sandboxing is enabled."""
        return self.get("execution_settings.enable_sandboxing", True)

    def should_save_intermediate_results(self) -> bool:
        """Check if intermediate results should be saved."""
        return self.get("execution_settings.save_intermediate_results", True)

    def is_verbose_logging(self) -> bool:
        """Check if verbose logging is enabled."""
        return self.get("execution_settings.verbose_logging", True)

    # User interaction
    def is_user_interaction_enabled(self, interaction_type: str) -> bool:
        """Check if user interaction is enabled for a type.

        Args:
            interaction_type: Type of interaction ('plan', 'code', 'review')

        Returns:
            True if enabled
        """
        return self.get(f"user_interaction.{interaction_type}", False)

    # Workflow mode
    def get_workflow_mode(self) -> str:
        """Get workflow mode ('simple' or 'enhanced')."""
        return self.get("workflow_mode.mode", "enhanced")

    def is_feedback_loops_enabled(self) -> bool:
        """Check if feedback loops are enabled."""
        return self.get("workflow_mode.enable_feedback_loops", True)

    def is_tool_retrieval_enabled(self) -> bool:
        """Check if tool retrieval is enabled."""
        return self.get("workflow_mode.enable_tool_retrieval", True)

    def is_memory_enabled(self) -> bool:
        """Check if memory system is enabled."""
        return self.get("workflow_mode.enable_memory", True)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "phases": [
                "Understand Background",
                "Preliminary Exploratory Data Analysis",
                "Data Cleaning",
                "In-depth Exploratory Data Analysis",
                "Feature Engineering",
                "Model Building, Validation, and Prediction",
            ],
            "phase_to_directory": {
                "Understand Background": "background",
                "Preliminary Exploratory Data Analysis": "pre_eda",
                "Data Cleaning": "data_cleaning",
                "In-depth Exploratory Data Analysis": "deep_eda",
                "Feature Engineering": "feature_engineering",
                "Model Building, Validation, and Prediction": "model_build_predict",
            },
            "phase_to_agents": {
                "Understand Background": ["planner"],
                "Preliminary Exploratory Data Analysis": ["eda"],
                "Data Cleaning": ["cleaner"],
                "In-depth Exploratory Data Analysis": ["eda"],
                "Feature Engineering": ["feature_engineer"],
                "Model Building, Validation, and Prediction": ["modeler"],
            },
            "retry_settings": {
                "max_phase_retries": 3,
                "max_code_retries": 5,
                "max_debug_iterations": 10,
            },
            "model_settings": {
                "default_model": "gpt-5-mini",
                "temperature": 1.0,  # gpt-5-mini only supports default temperature
                "max_tokens": 16000,
            },
            "workflow_mode": {"mode": "enhanced"},
        }


# Global config instance
_config_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration instance.

    Returns:
        ConfigManager instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = ConfigManager()

    return _config_instance


if __name__ == "__main__":
    # Test configuration
    config = ConfigManager()

    print("Phases:", config.get_phases())
    print("Max retries:", config.get_max_phase_retries())
    print("Planner model:", config.get_model_for_agent("planner"))
    print("Code timeout:", config.get_code_timeout())
    print("Workflow mode:", config.get_workflow_mode())

    # Test nested get
    print("\nPhase tools for Data Cleaning:")
    print(config.get_phase_tools("Data Cleaning"))
