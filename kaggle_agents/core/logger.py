"""
Centralized logging system for Kaggle Agents.

Clean, simple, and efficient logging with file rotation and structured output.
"""

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


class KaggleLogger:
    """
    Centralized logger for Kaggle Agents.

    Features:
    - Console output with Rich formatting
    - File output with rotation
    - Structured log messages
    - Configurable log levels
    """

    _instance: Optional["KaggleLogger"] = None
    _initialized: bool = False

    def __new__(cls):
        """Singleton pattern to ensure one logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger (only once)."""
        if self._initialized:
            return

        self.logger = logging.getLogger("kaggle_agents")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Remove existing handlers
        self.logger.handlers.clear()

        self._initialized = True

    def setup(
        self,
        log_dir: str = "./logs",
        log_level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
    ):
        """
        Setup logging configuration.

        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_console: Enable console output
            enable_file: Enable file output
        """
        # Clear existing handlers
        self.logger.handlers.clear()

        # Set log level
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)

        # Console handler with Rich
        if enable_console:
            console_handler = RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_path=False,
            )
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(
                "%(message)s",
                datefmt="[%X]",
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler with rotation
        if enable_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Main log file
            log_file = log_path / f"kaggle_agents_{datetime.now().strftime('%Y%m%d')}.log"

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(level)

            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a child logger for a specific module.

        Args:
            name: Module name (e.g., "search_agent")

        Returns:
            Logger instance
        """
        return self.logger.getChild(name)


# Global logger instance
_logger_instance = KaggleLogger()


def setup_logging(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
):
    """
    Setup global logging configuration.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_console: Enable console output
        enable_file: Enable file output
    """
    _logger_instance.setup(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file,
    )


def get_logger(name: str = "main") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically module name)

    Returns:
        Logger instance

    Example:
        logger = get_logger("search_agent")
        logger.info("Starting SOTA search")
        logger.error("Failed to connect", exc_info=True)
    """
    return _logger_instance.get_logger(name)


class LogContext:
    """
    Context manager for logging workflow stages.

    Example:
        with LogContext("search_agent", "Searching SOTA solutions"):
            # Do work
            logger.info("Found 10 notebooks")
    """

    def __init__(self, logger_name: str, stage_name: str):
        """
        Initialize log context.

        Args:
            logger_name: Logger name
            stage_name: Stage description
        """
        self.logger = get_logger(logger_name)
        self.stage_name = stage_name
        self.start_time = None

    def __enter__(self):
        """Enter context - log start."""
        self.start_time = datetime.now()
        self.logger.info(f"[bold blue]→ {self.stage_name}[/bold blue]")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - log completion or error."""
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.info(
                f"[bold green]✓ {self.stage_name} completed[/bold green] ({duration:.2f}s)"
            )
        else:
            self.logger.error(
                f"[bold red]✗ {self.stage_name} failed[/bold red] ({duration:.2f}s): {exc_val}"
            )

        return False  # Re-raise exception


def log_agent_start(logger: logging.Logger, agent_name: str):
    """Log agent execution start."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"AGENT: {agent_name}")
    logger.info(f"{'=' * 60}")


def log_agent_end(logger: logging.Logger, agent_name: str, success: bool = True):
    """Log agent execution end."""
    status = "✓ COMPLETE" if success else "✗ FAILED"
    logger.info(f"{status}: {agent_name}\n")


def log_metric(logger: logging.Logger, name: str, value: any):
    """Log a metric in a consistent format."""
    logger.info(f"  {name}: {value}")


def log_error_with_context(logger: logging.Logger, error: Exception, context: str):
    """Log error with additional context."""
    logger.error(f"Error in {context}: {error!s}", exc_info=True)
