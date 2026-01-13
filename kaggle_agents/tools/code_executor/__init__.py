"""
Code Executor module.

Provides sandboxed Python code execution with:
- Subprocess isolation
- Timeout handling
- Error parsing and categorization
- Artifact validation
- Resource monitoring
- Real-time stdout streaming for training logs
"""

from .artifact_validator import ArtifactValidator
from .dataclasses import ExecutionProgress, ExecutionResult
from .executor import CodeExecutor
from .process import (
    ENABLE_RESOURCE_LIMITS,
    kill_process_tree,
    set_resource_limits,
    start_new_process_group,
)


def execute_code(
    code: str,
    working_dir: str,
    timeout: int = 300,
) -> ExecutionResult:
    """
    Convenience function to execute code.

    Args:
        code: Python code
        working_dir: Working directory
        timeout: Timeout in seconds

    Returns:
        ExecutionResult
    """
    executor = CodeExecutor(timeout=timeout)
    return executor.execute(code, working_dir)


def validate_code_syntax(code: str) -> tuple:
    """
    Convenience function to validate syntax.

    Args:
        code: Python code

    Returns:
        Tuple of (is_valid, error_message)
    """
    executor = CodeExecutor()
    return executor.validate_syntax(code)


__all__ = [
    "CodeExecutor",
    "ExecutionResult",
    "ExecutionProgress",
    "ArtifactValidator",
    "execute_code",
    "validate_code_syntax",
    # Process utilities
    "ENABLE_RESOURCE_LIMITS",
    "set_resource_limits",
    "start_new_process_group",
    "kill_process_tree",
]
