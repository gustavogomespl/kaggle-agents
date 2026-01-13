"""
Code Executor Tool - Backward Compatibility Shim.

This module now re-exports from the code_executor submodule.
See kaggle_agents/tools/code_executor/ for the implementation.
"""

# Re-export for backward compatibility
from .code_executor import (
    ENABLE_RESOURCE_LIMITS,
    ArtifactValidator,
    CodeExecutor,
    ExecutionProgress,
    ExecutionResult,
    execute_code,
    kill_process_tree,
    set_resource_limits,
    start_new_process_group,
    validate_code_syntax,
)


# Private function aliases for backward compatibility
_set_resource_limits = set_resource_limits
_start_new_process_group = start_new_process_group
_kill_process_tree = kill_process_tree

__all__ = [
    "CodeExecutor",
    "ExecutionResult",
    "ExecutionProgress",
    "ArtifactValidator",
    "execute_code",
    "validate_code_syntax",
    "ENABLE_RESOURCE_LIMITS",
    # Private functions (kept for potential internal usage)
    "_set_resource_limits",
    "_start_new_process_group",
    "_kill_process_tree",
]
