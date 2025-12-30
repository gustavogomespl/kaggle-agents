"""
Developer Agent - Backwards compatibility stub.

This module re-exports from the new modular developer package.
For new code, import directly from kaggle_agents.agents.developer.

Deprecated: Use `from kaggle_agents.agents.developer import DeveloperAgent` instead.
"""

from .developer import (
    CodeFixerModule,
    CodeGeneratorModule,
    DeveloperAgent,
    developer_agent_node,
)


__all__ = [
    "CodeFixerModule",
    "CodeGeneratorModule",
    "DeveloperAgent",
    "developer_agent_node",
]
