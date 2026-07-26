"""
Specialized agents for autonomous Kaggle competition solving.
"""

from .developer_agent import DeveloperAgent, developer_agent_node


# The modular implementations are the only experiment implementations.
# Silently loading a legacy agent on ImportError changes the evaluated system
# across machines and can reintroduce policies that were not audited.
from .ensemble import EnsembleAgent, ensemble_agent_node

from .meta_evaluator_agent import MetaEvaluatorAgent, meta_evaluator_node


from .planner import PlannerAgent, planner_agent_node
from .robustness_agent import RobustnessAgent, robustness_agent_node
from .search_agent import SearchAgent, search_agent_node
from .submission_agent import SubmissionAgent, submission_agent_node


__all__ = [
    "DeveloperAgent",
    "EnsembleAgent",
    "MetaEvaluatorAgent",
    "PlannerAgent",
    "RobustnessAgent",
    "SearchAgent",
    "SubmissionAgent",
    "developer_agent_node",
    "ensemble_agent_node",
    "meta_evaluator_node",
    "planner_agent_node",
    "robustness_agent_node",
    "search_agent_node",
    "submission_agent_node",
]
