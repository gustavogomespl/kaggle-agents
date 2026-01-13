"""
Specialized agents for autonomous Kaggle competition solving.
"""

from .developer_agent import DeveloperAgent, developer_agent_node


# Import EnsembleAgent from new modular structure (with fallback to old file)
try:
    from .ensemble import EnsembleAgent, ensemble_agent_node
except ImportError:
    from .ensemble_agent import EnsembleAgent, ensemble_agent_node

from .meta_evaluator_agent import MetaEvaluatorAgent, meta_evaluator_node


# Import PlannerAgent from new modular structure (with fallback to old file)
try:
    from .planner import PlannerAgent, planner_agent_node
except ImportError:
    from .planner_agent import PlannerAgent, planner_agent_node
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
