"""State management for Kaggle agents workflow."""

from typing import Annotated, List, Dict, Any
from operator import add
from dataclasses import dataclass, field
from langgraph.graph import MessagesState


def merge_dict(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, with right values taking precedence."""
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


@dataclass
class KaggleState(MessagesState):
    """State for the Kaggle multi-agent workflow.

    Uses dataclass for better performance and dot-notation access.
    Implements custom reducers for list and dict merging.
    """

    # Competition metadata
    competition_name: str = ""
    competition_type: str = ""
    metric: str = ""

    # Data paths
    train_data_path: str = ""
    test_data_path: str = ""
    sample_submission_path: str = ""

    # EDA results - uses custom reducer to merge dictionaries
    eda_summary: Annotated[Dict[str, Any], merge_dict] = field(default_factory=dict)
    data_insights: Annotated[List[str], add] = field(default_factory=list)

    # Feature engineering - uses add operator to append to list
    features_engineered: Annotated[List[str], add] = field(default_factory=list)
    feature_importance: Annotated[Dict[str, float], merge_dict] = field(default_factory=dict)

    # Model training - uses add operator to append to list
    models_trained: Annotated[List[Dict[str, Any]], add] = field(default_factory=list)
    best_model: Annotated[Dict[str, Any], merge_dict] = field(default_factory=dict)
    cv_scores: Annotated[List[float], add] = field(default_factory=list)

    # Submission
    submission_path: str = ""
    submission_score: float = 0.0
    leaderboard_rank: int = 0

    # Workflow control
    iteration: int = 0
    max_iterations: int = 5

    # Error tracking - uses add operator to append to list
    errors: Annotated[List[str], add] = field(default_factory=list)
