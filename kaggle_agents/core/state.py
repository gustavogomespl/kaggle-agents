"""Enhanced state management with memory and phase tracking."""

import json
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from operator import add
from pathlib import Path
from langgraph.graph import MessagesState


def merge_dict(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, with right values taking precedence."""
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


class EnhancedKaggleState(MessagesState):
    """Enhanced state with memory management and phase tracking.

    This is a TypedDict for compatibility with LangGraph's state management.
    All access should be via dict operations: state["field"] or state.get("field", default)
    """

    # ========== Competition Metadata ==========
    competition_name: str
    competition_type: str
    metric: str
    competition_dir: str

    # ========== Data Paths ==========
    train_data_path: str
    test_data_path: str
    sample_submission_path: str

    # ========== EDA Results ==========
    eda_summary: Annotated[Dict[str, Any], merge_dict]
    data_insights: Annotated[List[str], add]

    # ========== Feature Engineering ==========
    features_engineered: Annotated[List[str], add]
    feature_importance: Annotated[Dict[str, float], merge_dict]

    # ========== Model Training ==========
    models_trained: Annotated[List[Dict[str, Any]], add]
    best_model: Annotated[Dict[str, Any], merge_dict]
    cv_scores: Annotated[List[float], add]

    # ========== Submission ==========
    submission_path: str
    submission_score: float
    leaderboard_rank: int

    # ========== Workflow Control ==========
    iteration: int
    max_iterations: int

    # ========== Error Tracking ==========
    errors: Annotated[List[str], add]

    # ========== Enhanced Fields ==========
    phase: str
    memory: Annotated[List[Dict[str, Any]], add]
    background_info: str
    rules: Dict[str, Any]
    retry_count: int
    max_phase_retries: int
    status: str


# Phase to directory mapping (module-level constant)
PHASE_TO_DIRECTORY = {
    "Understand Background": "background",
    "Preliminary Exploratory Data Analysis": "pre_eda",
    "Data Cleaning": "data_cleaning",
    "In-depth Exploratory Data Analysis": "deep_eda",
    "Feature Engineering": "feature_engineering",
    "Model Building, Validation, and Prediction": "model_build_predict"
}

# Context phases (module-level constant)
CONTEXT_PHASES = [
    "Understand Background",
    "Preliminary Exploratory Data Analysis",
    "Data Cleaning",
    "In-depth Exploratory Data Analysis",
    "Feature Engineering",
    "Model Building, Validation, and Prediction"
]


# Helper functions that work with state dict
def get_restore_dir(state: EnhancedKaggleState) -> Path:
    """Get the directory for storing phase-specific files."""
    phase = state.get("phase", "")
    competition_dir = state.get("competition_dir", ".")
    dir_name = PHASE_TO_DIRECTORY.get(phase, "unknown")
    path = Path(competition_dir) / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_dir_name(state: EnhancedKaggleState) -> str:
    """Get the directory name for current phase."""
    phase = state.get("phase", "")
    return PHASE_TO_DIRECTORY.get(phase, "unknown")


def get_state_info(state: EnhancedKaggleState) -> str:
    """Generate formatted state information."""
    info = f"""# STATE INFORMATION #
Phase: {state.get('phase', '')}
Directory: {get_dir_name(state)}
Competition: {state.get('competition_name', '')}
Competition Type: {state.get('competition_type', '')}
Metric: {state.get('metric', '')}
Retry Count: {state.get('retry_count', 0)}/{state.get('max_phase_retries', 3)}
Iteration: {state.get('iteration', 0)}/{state.get('max_iterations', 5)}
"""
    background = state.get("background_info", "")
    if background:
        info += f"\nBackground:\n{background[:500]}..."

    return info


def get_previous_phase(state: EnhancedKaggleState, type: str = "all") -> List[str]:
    """Get list of previous phases."""
    current_phase = state.get("phase", "")
    current_idx = CONTEXT_PHASES.index(current_phase) if current_phase in CONTEXT_PHASES else -1

    if current_idx <= 0:
        return []

    previous_phases = CONTEXT_PHASES[:current_idx]

    if type == "plan":
        plan_phases = [
            "Preliminary Exploratory Data Analysis",
            "Data Cleaning",
            "In-depth Exploratory Data Analysis",
            "Feature Engineering",
            "Model Building, Validation, and Prediction"
        ]
        return [p for p in previous_phases if p in plan_phases]
    elif type == "code":
        code_phases = [
            "Data Cleaning",
            "Feature Engineering",
            "Model Building, Validation, and Prediction"
        ]
        return [p for p in previous_phases if p in code_phases]

    return previous_phases


def set_background_info(state: EnhancedKaggleState, info: str) -> None:
    """Set background information for the competition."""
    state["background_info"] = info


def generate_rules(state: EnhancedKaggleState) -> str:
    """Generate formatted rules for agents."""
    rules = state.get("rules", {})
    if not rules:
        return "No specific rules configured."

    rules_str = "# RULES #\n"
    for key, value in rules.items():
        rules_str += f"- {key}: {value}\n"

    return rules_str


def add_memory(state: EnhancedKaggleState, phase_results: Dict[str, Any]) -> None:
    """Add results from a phase execution to memory."""
    memory_entry = {
        "phase": state.get("phase", ""),
        "iteration": state.get("iteration", 0),
        "retry_count": state.get("retry_count", 0),
        **phase_results
    }

    memory = state.get("memory", [])
    memory.append(memory_entry)
    state["memory"] = memory


def reset_retry_count(state: EnhancedKaggleState) -> None:
    """Reset retry count when moving to new phase."""
    state["retry_count"] = 0


def increment_retry_count(state: EnhancedKaggleState) -> None:
    """Increment retry count."""
    state["retry_count"] = state.get("retry_count", 0) + 1


def should_retry_phase(state: EnhancedKaggleState) -> bool:
    """Check if current phase should be retried."""
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_phase_retries", 3)
    return retry_count < max_retries


def next_phase(state: EnhancedKaggleState) -> None:
    """Move to the next phase in the workflow."""
    current_phase = state.get("phase", "")
    current_idx = CONTEXT_PHASES.index(current_phase) if current_phase in CONTEXT_PHASES else -1

    if current_idx < len(CONTEXT_PHASES) - 1:
        state["phase"] = CONTEXT_PHASES[current_idx + 1]
        reset_retry_count(state)
    else:
        # Workflow complete
        state["phase"] = "Complete"


def save_to_disk(state: EnhancedKaggleState) -> None:
    """Save state to disk for debugging and recovery."""
    restore_dir = get_restore_dir(state)
    state_file = restore_dir / "state.json"

    # Create a serializable copy
    state_copy = dict(state)
    # Remove non-serializable items
    if "messages" in state_copy:
        state_copy["messages"] = []  # Messages are too complex to serialize

    with open(state_file, 'w') as f:
        json.dump(state_copy, f, indent=2)


# Backward compatibility
KaggleState = EnhancedKaggleState
