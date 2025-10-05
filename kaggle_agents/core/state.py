"""Enhanced state management with memory and phase tracking."""

import json
from typing import Annotated, List, Dict, Any, Optional
from operator import add
from dataclasses import dataclass, field
from pathlib import Path
from langgraph.graph import MessagesState


def merge_dict(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, with right values taking precedence."""
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


@dataclass
class EnhancedKaggleState(MessagesState):
    """Enhanced state with memory management and phase tracking.

    Combines LangGraph's MessagesState with AutoKaggle's memory patterns.
    Supports both dict-style and dot-notation access for compatibility.
    """

    # ========== Competition Metadata ==========
    competition_name: str = ""
    competition_type: str = ""
    metric: str = ""
    competition_dir: str = ""

    # ========== Data Paths ==========
    train_data_path: str = ""
    test_data_path: str = ""
    sample_submission_path: str = ""

    # ========== EDA Results ==========
    eda_summary: Annotated[Dict[str, Any], merge_dict] = field(default_factory=dict)
    data_insights: Annotated[List[str], add] = field(default_factory=list)

    # ========== Feature Engineering ==========
    features_engineered: Annotated[List[str], add] = field(default_factory=list)
    feature_importance: Annotated[Dict[str, float], merge_dict] = field(default_factory=dict)

    # ========== Model Training ==========
    models_trained: Annotated[List[Dict[str, Any]], add] = field(default_factory=list)
    best_model: Annotated[Dict[str, Any], merge_dict] = field(default_factory=dict)
    cv_scores: Annotated[List[float], add] = field(default_factory=list)

    # ========== Submission ==========
    submission_path: str = ""
    submission_score: float = 0.0
    leaderboard_rank: int = 0

    # ========== Workflow Control ==========
    iteration: int = 0
    max_iterations: int = 5

    # ========== Error Tracking ==========
    errors: Annotated[List[str], add] = field(default_factory=list)

    # ========== Enhanced Fields (AutoKaggle Pattern) ==========

    # Phase management
    phase: str = "Understand Background"
    phase_to_directory: Dict[str, str] = field(default_factory=lambda: {
        "Understand Background": "background",
        "Preliminary Exploratory Data Analysis": "pre_eda",
        "Data Cleaning": "data_cleaning",
        "In-depth Exploratory Data Analysis": "deep_eda",
        "Feature Engineering": "feature_engineering",
        "Model Building, Validation, and Prediction": "model_build_predict"
    })

    # Memory: List of dictionaries, one per phase execution
    # Each entry contains agent outputs: {"planner": {...}, "developer": {...}, "reviewer": {...}}
    memory: Annotated[List[Dict[str, Any]], add] = field(default_factory=list)

    # Background information for the competition
    background_info: str = ""

    # Configuration rules
    rules: Dict[str, Any] = field(default_factory=dict)

    # Current phase retry count
    retry_count: int = 0
    max_phase_retries: int = 3

    # Context: phases that should be considered for planning
    context: List[str] = field(default_factory=lambda: [
        "Understand Background",
        "Preliminary Exploratory Data Analysis",
        "Data Cleaning",
        "In-depth Exploratory Data Analysis",
        "Feature Engineering",
        "Model Building, Validation, and Prediction"
    ])

    # Workflow status for routing
    status: str = "Continue"

    @property
    def restore_dir(self) -> Path:
        """Get the directory for storing phase-specific files."""
        dir_name = self.phase_to_directory.get(self.phase, "unknown")
        path = Path(self.competition_dir) / dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def dir_name(self) -> str:
        """Get the directory name for current phase."""
        return self.phase_to_directory.get(self.phase, "unknown")

    def get_state_info(self) -> str:
        """Generate formatted state information for agents.

        Returns:
            Formatted string with current phase, directory, and context
        """
        info = f"""# STATE INFORMATION #
Phase: {self.phase}
Directory: {self.dir_name}
Competition: {self.competition_name}
Competition Type: {self.competition_type}
Metric: {self.metric}
Retry Count: {self.retry_count}/{self.max_phase_retries}
Iteration: {self.iteration}/{self.max_iterations}
"""
        if self.background_info:
            info += f"\nBackground:\n{self.background_info[:500]}..."

        return info

    def get_previous_phase(self, type: str = "all") -> List[str]:
        """Get list of previous phases.

        Args:
            type: Filter type - "all", "plan", or "code"

        Returns:
            List of previous phase names
        """
        current_idx = self.context.index(self.phase) if self.phase in self.context else -1

        if current_idx <= 0:
            return []

        previous_phases = self.context[:current_idx]

        if type == "plan":
            # Only phases that involve planning
            plan_phases = [
                "Preliminary Exploratory Data Analysis",
                "Data Cleaning",
                "In-depth Exploratory Data Analysis",
                "Feature Engineering",
                "Model Building, Validation, and Prediction"
            ]
            return [p for p in previous_phases if p in plan_phases]
        elif type == "code":
            # Only phases that involve code development
            code_phases = [
                "Data Cleaning",
                "Feature Engineering",
                "Model Building, Validation, and Prediction"
            ]
            return [p for p in previous_phases if p in code_phases]

        return previous_phases

    def set_background_info(self, info: str):
        """Set background information for the competition."""
        self.background_info = info

    def generate_rules(self) -> str:
        """Generate formatted rules for agents.

        Returns:
            Formatted string with configuration rules
        """
        if not self.rules:
            return "No specific rules configured."

        rules_str = "# RULES #\n"
        for key, value in self.rules.items():
            rules_str += f"- {key}: {value}\n"

        return rules_str

    def add_memory(self, phase_results: Dict[str, Any]):
        """Add results from a phase execution to memory.

        Args:
            phase_results: Dictionary with agent outputs
        """
        memory_entry = {
            "phase": self.phase,
            "iteration": self.iteration,
            "retry_count": self.retry_count,
            **phase_results
        }
        self.memory.append(memory_entry)

    def get_last_agent_output(self, agent_role: str) -> Optional[Dict[str, Any]]:
        """Get the last output from a specific agent.

        Args:
            agent_role: Role of the agent (e.g., "planner", "developer")

        Returns:
            Agent output dictionary or None if not found
        """
        for memory_entry in reversed(self.memory):
            if agent_role in memory_entry:
                return memory_entry[agent_role]
        return None

    def save_to_disk(self, filepath: Optional[Path] = None):
        """Save state to disk for persistence.

        Args:
            filepath: Path to save state (default: competition_dir/state.json)
        """
        if filepath is None:
            filepath = Path(self.competition_dir) / "state.json"

        # Convert to dict for JSON serialization
        state_dict = {
            "competition_name": self.competition_name,
            "competition_type": self.competition_type,
            "metric": self.metric,
            "phase": self.phase,
            "iteration": self.iteration,
            "retry_count": self.retry_count,
            "memory": self.memory,
            "background_info": self.background_info,
            "rules": self.rules,
        }

        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)

    @classmethod
    def load_from_disk(cls, filepath: Path) -> 'EnhancedKaggleState':
        """Load state from disk.

        Args:
            filepath: Path to state file

        Returns:
            Loaded state object
        """
        with open(filepath, 'r') as f:
            state_dict = json.load(f)

        return cls(**state_dict)

    def reset_retry_count(self):
        """Reset retry count when moving to new phase."""
        self.retry_count = 0

    def increment_retry_count(self):
        """Increment retry count."""
        self.retry_count += 1

    def should_retry_phase(self) -> bool:
        """Check if current phase should be retried.

        Returns:
            True if retry count is below max
        """
        return self.retry_count < self.max_phase_retries

    def next_phase(self):
        """Move to the next phase in the workflow."""
        current_idx = self.context.index(self.phase) if self.phase in self.context else -1

        if current_idx < len(self.context) - 1:
            self.phase = self.context[current_idx + 1]
            self.reset_retry_count()
        else:
            # Workflow complete
            self.phase = "Complete"


# Backward compatibility: Keep original KaggleState for simple workflow
KaggleState = EnhancedKaggleState
