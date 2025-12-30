"""
Advanced RL types for learning (WEBRL, Eureka, GRPO, DPO, Quiet-STaR).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

from .competition import AblationComponent


@dataclass
class SubTask:
    """
    WEBRL-style sub-task generated from failure.

    When the agent fails, creates specific sub-tasks to resolve
    the problem before proceeding.
    """

    parent_component: str
    failure_type: str  # "memory", "timeout", "syntax", "validation", etc.
    task_description: str
    priority: int  # 1 (highest) to 5 (lowest)
    status: Literal["pending", "in_progress", "resolved", "skipped"] = "pending"
    resolution_code: Optional[str] = None
    resolution_guidance: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CandidatePlan:
    """
    Eureka-style candidate plan with fitness score.

    Multiple plans are generated with different strategies,
    evaluated, and the best elements are combined.
    """

    components: list[AblationComponent] = field(default_factory=list)
    strategy: str = "balanced"  # "conservative", "aggressive", "balanced"
    fitness_score: float = 0.0
    generation: int = 0  # Evolutionary generation number
    execution_results: dict[str, float] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    """
    GRPO-style reasoning trace for code generation.

    Structured chain-of-thought before generating code,
    with process rewards for intermediate steps.
    """

    component_name: str
    requirements_analysis: str = ""
    potential_issues: list[str] = field(default_factory=list)
    solution_approach: str = ""
    implementation_plan: str = ""
    validation_checklist: list[str] = field(default_factory=list)
    step_scores: dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PreferencePair:
    """
    DPO-style preference pair for learning.

    Captures chosen (good) vs rejected (bad) code examples
    for preference-based optimization.
    """

    context: str  # Component/prompt description
    chosen: str  # Better code (succeeded)
    rejected: str  # Worse code (failed)
    margin: float = 0.0  # How much better is chosen (0-1)
    component_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SelfEvaluation:
    """
    Quiet-STaR style self-evaluation result.

    Internal reflection before finalizing code generation.
    """

    confidence: float = 0.0  # 0-1
    concerns: list[str] = field(default_factory=list)
    suggested_fixes: list[str] = field(default_factory=list)
    proceed: bool = True
    reflection_summary: str = ""
