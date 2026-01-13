"""
Meta-Evaluator Agent with Reinforcement Learning.

This agent analyzes code generation results and optimize prompts for other agents using RL techniques.

Based on:
- CodeRL+: Execution Semantics Alignment
- PREFACE: Error-guided prompt repair
- RLPrompt: Discrete prompt optimization
- ML-Agent: RL for ML engineering
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from ...core.config import get_config, get_llm_for_role
from ...optimization import create_training_collector
from .analysis import AnalysisMixin
from .detection import DetectionMixin
from .eureka import EurekaMixin
from .guidance import GuidanceMixin
from .memory import MemoryMixin
from .rewards import RewardsMixin
from .training import TrainingMixin


if TYPE_CHECKING:
    from ...core.state import KaggleState


class MetaEvaluatorAgent(
    AnalysisMixin,
    DetectionMixin,
    RewardsMixin,
    GuidanceMixin,
    EurekaMixin,
    TrainingMixin,
    MemoryMixin,
):
    """
    Meta-agent that evaluates other agents and optimizes their prompts using RL.

    Features:
    - Analyzes code generation failures and successes
    - Extracts error patterns (PREFACE pattern)
    - Calculates reward signals (CodeRL+ pattern)
    - Generates refinement guidance for prompt optimization
    - Collects training data for DSPy optimization
    """

    def __init__(self):
        """Initialize meta-evaluator with configured model."""
        self.config = get_config()

        # Use configured LLM (supports OpenAI and Anthropic)
        self.llm = get_llm_for_role(role="evaluator")

        provider = self.config.llm.provider.upper()
        model = self.config.llm.model
        print(f"   🧠 Meta-Evaluator initialized with {provider} ({model})")

        # Training data collector for RL
        self.training_collector = create_training_collector()

    def __call__(self, state: KaggleState) -> dict[str, Any]:
        """
        Execute meta-evaluation after performance evaluation.

        Args:
            state: Current workflow state

        Returns:
            State updates with failure analysis and refinement guidance
        """
        print("\n" + "=" * 60)
        print("= META-EVALUATOR: Analyzing Performance & Optimizing Prompts")
        print("=" * 60)

        current_iteration = state.get("current_iteration", 0)
        print(f"\n📊 Iteration: {current_iteration}")

        # Analyze component performance
        failure_analysis = self._analyze_failures(state)

        # Calculate reward signals (CodeRL+ pattern)
        reward_signals = self._calculate_reward_signals(state, failure_analysis)

        # Generate refinement guidance (PREFACE pattern)
        refinement_guidance = self._generate_refinement_guidance(
            state, failure_analysis, reward_signals
        )

        # Create iteration memory for learning
        iteration_memory = self._create_iteration_memory(state, failure_analysis, reward_signals)

        # Collect training data for DSPy optimization
        self._collect_training_data(state, failure_analysis, reward_signals)

        # Eureka: Perform evolutionary crossover for next generation planning
        crossover_guidance = self._evolutionary_crossover(state)

        # Inner Loop Refinement: Check for performance gaps that need debug loops
        debug_loop_trigger = self._check_performance_gap_for_debug(state)

        # Detect stagnation for SOTA search trigger
        stagnation_detection = self._detect_stagnation(state)

        # Update state
        debug_updates = {}
        if debug_loop_trigger.get("trigger_debug"):
            debug_updates = {
                "trigger_debug_loop": True,
                "debug_target_model": debug_loop_trigger.get("worst_model"),
                "debug_hints": debug_loop_trigger.get("debug_hints", []),
                "performance_gap": debug_loop_trigger.get("gap"),
            }
            print(f"\n   ⚠️  TRIGGERING DEBUG LOOP for {debug_loop_trigger.get('worst_model')}")

        result = {
            "failure_analysis": failure_analysis,
            "reward_signals": reward_signals,
            "refinement_guidance": refinement_guidance,
            "crossover_guidance": crossover_guidance,  # Eureka: for planner
            "stagnation_detection": stagnation_detection,  # For SOTA search trigger
            "iteration_memory": [iteration_memory],  # Append to list
            "last_updated": datetime.now(),
        }
        result.update(debug_updates)  # Add debug loop trigger if applicable
        return result


def meta_evaluator_node(state: KaggleState) -> dict[str, Any]:
    """
    LangGraph node function for meta-evaluation.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = MetaEvaluatorAgent()
    return agent(state)
