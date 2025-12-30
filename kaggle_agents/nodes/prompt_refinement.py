"""
Prompt Refinement Node with RL-based Optimization.

Uses DSPy MIPROv2 to optimize prompts based on collected training data.
Implements RLPrompt pattern for discrete prompt optimization.
"""

from typing import Any

from ..core.state import KaggleState
from ..optimization import create_optimizer, create_training_collector
from ..optimization.reward_model import (
    create_developer_metric,
    create_planner_metric,
)


class PromptRefinementDecider:
    """
    Decide when to trigger prompt refinement.

    Based on:
    - Iteration count (every N iterations)
    - Available training data
    - Performance plateau detection
    """

    def __init__(
        self,
        optimization_frequency: int = 5,
        min_training_examples: int = 5,
    ):
        """
        Initialize decider.

        Args:
            optimization_frequency: Re-optimize every N iterations
            min_training_examples: Minimum examples needed
        """
        self.optimization_frequency = optimization_frequency
        self.min_training_examples = min_training_examples
        self.training_collector = create_training_collector()

    def _analyze_performance_gaps(self, state: KaggleState) -> dict[str, bool]:
        """
        Analyze performance gaps to trigger adaptive optimization.

        Returns:
            Dict[str, bool]: {agent_name: should_trigger_immediately}
        """
        dev_results = state.get("development_results", [])
        if not dev_results:
            return {"planner": False, "developer": False}

        # Developer Check: High failure rate
        recent_results = dev_results[-5:]  # Look at last 5 attempts
        usage_count = len(recent_results)
        success_count = sum(1 for r in recent_results if r.success)

        developer_struggling = False
        if usage_count >= 3:
            success_rate = success_count / usage_count
            if success_rate < 0.3:  # Less than 30% success
                print(f"   ⚠️ Adaptive Trigger: Developer success rate low ({success_rate:.0%})")
                developer_struggling = True

        # Planner Check: Empty or invalid plans
        # We infer planner struggle if multiple components completely fail to implement
        # or if the plan was empty (though that usually crashes earlier)
        planner_struggling = False
        if usage_count >= 3 and success_count == 0:
            # If EVERYTHING failed recently, maybe the plan is bad?
            # This is a heuristic; consistent failure suggests bad strategy.
            print("   ⚠️ Adaptive Trigger: Consistent failure suggests strategic (Planner) issues")
            planner_struggling = True

        return {"planner": planner_struggling, "developer": developer_struggling}

    def should_optimize(self, state: KaggleState) -> dict[str, bool]:
        """
        Decide which agents should have prompts optimized.

        Args:
            state: Current workflow state

        Returns:
            Dictionary mapping agent names to optimization decisions
        """
        current_iteration = state.get("current_iteration", 0)

        # Don't optimize in early iterations (need data first)
        if current_iteration < 2:
            return {"planner": False, "developer": False}

        # Check for adaptive triggers (performance gaps)
        gaps = self._analyze_performance_gaps(state)

        # Check if it's time for periodic optimization
        is_optimization_cycle = current_iteration % self.optimization_frequency == 0

        if not is_optimization_cycle and not any(gaps.values()):
            return {"planner": False, "developer": False}

        print(f"\n🔄 Iteration {current_iteration}: Checking prompt optimization eligibility...")

        decisions = {}

        # Check planner
        planner_examples = self.training_collector.convert_to_dspy_examples(
            "planner",
            min_score=0.3,
        )
        # optimize if (cycle OR gap) AND enough data
        decisions["planner"] = (is_optimization_cycle or gaps["planner"]) and len(
            planner_examples
        ) >= self.min_training_examples

        if decisions["planner"]:
            reason = "Gap detected" if gaps["planner"] else "Cycle"
            print(f"   ✓ Planner: Optimize ({reason}), {len(planner_examples)} examples available")
        else:
            print(
                f"   ⏭️ Planner: Skipped (Data: {len(planner_examples)}/{self.min_training_examples})"
            )

        # Check developer
        developer_examples = self.training_collector.convert_to_dspy_examples(
            "developer_generator",
            min_score=0.5,
        )
        decisions["developer"] = (is_optimization_cycle or gaps["developer"]) and len(
            developer_examples
        ) >= self.min_training_examples

        if decisions["developer"]:
            reason = "Gap detected" if gaps["developer"] else "Cycle"
            print(
                f"   ✓ Developer: Optimize ({reason}), {len(developer_examples)} examples available"
            )
        else:
            print(
                f"   ⏭️ Developer: Skipped (Data: {len(developer_examples)}/{self.min_training_examples})"
            )

        return decisions


class PromptOptimizer:
    """
    Optimize prompts using DSPy MIPROv2.

    Implements RLPrompt pattern with stabilization techniques.
    """

    def __init__(self):
        """Initialize optimizer."""
        self.optimizer = create_optimizer()
        self.training_collector = create_training_collector()

    def optimize_planner_prompt(self) -> bool:
        """
        Optimize planner agent prompt.

        Returns:
            True if optimization succeeded, False otherwise
        """
        print("\n   🎯 Optimizing Planner prompt...")

        try:
            # Get training examples
            training_examples = self.training_collector.convert_to_dspy_examples(
                "planner",
                min_score=0.3,  # Include medium-quality plans for diversity
            )

            if len(training_examples) < 3:
                print("   ⚠️ Not enough training examples")
                return False

            print(f"   📊 Using {len(training_examples)} training examples")

            # Create metric
            metric = create_planner_metric()

            # Import planner module
            from ..agents.planner_agent import AblationPlannerModule

            # Optimize using DSPy MIPROv2
            optimized_module = self.optimizer.optimize_prompt(
                module=AblationPlannerModule(),
                trainset=training_examples,
                metric=metric,
                agent_name="planner",
            )

            if optimized_module is not None:
                # Save optimized module
                self.optimizer.save_optimized_prompt(optimized_module, "planner")
                print("   ✅ Planner prompt optimized and saved")
                return True
            print("   ❌ Optimization failed")
            return False

        except Exception as e:
            print(f"   ❌ Error optimizing planner: {e}")
            return False

    def optimize_developer_prompt(self) -> bool:
        """
        Optimize developer agent prompt.

        Returns:
            True if optimization succeeded, False otherwise
        """
        print("\n   🎯 Optimizing Developer prompt...")

        try:
            # Get training examples
            training_examples = self.training_collector.convert_to_dspy_examples(
                "developer_generator",
                min_score=0.5,  # Only successful code generation
            )

            if len(training_examples) < 3:
                print("   ⚠️ Not enough training examples")
                return False

            print(f"   📊 Using {len(training_examples)} training examples")

            # Create metric
            metric = create_developer_metric()

            # Import developer module
            from ..agents.developer_agent import CodeGeneratorModule

            # Optimize using DSPy MIPROv2
            optimized_module = self.optimizer.optimize_prompt(
                module=CodeGeneratorModule(),
                trainset=training_examples,
                metric=metric,
                agent_name="developer_generator",
            )

            if optimized_module is not None:
                # Save optimized module
                self.optimizer.save_optimized_prompt(optimized_module, "developer_generator")
                print("   ✅ Developer prompt optimized and saved")
                return True
            print("   ❌ Optimization failed")
            return False

        except Exception as e:
            print(f"   ❌ Error optimizing developer: {e}")
            return False


# ==================== Node Function ====================


def prompt_refinement_node(state: KaggleState) -> dict[str, Any]:
    """
    Check if prompts should be refined and optimize if needed.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    print("\n" + "=" * 60)
    print("= PROMPT REFINEMENT: RL-based Optimization")
    print("=" * 60)

    # Decide if optimization is needed
    decider = PromptRefinementDecider()
    optimization_decisions = decider.should_optimize(state)

    # Check if any optimization needed
    if not any(optimization_decisions.values()):
        print("\n⏭️ No prompt optimization needed at this iteration")
        return {}

    # Perform optimization
    optimizer = PromptOptimizer()
    results = {}

    if optimization_decisions["planner"]:
        success = optimizer.optimize_planner_prompt()
        results["planner_optimized"] = success

        if success:
            # Set flag to reload planner
            results["reload_planner"] = True

    if optimization_decisions["developer"]:
        success = optimizer.optimize_developer_prompt()
        results["developer_optimized"] = success

        if success:
            # Set flag to reload developer
            results["reload_developer"] = True

    print("\n✅ Prompt refinement completed")
    print(f"   Results: {results}")

    return results


def should_refine_prompts(state: KaggleState) -> bool:
    """
    Quick check if prompt refinement should run.

    Args:
        state: Current workflow state

    Returns:
        True if should run refinement node
    """
    current_iteration = state.get("current_iteration", 0)

    # Only check every 5 iterations after iteration 2
    if current_iteration < 2:
        return False

    return current_iteration % 5 == 0
