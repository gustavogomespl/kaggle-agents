"""
DSPy-based Prompt Optimization System.

This module provides automatic prompt optimization using DSPy's MIPROv2
optimizer and custom reward models based on Kaggle competition scores.
"""

import json
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2

from ..core.config import get_config


class PromptOptimizer:
    """
    Optimizer for agent prompts using DSPy.

    This class manages:
    - DSPy configuration and LM setup
    - Training data collection
    - Prompt optimization via MIPROv2
    - Prompt persistence and loading
    """

    def __init__(self):
        """Initialize the prompt optimizer."""
        self.config = get_config()
        self._setup_dspy()

    def _setup_dspy(self) -> None:
        """Configure DSPy with the appropriate language model."""
        # Get safe max_tokens based on model
        max_tokens = self.config.llm.max_tokens

        # Configure LM based on provider
        if self.config.llm.provider == "openai":
            lm = dspy.LM(
                model=f"openai/{self.config.llm.model}",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=max_tokens,
                temperature=self.config.llm.temperature,
            )
        elif self.config.llm.provider == "anthropic":
            # DSPy supports Anthropic via LiteLLM
            lm = dspy.LM(
                model=f"anthropic/{self.config.llm.model}",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                max_tokens=max_tokens,
                temperature=self.config.llm.temperature,
            )
        elif self.config.llm.provider == "gemini":
            # DSPy supports Google Gemini via LiteLLM
            lm = dspy.LM(
                model=f"gemini/{self.config.llm.model}",
                api_key=os.getenv("GOOGLE_API_KEY"),
                max_tokens=max_tokens,
                temperature=self.config.llm.temperature,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")

        # Configure DSPy
        dspy.settings.configure(lm=lm)

        print(f" DSPy configured with {self.config.llm.provider}/{self.config.llm.model}")

    def optimize_prompt(
        self,
        module: dspy.Module,
        trainset: list[dspy.Example],
        metric: Callable,
        agent_name: str,
        save_path: Path | None = None,
    ) -> dspy.Module:
        """
        Optimize prompts for a DSPy module using MIPROv2.

        Args:
            module: DSPy module to optimize
            trainset: Training examples
            metric: Evaluation metric function
            agent_name: Name of the agent (for saving)
            save_path: Optional path to save optimized module

        Returns:
            Optimized DSPy module
        """
        print(f"\n=' Optimizing prompts for {agent_name}...")
        print(f"   Training examples: {len(trainset)}")

        # Choose optimizer based on config
        if self.config.dspy.optimizer == "MIPROv2":
            optimizer = MIPROv2(
                metric=metric,
                num_candidates=20,
                init_temperature=1.0,
            )
        elif self.config.dspy.optimizer == "BootstrapFewShot":
            optimizer = BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=4,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.dspy.optimizer}")

        # Compile (optimize) the module
        print(f"   Running {self.config.dspy.optimizer} optimizer...")
        optimized_module = optimizer.compile(
            module,
            trainset=trainset,
            num_trials=self.config.dspy.max_iterations,
        )

        # Save if path provided
        if save_path is None:
            save_path = self.config.paths.base_dir / "prompts" / "optimized" / f"{agent_name}.pkl"

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(optimized_module, f)

        print(f" Optimized prompts saved to {save_path}")

        return optimized_module

    def load_optimized_prompt(
        self,
        agent_name: str,
        load_path: Path | None = None,
    ) -> dspy.Module | None:
        """
        Load previously optimized prompts for an agent.

        Args:
            agent_name: Name of the agent
            load_path: Optional custom load path

        Returns:
            Loaded DSPy module or None if not found
        """
        if load_path is None:
            load_path = self.config.paths.base_dir / "prompts" / "optimized" / f"{agent_name}.pkl"

        if not load_path.exists():
            print(f"  No optimized prompts found for {agent_name}")
            return None

        try:
            with open(load_path, 'rb') as f:
                module = pickle.load(f)
            print(f" Loaded optimized prompts for {agent_name}")
            return module
        except Exception as e:
            print(f"L Error loading optimized prompts: {e}")
            return None

    def evaluate_prompt(
        self,
        module: dspy.Module,
        testset: list[dspy.Example],
        metric: Callable,
    ) -> float:
        """
        Evaluate a prompt's performance on a test set.

        Args:
            module: DSPy module to evaluate
            testset: Test examples
            metric: Evaluation metric

        Returns:
            Average metric score
        """
        scores = []
        for example in testset:
            try:
                prediction = module(**example.inputs())
                score = metric(example, prediction)
                scores.append(score)
            except Exception as e:
                print(f"  Evaluation error: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0


class TrainingDataCollector:
    """
    Collects training data for prompt optimization.

    This class manages:
    - Example collection from successful runs
    - Example storage and retrieval
    - Quality filtering
    """

    def __init__(self):
        """Initialize the training data collector."""
        self.config = get_config()
        self.storage_path = self.config.paths.base_dir / "training_data"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def add_example(
        self,
        agent_name: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        score: float,
    ) -> None:
        """
        Add a training example from an agent execution.

        Args:
            agent_name: Name of the agent
            inputs: Input data
            outputs: Output data
            score: Quality score (e.g., Kaggle score)
        """
        example = {
            "inputs": inputs,
            "outputs": outputs,
            "score": score,
        }

        # Load existing examples
        examples_file = self.storage_path / f"{agent_name}.json"
        examples = []

        if examples_file.exists():
            with open(examples_file) as f:
                examples = json.load(f)

        # Add new example
        examples.append(example)

        # Save
        with open(examples_file, 'w') as f:
            json.dump(examples, f, indent=2, default=str)

        print(f" Added training example for {agent_name} (score: {score:.4f})")

    def get_examples(
        self,
        agent_name: str,
        min_score: float | None = None,
        max_examples: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve training examples for an agent.

        Args:
            agent_name: Name of the agent
            min_score: Minimum score threshold
            max_examples: Maximum number of examples to return

        Returns:
            List of training examples
        """
        examples_file = self.storage_path / f"{agent_name}.json"

        if not examples_file.exists():
            return []

        with open(examples_file) as f:
            examples = json.load(f)

        # Filter by score
        if min_score is not None:
            examples = [ex for ex in examples if ex.get("score", 0) >= min_score]

        # Sort by score (descending)
        examples.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Limit count
        if max_examples is not None:
            examples = examples[:max_examples]

        return examples

    def convert_to_dspy_examples(
        self,
        agent_name: str,
        min_score: float | None = None,
    ) -> list[dspy.Example]:
        """
        Convert stored examples to DSPy Example format.

        Args:
            agent_name: Name of the agent
            min_score: Minimum score threshold

        Returns:
            List of DSPy examples
        """
        examples = self.get_examples(agent_name, min_score)

        required_inputs: dict[str, set[str]] = {
            "planner": {"competition_info", "domain", "sota_summary", "domain_guidance"},
            "developer_generator": {
                "component_details",
                "competition_context",
                "data_paths",
                "requirements",
            },
            "developer_fixer": {"code", "error", "error_type"},
        }
        required = required_inputs.get(agent_name, set())

        dspy_examples = []
        for ex in examples:
            inputs = ex.get("inputs", {})
            outputs = ex.get("outputs", {})

            if required and not required.issubset(set(inputs.keys())):
                # Skip legacy/malformed examples to keep optimization stable.
                continue

            dspy_ex = dspy.Example(
                **inputs,
                **outputs,
            ).with_inputs(*inputs.keys())

            dspy_examples.append(dspy_ex)

        return dspy_examples


# ==================== Convenience Functions ====================

def create_optimizer() -> PromptOptimizer:
    """
    Create a prompt optimizer instance.

    Returns:
        PromptOptimizer instance
    """
    return PromptOptimizer()


def create_training_collector() -> TrainingDataCollector:
    """
    Create a training data collector instance.

    Returns:
        TrainingDataCollector instance
    """
    return TrainingDataCollector()
