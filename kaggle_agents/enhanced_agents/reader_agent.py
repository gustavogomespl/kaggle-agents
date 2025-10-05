"""Reader agent for extracting competition background information."""

import json
import logging
from typing import Dict, Any
from pathlib import Path

from ..core.agent_base import Agent
from ..core.state import EnhancedKaggleState, get_restore_dir, set_background_info
from ..prompts.prompt_reader import PROMPT_READER, PROMPT_READER_TASK, PROMPT_EXTRACT_METRIC

logger = logging.getLogger(__name__)


class ReaderAgent(Agent):
    """Agent responsible for reading and understanding competition background."""

    def __init__(self, model: str = "gpt-5-mini"):
        """Initialize Reader agent.

        Args:
            model: LLM model to use
        """
        super().__init__(
            role="reader",
            description="You are excellent at reading and extracting key information from competition descriptions.",
            model=model
        )

    def _execute(self, state: EnhancedKaggleState, role_prompt: str) -> Dict[str, Any]:
        """Execute reader agent logic.

        Args:
            state: Current state
            role_prompt: Role-specific prompt

        Returns:
            Dictionary with reader results
        """
        logger.info(f"Reader Agent executing for phase: {state.get('phase', '')}")

        history = []

        # Initialize system message
        if self.model == 'gpt-5-mini':
            history.append({"role": "system", "content": f"{role_prompt}{self.description}"})
        elif self.model == 'o1-mini':
            history.append({"role": "user", "content": f"{role_prompt}{self.description}"})

        # Read competition information
        competition_dir = Path(state.get("competition_dir", "."))
        overview_file = competition_dir / "competition_info.txt"
        data_desc_file = competition_dir / "data_description.txt"

        # Load files if they exist
        overview = ""
        data_description = ""

        if overview_file.exists():
            with open(overview_file, 'r', encoding='utf-8') as f:
                overview = f.read()
        else:
            logger.warning(f"Competition info file not found: {overview_file}")
            overview = "No competition overview available."

        if data_desc_file.exists():
            with open(data_desc_file, 'r', encoding='utf-8') as f:
                data_description = f.read()
        else:
            logger.warning(f"Data description file not found: {data_desc_file}")
            data_description = "No data description available."

        # Task for reader
        task = PROMPT_READER_TASK

        # Create main prompt
        input_prompt = PROMPT_READER.format(
            overview=overview[:5000],  # Limit to avoid token issues
            data_description=data_description[:5000]
        )

        # Generate response
        raw_reply, history = self.generate(
            input_prompt,
            history,
            max_completion_tokens=4096
        )

        # Parse markdown response
        background_summary = self._parse_markdown(raw_reply)

        # Save background to file
        restore_dir = get_restore_dir(state)
        background_file = restore_dir / "background_summary.md"
        with open(background_file, 'w', encoding='utf-8') as f:
            f.write(background_summary)

        # Extract competition type and metric
        competition_type = self._extract_competition_type(background_summary)
        metric = self._extract_metric(overview, history)

        # Update state
        state["competition_type"] = competition_type
        state["metric"] = metric
        set_background_info(state, background_summary)

        # Save history
        history_file = restore_dir / f"{self.role}_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"Reader Agent completed. Type: {competition_type}, Metric: {metric}")

        return {
            self.role: {
                "history": history,
                "role": self.role,
                "description": self.description,
                "task": task,
                "input": f"Overview: {overview[:500]}...",
                "result": background_summary,
                "competition_type": competition_type,
                "metric": metric
            }
        }

    def _extract_competition_type(self, background: str) -> str:
        """Extract competition type from background text.

        Args:
            background: Background summary text

        Returns:
            Competition type (classification, regression, etc.)
        """
        background_lower = background.lower()

        if 'classification' in background_lower:
            if 'binary' in background_lower:
                return "Binary Classification"
            elif 'multi-class' in background_lower or 'multiclass' in background_lower:
                return "Multi-class Classification"
            else:
                return "Classification"
        elif 'regression' in background_lower:
            return "Regression"
        elif 'ranking' in background_lower:
            return "Ranking"
        elif 'recommendation' in background_lower:
            return "Recommendation"
        elif 'time series' in background_lower or 'timeseries' in background_lower:
            return "Time Series"
        elif 'object detection' in background_lower:
            return "Object Detection"
        elif 'segmentation' in background_lower:
            return "Segmentation"
        else:
            return "Unknown"

    def _extract_metric(self, overview: str, history: list) -> str:
        """Extract evaluation metric from competition information.

        Args:
            overview: Competition overview text
            history: Conversation history

        Returns:
            Evaluation metric description
        """
        # Common metrics to check for
        overview_lower = overview.lower()

        # Direct pattern matching
        if 'accuracy' in overview_lower:
            return "Accuracy"
        elif 'auc' in overview_lower or 'roc' in overview_lower:
            return "AUC-ROC"
        elif 'rmse' in overview_lower or 'root mean squared error' in overview_lower:
            return "RMSE (Root Mean Squared Error)"
        elif 'mae' in overview_lower or 'mean absolute error' in overview_lower:
            return "MAE (Mean Absolute Error)"
        elif 'f1' in overview_lower or 'f1-score' in overview_lower:
            return "F1-Score"
        elif 'log loss' in overview_lower or 'logloss' in overview_lower:
            return "Log Loss"
        elif 'mse' in overview_lower or 'mean squared error' in overview_lower:
            return "MSE (Mean Squared Error)"
        elif 'r2' in overview_lower or 'r-squared' in overview_lower:
            return "R² (R-Squared)"
        elif 'map' in overview_lower and 'mean average precision' in overview_lower:
            return "MAP (Mean Average Precision)"

        # If no direct match, ask LLM
        try:
            prompt = PROMPT_EXTRACT_METRIC.format(competition_info=overview[:2000])
            raw_reply, _ = self.generate(prompt, history=[], max_completion_tokens=1024)
            metric = self._parse_markdown(raw_reply)
            return metric.strip()
        except Exception as e:
            logger.error(f"Error extracting metric: {e}")
            return "Unknown metric"


if __name__ == '__main__':
    # Test Reader Agent
    from ..core.state import EnhancedKaggleState

    # Create test state
    state = EnhancedKaggleState(
        competition_name="test-competition",
        competition_dir="./test_data/titanic",
        phase="Understand Background"
    )

    # Create and run reader
    reader = ReaderAgent()
    result = reader.action(state)

    print("Reader Result:")
    print(f"Competition Type: {result['reader']['competition_type']}")
    print(f"Metric: {result['reader']['metric']}")
    print(f"\nBackground:\n{result['reader']['result'][:500]}...")
