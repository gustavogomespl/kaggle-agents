"""Summarizer agent for generating phase reports."""

import json
import logging
from typing import Dict, Any
from pathlib import Path

from ..core.agent_base import Agent
from ..core.state import EnhancedKaggleState, get_restore_dir

logger = logging.getLogger(__name__)


class SummarizerAgent(Agent):
    """Agent responsible for generating comprehensive phase reports."""

    def __init__(self, model: str = "gpt-5-mini"):
        """Initialize Summarizer agent.

        Args:
            model: LLM model to use
        """
        super().__init__(
            role="summarizer",
            description="You are skilled at synthesizing information and creating clear, comprehensive reports.",
            model=model
        )

    def _create_summary_prompt(self, state: EnhancedKaggleState, phase_results: Dict[str, Any]) -> str:
        """Create prompt for summarization.

        Args:
            state: Current state
            phase_results: Results from all agents in the phase

        Returns:
            Summary prompt
        """
        phase = state.get("phase", "")
        competition_name = state.get("competition_name", "")
        competition_type = state.get("competition_type", "")
        metric = state.get("metric", "")
        iteration = state.get("iteration", 0)
        retry_count = state.get("retry_count", 0)

        prompt = f"""# PHASE SUMMARY TASK #

You need to create a comprehensive summary report for the "{phase}" phase of the Kaggle competition workflow.

## Competition Context ##
- Name: {competition_name}
- Type: {competition_type}
- Metric: {metric}

## Phase Information ##
Phase: {phase}
Iteration: {iteration}
Retry Count: {retry_count}

"""

        # Add results from each agent
        if "reader" in phase_results:
            reader_result = phase_results["reader"].get("result", "")
            prompt += f"""
## Reader Output ##
{reader_result[:1000]}
"""

        if "planner" in phase_results:
            planner_result = phase_results["planner"].get("result", "")
            prompt += f"""
## Planner Output ##
{planner_result[:1500]}
"""

        if "developer" in phase_results:
            developer_result = phase_results["developer"].get("result", "")
            developer_status = phase_results["developer"].get("status", "unknown")
            prompt += f"""
## Developer Output ##
Status: {developer_status}
{developer_result[:1000]}
"""

        if "reviewer" in phase_results:
            reviewer_result = phase_results["reviewer"].get("result", "")
            average_score = phase_results["reviewer"].get("average_score", 0)
            prompt += f"""
## Reviewer Output ##
Average Score: {average_score}/5.0
{reviewer_result}
"""

        prompt += """

# TASK #
Create a comprehensive summary report in markdown format that includes:

1. **Phase Overview**: Brief description of what was accomplished
2. **Key Activities**: Main actions taken by each agent
3. **Results**: Concrete outputs and artifacts created
4. **Quality Assessment**: Review scores and feedback
5. **Next Steps**: Recommendations for the next phase
6. **Issues & Challenges**: Any problems encountered and how they were addressed

# RESPONSE FORMAT #
```markdown
# {Phase Name} - Summary Report

## Phase Overview
[Brief overview of the phase execution]

## Key Activities

### Reader/Planner
[What the planner did]

### Developer
[What the developer did]

### Reviewer
[Review results]

## Results
[Concrete outputs and artifacts]

## Quality Assessment
[Scores and feedback]

## Next Steps
[Recommendations]

## Issues & Challenges
[Problems and solutions]
```
"""

        return prompt

    def _execute(self, state: EnhancedKaggleState, role_prompt: str) -> Dict[str, Any]:
        """Execute summarizer agent to create phase report.

        Args:
            state: Current state
            role_prompt: Role-specific prompt

        Returns:
            Dictionary with summarizer results
        """
        phase = state.get("phase", "")
        logger.info(f"Summarizer Agent executing for phase: {phase}")

        history = []

        # Initialize system message
        if self.model == 'gpt-5-mini':
            history.append({"role": "system", "content": f"{role_prompt}{self.description}"})
        elif self.model == 'o1-mini':
            history.append({"role": "user", "content": f"{role_prompt}{self.description}"})

        # Get phase results from last memory entry
        memory = state.get("memory", [])
        if not memory or len(memory) == 0:
            logger.warning("No memory entries to summarize")
            return {
                self.role: {
                    "result": "No phase results to summarize",
                    "summary": ""
                }
            }

        phase_results = memory[-1]

        # Create summary prompt
        summary_prompt = self._create_summary_prompt(state, phase_results)

        # Generate summary
        raw_reply, history = self.generate(summary_prompt, history)

        # Parse markdown summary
        summary = self._parse_markdown(raw_reply)

        # Save summary
        restore_dir = get_restore_dir(state)
        summary_file = restore_dir / "report.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)

        # Also save as markdown
        summary_md_file = restore_dir / "report.md"
        with open(summary_md_file, 'w') as f:
            f.write(summary)

        # Save history
        history_file = restore_dir / f"{self.role}_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"Summarizer completed. Report saved to: {summary_file}")

        return {
            self.role: {
                "history": history,
                "role": self.role,
                "description": self.description,
                "result": summary,
                "summary": summary,
                "report_file": str(summary_file)
            }
        }


if __name__ == '__main__':
    # Test Summarizer Agent
    from ..core.state import EnhancedKaggleState

    # Create test state with mock memory
    state = EnhancedKaggleState(
        competition_name="titanic",
        competition_dir="./test_data/titanic",
        phase="Data Cleaning",
        competition_type="Binary Classification",
        metric="Accuracy"
    )

    # Add mock memory
    state.memory.append({
        "planner": {
            "role": "planner",
            "result": "# Data Cleaning Plan\n1. Handle missing values\n2. Remove outliers\n3. Encode categorical variables"
        },
        "developer": {
            "role": "developer",
            "result": "Code implementation completed successfully",
            "status": "success",
            "success": True
        },
        "reviewer": {
            "role": "reviewer",
            "result": "Review completed. Average score: 4.5/5.0",
            "average_score": 4.5,
            "should_proceed": True
        }
    })

    # Create and run summarizer
    summarizer = SummarizerAgent()
    result = summarizer.action(state)

    print("Summarizer Result:")
    print(f"Report saved to: {result['summarizer']['report_file']}")
    print(f"\nSummary preview:\n{result['summarizer']['summary'][:500]}...")
