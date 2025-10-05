"""Reviewer agent for scoring and providing feedback on agent outputs."""

import json
import logging
from typing import Dict, Any, List, Tuple

from ..core.agent_base import Agent
from ..core.state import EnhancedKaggleState
from ..prompts.prompt_reviewer import (
    PROMPT_REVIEWER,
    PROMPT_REVIEWER_TASK,
    PROMPT_MULTI_AGENT_REVIEW
)

logger = logging.getLogger(__name__)


class ReviewerAgent(Agent):
    """Agent responsible for reviewing and scoring other agents' work."""

    def __init__(self, model: str = "gpt-4o"):
        """Initialize Reviewer agent.

        Args:
            model: LLM model to use
        """
        super().__init__(
            role="reviewer",
            description="You are an experienced code reviewer and quality assurance expert.",
            model=model
        )

    def _review_single_agent(
        self,
        agent_role: str,
        agent_output: Dict[str, Any],
        state: EnhancedKaggleState,
        history: list
    ) -> Tuple[Dict[str, Any], list]:
        """Review a single agent's output.

        Args:
            agent_role: Role of agent to review
            agent_output: Output from the agent
            state: Current state
            history: Conversation history

        Returns:
            Tuple of (review_result, updated_history)
        """
        task = PROMPT_REVIEWER_TASK.format(agent_role=agent_role)

        # Get agent task, input, and output
        agent_task = agent_output.get('task', 'No task specified')
        agent_input = agent_output.get('input', 'No input specified')
        agent_result = agent_output.get('result', 'No output specified')

        # Get background
        background = state.background_info[:1000] if state.background_info else "No background available"

        # Create review prompt
        input_prompt = PROMPT_REVIEWER.format(
            agent_role=agent_role,
            phase_name=state.phase,
            task=agent_task,
            input=agent_input,
            output=agent_result,
            background=background
        )

        # Generate review
        raw_reply, history = self.generate(input_prompt, history, max_completion_tokens=4096)

        # Parse JSON review
        try:
            review = self._parse_json(raw_reply)
        except Exception as e:
            logger.error(f"Error parsing review JSON: {e}")
            # Fallback to basic structure
            review = {
                "agent": agent_role,
                "score": 3,
                "analysis": {
                    "strengths": ["Unable to parse detailed review"],
                    "weaknesses": ["Unable to parse detailed review"],
                    "specific_issues": []
                },
                "suggestion": "Please review manually",
                "requires_revision": False
            }

        return review, history

    def _execute(self, state: EnhancedKaggleState, role_prompt: str) -> Dict[str, Any]:
        """Execute reviewer agent to score multiple agents.

        Args:
            state: Current state
            role_prompt: Role-specific prompt

        Returns:
            Dictionary with reviewer results
        """
        logger.info(f"Reviewer Agent executing for phase: {state.phase}")

        history = []

        # Initialize system message
        if self.model == 'gpt-4o':
            history.append({"role": "system", "content": f"{role_prompt}{self.description}"})
        elif self.model == 'o1-mini':
            history.append({"role": "user", "content": f"{role_prompt}{self.description}"})

        # Get agents to review from last memory entry
        if not state.memory or len(state.memory) == 0:
            logger.warning("No memory entries to review")
            return {
                self.role: {
                    "reviews": {},
                    "overall_score": 0,
                    "should_proceed": False
                }
            }

        last_memory = state.memory[-1]

        # Determine which agents to review based on phase
        agents_to_review = []
        if state.phase == "Understand Background":
            agents_to_review = ["reader"]
        else:
            agents_to_review = ["planner", "developer"]

        # Review each agent
        reviews = {}
        scores = []

        for agent_role in agents_to_review:
            if agent_role in last_memory:
                logger.info(f"Reviewing {agent_role}")

                agent_output = last_memory[agent_role]
                review, history = self._review_single_agent(
                    agent_role,
                    agent_output,
                    state,
                    history
                )

                reviews[f"agent {agent_role}"] = review
                scores.append(review.get('score', 3))

            else:
                logger.warning(f"Agent {agent_role} not found in memory")

        # Calculate overall assessment
        if scores:
            average_score = sum(scores) / len(scores)
            should_proceed = average_score >= 3.0
        else:
            average_score = 0
            should_proceed = False

        # Save history
        history_file = state.restore_dir / f"{self.role}_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        # Save reviews
        reviews_file = state.restore_dir / "reviews.json"
        with open(reviews_file, 'w') as f:
            json.dump(reviews, f, indent=2)

        logger.info(f"Reviewer completed. Average score: {average_score:.2f}, Proceed: {should_proceed}")

        # Format scores and suggestions for memory
        score_dict = {}
        suggestion_dict = {}

        for agent_key, review in reviews.items():
            score_dict[agent_key] = review.get('score', 3)
            suggestion_dict[agent_key] = review.get('suggestion', 'No specific suggestions')

        return {
            self.role: {
                "history": history,
                "role": self.role,
                "description": self.description,
                "reviews": reviews,
                "score": score_dict,
                "suggestion": suggestion_dict,
                "average_score": average_score,
                "should_proceed": should_proceed,
                "result": f"Review completed. Average score: {average_score:.2f}/5.0"
            }
        }


if __name__ == '__main__':
    # Test Reviewer Agent
    from ..core.state import EnhancedKaggleState

    # Create test state with mock memory
    state = EnhancedKaggleState(
        competition_name="titanic",
        competition_dir="./test_data/titanic",
        phase="Data Cleaning"
    )

    # Add mock memory
    state.memory.append({
        "planner": {
            "role": "planner",
            "task": "Create data cleaning plan",
            "input": "Background info...",
            "result": "# Data Cleaning Plan\n1. Handle missing values\n2. Remove outliers",
            "plan": "..."
        },
        "developer": {
            "role": "developer",
            "task": "Implement data cleaning",
            "input": "Plan...",
            "result": "Code implementation completed successfully",
            "code": "import pandas as pd\n...",
            "success": True
        }
    })

    # Create and run reviewer
    reviewer = ReviewerAgent()
    result = reviewer.action(state)

    print("Reviewer Result:")
    print(f"Average Score: {result['reviewer']['average_score']}")
    print(f"Should Proceed: {result['reviewer']['should_proceed']}")
    print(f"\nScores: {result['reviewer']['score']}")
