"""Leaderboard monitoring agent."""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..tools.kaggle_api import KaggleAPIClient
from ..utils.config import Config
from ..utils.state import KaggleState


class LeaderboardMonitorAgent:
    """Agent responsible for monitoring leaderboard and analyzing results."""

    def __init__(self):
        """Initialize leaderboard monitor agent."""
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL, temperature=Config.TEMPERATURE
        )
        self.kaggle_client = KaggleAPIClient()

    def __call__(self, state: KaggleState) -> KaggleState:
        """Monitor leaderboard and provide insights.

        Args:
            state: Current workflow state

        Returns:
            Updated state with leaderboard information
        """
        print("📊 Leaderboard Monitor: Checking results...")

        try:
            # Get user's submissions
            submissions = self.kaggle_client.get_my_submissions(
                state["competition_name"]
            )

            if not submissions:
                print("⚠️  No submissions found yet")
                state["next_agent"] = "end"
                return state

            # Get latest submission
            latest = submissions[0]
            state["submission_score"] = latest.get("publicScore", 0.0)

            # Get leaderboard
            leaderboard = self.kaggle_client.get_leaderboard(
                state["competition_name"], top_n=100
            )

            # Find user's rank (approximate)
            user_score = state["submission_score"]
            rank = 1
            for entry in leaderboard:
                if entry["score"] > user_score:
                    rank += 1
                else:
                    break

            state["leaderboard_rank"] = rank

            # Calculate percentile
            total_teams = len(leaderboard)
            percentile = (rank / total_teams) * 100 if total_teams > 0 else 0

            # Use LLM to analyze performance
            system_msg = SystemMessage(
                content="""You are a Kaggle competition expert analyzing results.
                Provide insights on the performance and suggestions for improvement."""
            )

            human_msg = HumanMessage(
                content=f"""Competition Results Analysis:

Competition: {state['competition_name']}
Evaluation Metric: {state.get('metric', 'unknown')}

Our Performance:
- Public Score: {state['submission_score']}
- Estimated Rank: {rank} / {total_teams}
- Percentile: Top {percentile:.1f}%
- Model Used: {state['best_model']['name']}
- CV Score: {state['best_model']['mean_cv_score']:.4f}

Top 3 Leaderboard:
{chr(10).join(f"{i+1}. {entry['teamName']}: {entry['score']}" for i, entry in enumerate(leaderboard[:3]))}

Features Engineered: {len(state.get('features_engineered', []))}
Models Tried: {len(state.get('models_trained', []))}

Analyze our performance and suggest specific improvements to reach top 20%."""
            )

            response = self.llm.invoke([system_msg, human_msg])

            state["messages"].append(
                HumanMessage(
                    content=f"Leaderboard Analysis: Rank {rank}/{total_teams} (Top {percentile:.1f}%). {response.content}"
                )
            )

            # Update iteration counter
            state["iteration"] = state.get("iteration", 0) + 1

            # Check if we should iterate
            if percentile > 20 and state["iteration"] < state.get("max_iterations", 5):
                print(f"Current rank: Top {percentile:.1f}% - Iterating to improve...")
            else:
                if percentile <= 20:
                    print(f"SUCCESS: Achieved Top {percentile:.1f}%")
                else:
                    print(f"Reached max iterations. Final rank: Top {percentile:.1f}%")

            print(f"Leaderboard Monitor: Rank {rank}/{total_teams} (Top {percentile:.1f}%)")

        except Exception as e:
            error_msg = f"Leaderboard monitoring failed: {str(e)}"
            print(f"Leaderboard Monitor ERROR: {error_msg}")
            return {"errors": [error_msg]}

        return state
