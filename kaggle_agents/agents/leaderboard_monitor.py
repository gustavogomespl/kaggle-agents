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
        """Monitor leaderboard and provide intelligent iteration decisions.

        Args:
            state: Current workflow state

        Returns:
            Updated state with leaderboard information and iteration strategy
        """
        print("Leaderboard Monitor: Checking results...")

        try:
            # Get user's submissions
            submissions = self.kaggle_client.get_my_submissions(
                state.competition_name
            )

            if not submissions:
                print("  No submissions found yet")
                return state

            # Get latest submission
            latest = submissions[0]
            public_score = latest.get("publicScore", 0.0)

            # Get full leaderboard to calculate precise percentile
            try:
                # Try to get all leaderboard entries for accurate count
                leaderboard = self.kaggle_client.get_leaderboard(
                    state.competition_name, top_n=10000
                )
                total_teams = len(leaderboard)
            except Exception:
                # Fallback to top 100
                leaderboard = self.kaggle_client.get_leaderboard(
                    state.competition_name, top_n=100
                )
                # Estimate total teams (usually much more than visible)
                total_teams = 1000

            # Find user's rank
            rank = 1
            for entry in leaderboard:
                if entry["score"] > public_score:
                    rank += 1
                else:
                    break

            # Calculate precise percentile
            percentile = (rank / total_teams) * 100 if total_teams > 0 else 0

            # Get CV score from best model for overfitting analysis
            cv_score = state.best_model.get("mean_cv_score", 0.0)

            # Detect overfitting
            # For metrics where higher is better (accuracy, AUC)
            # CV > Public suggests overfitting
            # For metrics where lower is better (RMSE, MAE)
            # Public > CV suggests overfitting
            metric = state.metric.lower()
            is_higher_better = any(m in metric for m in ["accuracy", "auc", "f1", "recall", "precision"])

            overfitting_detected = False
            overfitting_severity = "none"

            if cv_score > 0 and public_score > 0:
                if is_higher_better:
                    score_diff = cv_score - public_score
                    if score_diff > 0.05:  # 5% drop
                        overfitting_detected = True
                        overfitting_severity = "severe" if score_diff > 0.1 else "moderate"
                else:
                    # For negative metrics like MSE, both should be negative
                    score_diff = abs(public_score) - abs(cv_score)
                    if score_diff > 0.05:
                        overfitting_detected = True
                        overfitting_severity = "severe" if score_diff > 0.1 else "moderate"

            # Store analysis results
            iteration_strategy = {
                "overfitting_detected": overfitting_detected,
                "overfitting_severity": overfitting_severity,
                "cv_score": cv_score,
                "public_score": public_score,
                "rank": rank,
                "total_teams": total_teams,
                "percentile": percentile,
            }

            # Use LLM to analyze performance
            system_msg = SystemMessage(
                content="""You are a Kaggle competition expert analyzing results.
                Provide insights on the performance and suggestions for improvement."""
            )

            human_msg = HumanMessage(
                content=f"""Competition Results Analysis:

Competition: {state.competition_name}
Evaluation Metric: {state.metric}

Our Performance:
- Public Score: {public_score:.4f}
- CV Score: {cv_score:.4f}
- Score Difference: {abs(public_score - cv_score):.4f}
- Overfitting Detected: {overfitting_detected} ({overfitting_severity})
- Rank: {rank} / {total_teams}
- Percentile: Top {percentile:.1f}%

Model Used: {state.best_model['name']}
Features Engineered: {len(state.features_engineered)}
Models Tried: {len(state.models_trained)}
Iteration: {state.iteration + 1}/{state.max_iterations}

Analyze performance and recommend next steps."""
            )

            response = self.llm.invoke([system_msg, human_msg])

            state.messages.append(
                HumanMessage(
                    content=f"Leaderboard Analysis: Rank {rank}/{total_teams} (Top {percentile:.1f}%). Overfitting: {overfitting_severity}. {response.content}"
                )
            )

            # Store detailed metrics for next iteration
            print(f"Leaderboard Monitor: Rank {rank}/{total_teams} (Top {percentile:.1f}%)")
            print(f"  CV Score: {cv_score:.4f} | Public Score: {public_score:.4f}")

            if overfitting_detected:
                print(f"  WARNING: {overfitting_severity.upper()} overfitting detected")

            # Decision on whether to continue iterating
            if percentile <= 20:
                print(f"SUCCESS: Achieved Top {percentile:.1f}%")
            elif state.iteration + 1 >= state.max_iterations:
                print(f"Reached max iterations. Final rank: Top {percentile:.1f}%")
            else:
                if overfitting_detected:
                    print(f"Iterating to address overfitting and improve to top 20%")
                else:
                    print(f"Iterating to improve from Top {percentile:.1f}% to top 20%")

            return {
                "submission_score": public_score,
                "leaderboard_rank": rank,
                "iteration": state.iteration + 1,
                "eda_summary": {
                    **state.eda_summary,
                    "iteration_strategy": iteration_strategy,
                }
            }

        except Exception as e:
            error_msg = f"Leaderboard monitoring failed: {str(e)}"
            print(f"Leaderboard Monitor ERROR: {error_msg}")
            return {"errors": [error_msg]}
