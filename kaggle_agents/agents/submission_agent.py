"""Submission agent for Kaggle competitions."""

import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from ..tools.kaggle_api import KaggleAPIClient
from ..utils.config import Config
from ..utils.state import KaggleState


class SubmissionAgent:
    """Agent responsible for generating predictions and submitting to Kaggle."""

    def __init__(self):
        """Initialize submission agent."""
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL, temperature=Config.TEMPERATURE
        )
        self.kaggle_client = KaggleAPIClient()

    def __call__(self, state: KaggleState) -> KaggleState:
        """Execute prediction and submission.

        Args:
            state: Current workflow state

        Returns:
            Updated state with submission results
        """
        print("📤 Submission Agent: Creating submission...")

        try:
            # Load test data
            test_df = pd.read_csv(state["test_data_path"])

            # Load sample submission to understand format
            sample_sub = pd.read_csv(state["sample_submission_path"])

            # Load best model
            model = joblib.load(state["best_model"]["path"])

            # Get ID column (usually first column in sample submission)
            id_col = sample_sub.columns[0]
            target_col = sample_sub.columns[1]

            # Ensure test data has same columns as training (minus target)
            # The ID column should be preserved
            test_ids = test_df[id_col] if id_col in test_df.columns else None

            # Drop ID column for prediction
            X_test = test_df.drop(columns=[id_col], errors="ignore")

            # Make predictions
            print("  Generating predictions...")
            predictions = model.predict(X_test)

            # Create submission dataframe
            submission = pd.DataFrame({
                id_col: test_ids if test_ids is not None else range(len(predictions)),
                target_col: predictions
            })

            # Save submission
            Path(Config.SUBMISSIONS_DIR).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            submission_filename = f"submission_{state['competition_name']}_{timestamp}.csv"
            submission_path = f"{Config.SUBMISSIONS_DIR}/{submission_filename}"
            submission.to_csv(submission_path, index=False)

            state["submission_path"] = submission_path

            # Submit to Kaggle if credentials are available
            if Config.KAGGLE_USERNAME and Config.KAGGLE_KEY:
                try:
                    print("  Submitting to Kaggle...")
                    submission_message = f"AutoKaggle: {state['best_model']['name']} (CV: {state['best_model']['mean_cv_score']:.4f})"

                    self.kaggle_client.submit_prediction(
                        state["competition_name"],
                        submission_path,
                        submission_message
                    )

                    state["messages"].append(
                        HumanMessage(
                            content=f"Submission uploaded to Kaggle: {submission_message}"
                        )
                    )

                    print(f"Submission Agent: Submitted to Kaggle")

                except Exception as e:
                    print(f"WARNING: Could not submit to Kaggle: {str(e)}")
                    print(f"Submission saved locally at: {submission_path}")
            else:
                print(f"Submission Agent: Saved submission to {submission_path}")
                print("Set KAGGLE_USERNAME and KAGGLE_KEY to submit automatically")

        except Exception as e:
            error_msg = f"Submission failed: {str(e)}"
            print(f"Submission Agent ERROR: {error_msg}")
            # Return state with error appended, don't lose existing state
            errors = state.get("errors", []) if isinstance(state, dict) else state.errors
            return {"errors": errors + [error_msg]}

        return state
