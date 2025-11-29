"""Data collection agent for Kaggle competitions."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..tools.kaggle_api import KaggleAPIClient
from ..utils.config import Config
from ..utils.state import KaggleState


class DataCollectorAgent:
    """Agent responsible for collecting competition data."""

    def __init__(self):
        """Initialize data collector agent."""
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL, temperature=Config.TEMPERATURE
        )
        self.kaggle_client = KaggleAPIClient()

    def __call__(self, state: KaggleState) -> KaggleState:
        """Execute data collection.

        Args:
            state: Current workflow state

        Returns:
            Updated state with downloaded data paths
        """
        competition = state["competition_name"]

        print(f"Data Collector: Fetching data for '{competition}'...")

        try:
            # Get competition info
            comp_info = self.kaggle_client.get_competition_info(competition)
            state["competition_type"] = comp_info["category"]
            state["metric"] = comp_info["evaluation"]

            # Download data
            data_paths = self.kaggle_client.download_competition_data(
                competition, path=f"{Config.DATA_DIR}/{competition}"
            )

            state["train_data_path"] = data_paths.get("train", "")
            state["test_data_path"] = data_paths.get("test", "")
            state["sample_submission_path"] = data_paths.get("sample_submission", "")

            # Detect target column
            target_col = "target"  # Default
            try:
                import pandas as pd
                if state["sample_submission_path"]:
                    sub_df = pd.read_csv(state["sample_submission_path"], nrows=1)
                    # usually id, target. So second column is target
                    if len(sub_df.columns) > 1:
                        target_col = sub_df.columns[1]
                elif state["train_data_path"]:
                    train_df = pd.read_csv(state["train_data_path"], nrows=1)
                    # Heuristic: last column or column with 'target' in name
                    target_candidates = [c for c in train_df.columns if c.lower() in ['target', 'label', 'class', 'loan_paid_back', 'survived', 'price', 'sales']]
                    target_col = target_candidates[0] if target_candidates else train_df.columns[-1]

                print(f"Data Collector: Detected target column: '{target_col}'")
                state["target_col"] = target_col
            except Exception as e:
                print(f"Data Collector Warning: Could not detect target column: {e}")
                state["target_col"] = "target"

            # Use LLM to understand competition requirements
            system_msg = SystemMessage(
                content="""You are a data science expert analyzing Kaggle competitions.
                Provide a brief analysis of the competition requirements and what approach might work."""
            )

            human_msg = HumanMessage(
                content=f"""Competition: {comp_info['title']}
Description: {comp_info['description']}
Evaluation Metric: {comp_info['evaluation']}
Category: {comp_info['category']}

What are the key characteristics of this competition and what should we focus on?"""
            )

            response = self.llm.invoke([system_msg, human_msg])

            state["messages"].append(
                HumanMessage(
                    content=f"Data collected for {competition}. Competition Analysis: {response.content}"
                )
            )

            print(f"Data Collector: Downloaded data to {Config.DATA_DIR}/{competition}")

        except Exception as e:
            error_msg = f"Data collection failed: {e!s}"
            print(f"Data Collector ERROR: {error_msg}")
            # Return state with error appended, don't lose existing state
            errors = state.get("errors", []) if isinstance(state, dict) else state.errors
            return {"errors": [*errors, error_msg]}

        return state
