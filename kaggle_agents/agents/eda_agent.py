"""Exploratory Data Analysis (EDA) agent."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..utils.config import Config
from ..utils.state import KaggleState


class EDAAgent:
    """Agent responsible for exploratory data analysis."""

    def __init__(self):
        """Initialize EDA agent."""
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL, temperature=Config.TEMPERATURE
        )

    def analyze_dataframe(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """Analyze a dataframe and extract key statistics.

        Args:
            df: DataFrame to analyze
            name: Name of the dataset (train/test)

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "name": name,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
        }

        # Numeric columns stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis["numeric_stats"] = df[numeric_cols].describe().to_dict()

        # Categorical columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            analysis["categorical_stats"] = {
                col: {
                    "unique_values": df[col].nunique(),
                    "most_common": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                }
                for col in categorical_cols
            }

        return analysis

    def __call__(self, state: KaggleState) -> KaggleState:
        """Execute exploratory data analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with EDA results
        """
        print("📊 EDA Agent: Analyzing data...")

        try:
            # Load data
            train_df = pd.read_csv(state["train_data_path"])
            test_df = pd.read_csv(state["test_data_path"])

            # Analyze both datasets
            train_analysis = self.analyze_dataframe(train_df, "train")
            test_analysis = self.analyze_dataframe(test_df, "test")

            # Store analysis in state
            state["eda_summary"] = {
                "train": train_analysis,
                "test": test_analysis,
            }

            # Use LLM to generate insights
            system_msg = SystemMessage(
                content="""You are a data scientist performing EDA on a Kaggle competition dataset.
                Analyze the data characteristics and provide actionable insights for modeling."""
            )

            human_msg = HumanMessage(
                content=f"""Train Data Analysis:
- Shape: {train_analysis['shape']}
- Columns: {train_analysis['columns']}
- Missing values: {train_analysis['missing_values']}

Test Data Analysis:
- Shape: {test_analysis['shape']}
- Columns: {test_analysis['columns']}
- Missing values: {test_analysis['missing_values']}

Competition Type: {state.get('competition_type', 'unknown')}
Evaluation Metric: {state.get('metric', 'unknown')}

Provide 5-7 key insights about this data and what we should focus on."""
            )

            response = self.llm.invoke([system_msg, human_msg])

            # Extract insights from LLM response
            insights = [
                line.strip("- ").strip()
                for line in response.content.split("\n")
                if line.strip() and (line.strip().startswith("-") or (len(line.strip()) > 0 and line.strip()[0].isdigit()))
            ]

            state["data_insights"] = insights

            # Handle messages state access
            messages = state.get("messages", []) if isinstance(state, dict) else state.messages
            messages.append(
                HumanMessage(
                    content=f"EDA completed. Key insights: {response.content}"
                )
            )

            print(f"EDA Agent: Analysis complete. Found {len(insights)} key insights")

            state["messages"] = messages

        except Exception as e:
            error_msg = f"EDA failed: {str(e)}"
            print(f"EDA Agent ERROR: {error_msg}")
            # Return state with error appended, don't lose existing state
            errors = state.get("errors", []) if isinstance(state, dict) else state.errors
            return {"errors": errors + [error_msg]}

        return state
