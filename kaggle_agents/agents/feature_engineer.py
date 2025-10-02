"""Feature engineering agent."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..utils.config import Config
from ..utils.state import KaggleState


class FeatureEngineeringAgent:
    """Agent responsible for feature engineering."""

    def __init__(self):
        """Initialize feature engineering agent."""
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL, temperature=Config.TEMPERATURE
        )

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with missing values handled
        """
        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        # Categorical columns: fill with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown", inplace=True)

        return df

    def encode_categorical(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """Encode categorical variables.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe

        Returns:
            Tuple of (encoded_train, encoded_test)
        """
        categorical_cols = train_df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on combined data to handle test categories not in train
            combined = pd.concat([train_df[col], test_df[col]], axis=0)
            le.fit(combined.astype(str))

            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))

        return train_df, test_df

    def create_features(self, df: pd.DataFrame, insights: List[str]) -> pd.DataFrame:
        """Create new features based on domain knowledge and insights.

        Args:
            df: Input dataframe
            insights: Insights from EDA

        Returns:
            DataFrame with new features
        """
        new_features = []

        # Create interaction features for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            # Create sum/product of first two numeric features
            df[f"{numeric_cols[0]}_x_{numeric_cols[1]}"] = df[numeric_cols[0]] * df[numeric_cols[1]]
            new_features.append(f"{numeric_cols[0]}_x_{numeric_cols[1]}")

        # Create aggregated features if there are multiple numeric columns
        if len(numeric_cols) >= 3:
            df["numeric_sum"] = df[numeric_cols].sum(axis=1)
            df["numeric_mean"] = df[numeric_cols].mean(axis=1)
            df["numeric_std"] = df[numeric_cols].std(axis=1)
            new_features.extend(["numeric_sum", "numeric_mean", "numeric_std"])

        return df, new_features

    def __call__(self, state: KaggleState) -> KaggleState:
        """Execute feature engineering.

        Args:
            state: Current workflow state

        Returns:
            Updated state with engineered features
        """
        print("🔧 Feature Engineering Agent: Creating features...")

        try:
            # Load data
            train_df = pd.read_csv(state["train_data_path"])
            test_df = pd.read_csv(state["test_data_path"])

            # Identify target column (usually last column or 'target')
            potential_targets = ["target", "label", train_df.columns[-1]]
            target_col = None
            for col in potential_targets:
                if col in train_df.columns and col not in test_df.columns:
                    target_col = col
                    break

            # Separate target
            if target_col:
                y_train = train_df[target_col]
                train_df = train_df.drop(columns=[target_col])

            # Use LLM to suggest features
            system_msg = SystemMessage(
                content="""You are a feature engineering expert for Kaggle competitions.
                Suggest specific feature engineering strategies based on the data insights."""
            )

            human_msg = HumanMessage(
                content=f"""Data Insights:
{chr(10).join(f'- {insight}' for insight in state.get('data_insights', []))}

Competition Type: {state.get('competition_type', 'unknown')}
Columns: {train_df.columns.tolist()}

Suggest 3-5 specific feature engineering techniques to apply."""
            )

            response = self.llm.invoke([system_msg, human_msg])

            # Handle missing values
            train_df = self.handle_missing_values(train_df)
            test_df = self.handle_missing_values(test_df)

            # Encode categorical variables
            train_df, test_df = self.encode_categorical(train_df, test_df)

            # Create features
            train_df, new_features = self.create_features(
                train_df, state.get("data_insights", [])
            )
            test_df, _ = self.create_features(test_df, state.get("data_insights", []))

            # Re-add target to train
            if target_col:
                train_df[target_col] = y_train

            # Save processed data
            train_processed_path = state["train_data_path"].replace(".csv", "_processed.csv")
            test_processed_path = state["test_data_path"].replace(".csv", "_processed.csv")

            train_df.to_csv(train_processed_path, index=False)
            test_df.to_csv(test_processed_path, index=False)

            # Update state
            state["train_data_path"] = train_processed_path
            state["test_data_path"] = test_processed_path
            state["features_engineered"] = new_features

            state["messages"].append(
                HumanMessage(
                    content=f"Feature engineering completed. Created {len(new_features)} new features. Strategies: {response.content}"
                )
            )

            print(f"Feature Engineering Agent: Created {len(new_features)} features")

        except Exception as e:
            error_msg = f"Feature engineering failed: {str(e)}"
            print(f"Feature Engineering Agent ERROR: {error_msg}")
            return {"errors": [error_msg]}

        return state
