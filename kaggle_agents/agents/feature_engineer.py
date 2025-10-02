"""Feature engineering agent with advanced techniques."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..utils.config import Config
from ..utils.state import KaggleState
from ..utils.feature_engineering import AdvancedFeatureEngineer


class FeatureEngineeringAgent:
    """Agent responsible for feature engineering."""

    def __init__(self):
        """Initialize feature engineering agent."""
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL, temperature=Config.TEMPERATURE
        )
        self.engineer = AdvancedFeatureEngineer()

    def __call__(self, state: KaggleState) -> KaggleState:
        """Execute advanced feature engineering based on strategy.

        Args:
            state: Current workflow state

        Returns:
            Updated state with engineered features
        """
        print("Feature Engineering Agent: Creating features...")

        try:
            # Handle both dict and dataclass state access
            train_data_path = state.get("train_data_path", "") if isinstance(state, dict) else state.train_data_path
            test_data_path = state.get("test_data_path", "") if isinstance(state, dict) else state.test_data_path
            eda_summary = state.get("eda_summary", {}) if isinstance(state, dict) else state.eda_summary
            iteration = state.get("iteration", 0) if isinstance(state, dict) else state.iteration

            # Load data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Get strategy recommendations
            strategy = eda_summary.get("strategy", {})
            fe_priorities = strategy.get("feature_engineering_priorities", [])
            encoding_strategy = strategy.get("encoding_strategy", {"low_cardinality": "label", "high_cardinality": "target"})
            scaling_required = strategy.get("scaling_required", False)
            date_columns = strategy.get("data_characteristics", {}).get("temporal", {}).get("date_columns", [])

            # Identify target column
            potential_targets = ["target", "label", train_df.columns[-1]]
            target_col = None
            for col in potential_targets:
                if col in train_df.columns and col not in test_df.columns:
                    target_col = col
                    break

            # Separate target
            y_train = None
            if target_col:
                y_train = train_df[target_col]
                train_df = train_df.drop(columns=[target_col])

            print(f"  Applying {len(fe_priorities)} feature engineering techniques...")

            # 1. Handle missing values (advanced method)
            train_df, test_df = self.engineer.handle_missing_values_advanced(
                train_df, test_df, strategy="advanced"
            )

            # 2. Extract date features if applicable
            if date_columns:
                print(f"  Extracting features from {len(date_columns)} date columns")
                train_df, test_df = self.engineer.extract_date_features(
                    train_df, test_df, date_columns
                )

            # 3. Encode categorical variables adaptively
            if target_col:
                # Temporarily add target back for target encoding
                train_df[target_col] = y_train

            train_df, test_df = self.engineer.encode_categorical_adaptive(
                train_df, test_df, target_col if target_col else "", encoding_strategy
            )

            if target_col:
                y_train = train_df[target_col]
                train_df = train_df.drop(columns=[target_col])

            # 4. Create polynomial features if recommended
            if any("polynomial" in priority.lower() for priority in fe_priorities):
                print("  Creating polynomial features")
                train_df, test_df = self.engineer.create_polynomial_features(
                    train_df, test_df, degree=2
                )

            # 5. Create aggregation features
            train_df, test_df = self.engineer.create_aggregation_features(train_df, test_df)

            # 6. Scale features if required
            if scaling_required:
                print("  Scaling features")
                train_df, test_df = self.engineer.scale_features(
                    train_df, test_df, method="standard"
                )

            # Re-add target to train
            if target_col and y_train is not None:
                train_df[target_col] = y_train

            # Save processed data
            iteration_suffix = f"_iter{iteration}" if iteration > 0 else ""
            train_processed_path = train_data_path.replace(".csv", f"_processed{iteration_suffix}.csv")
            test_processed_path = test_data_path.replace(".csv", f"_processed{iteration_suffix}.csv")

            train_df.to_csv(train_processed_path, index=False)
            test_df.to_csv(test_processed_path, index=False)

            # Update state
            new_features_count = len(self.engineer.created_features)

            state.messages.append(
                HumanMessage(
                    content=f"Feature engineering completed. Created {new_features_count} new features using advanced techniques."
                )
            )

            print(f"Feature Engineering Agent: Created {new_features_count} features")

            return {
                "train_data_path": train_processed_path,
                "test_data_path": test_processed_path,
                "features_engineered": self.engineer.created_features,
            }

        except Exception as e:
            error_msg = f"Feature engineering failed: {str(e)}"
            print(f"Feature Engineering Agent ERROR: {error_msg}")
            # Return state with error appended, don't lose existing state
            errors = state.get("errors", []) if isinstance(state, dict) else state.errors
            return {"errors": errors + [error_msg]}
