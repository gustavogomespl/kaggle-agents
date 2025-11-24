"""Strategy agent for high-level decision making."""

import pandas as pd
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from ..utils.config import Config
from ..utils.state import KaggleState


class CompetitionStrategy(BaseModel):
    """Strategy recommendations for the competition."""

    problem_characteristics: List[str] = Field(
        description="Key characteristics of the problem (e.g., 'high cardinality categoricals', 'time series', 'imbalanced classes')"
    )
    recommended_models: List[str] = Field(
        description="Recommended models to prioritize (e.g., 'LightGBM', 'CatBoost', 'XGBoost')"
    )
    feature_engineering_priorities: List[str] = Field(
        description="Specific feature engineering techniques to apply (e.g., 'target encoding', 'polynomial features', 'date extraction')"
    )
    validation_strategy: str = Field(
        description="Recommended cross-validation strategy (e.g., 'StratifiedKFold', 'TimeSeriesSplit', 'GroupKFold')"
    )
    encoding_low_cardinality: str = Field(
        default="label",
        description="Encoding strategy for low cardinality categoricals (e.g., 'label', 'onehot')",
    )
    encoding_high_cardinality: str = Field(
        default="target",
        description="Encoding strategy for high cardinality categoricals (e.g., 'target', 'catboost', 'frequency')",
    )
    scaling_required: bool = Field(
        description="Whether feature scaling is required based on selected models"
    )
    ensemble_strategy: str = Field(
        description="Recommended ensemble approach (e.g., 'stacking', 'blending', 'voting')"
    )


class StrategyAgent:
    """Agent responsible for high-level strategic decisions."""

    def __init__(self):
        """Initialize strategy agent."""
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=Config.TEMPERATURE)

    def analyze_data_characteristics(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset characteristics for strategy formulation.

        Args:
            train_df: Training dataframe

        Returns:
            Dictionary with data characteristics
        """
        characteristics = {}

        # Dataset size
        n_rows, n_cols = train_df.shape
        characteristics["size"] = {"rows": n_rows, "columns": n_cols}
        characteristics["size_category"] = (
            "small" if n_rows < 10000 else "medium" if n_rows < 100000 else "large"
        )

        # Numeric vs categorical ratio
        numeric_cols = train_df.select_dtypes(include=["number"]).columns
        categorical_cols = train_df.select_dtypes(
            include=["object", "category"]
        ).columns

        characteristics["column_types"] = {
            "numeric_count": len(numeric_cols),
            "categorical_count": len(categorical_cols),
            "ratio_numeric": len(numeric_cols) / n_cols if n_cols > 0 else 0,
        }

        # Categorical cardinality analysis
        if len(categorical_cols) > 0:
            cardinalities = {col: train_df[col].nunique() for col in categorical_cols}
            characteristics["categorical_cardinality"] = {
                "low": sum(1 for v in cardinalities.values() if v < 10),
                "medium": sum(1 for v in cardinalities.values() if 10 <= v < 50),
                "high": sum(1 for v in cardinalities.values() if v >= 50),
                "max": max(cardinalities.values()),
            }

        # Missing values analysis
        missing_ratio = train_df.isnull().sum() / len(train_df)
        characteristics["missing_values"] = {
            "has_missing": (missing_ratio > 0).any(),
            "columns_with_missing": (missing_ratio > 0).sum(),
            "max_missing_ratio": missing_ratio.max(),
        }

        # Check for potential time series
        date_cols = []
        for col in train_df.columns:
            if "date" in col.lower() or "time" in col.lower() or "year" in col.lower():
                date_cols.append(col)
        characteristics["temporal"] = {
            "has_date_columns": len(date_cols) > 0,
            "date_columns": date_cols,
        }

        # Target analysis (if identifiable)
        potential_target_cols = [
            col
            for col in train_df.columns
            if col.lower() in ["target", "label", train_df.columns[-1]]
        ]
        if potential_target_cols:
            target_col = potential_target_cols[0]
            if target_col in train_df.columns:
                characteristics["target"] = {
                    "name": target_col,
                    "type": "numeric"
                    if pd.api.types.is_numeric_dtype(train_df[target_col])
                    else "categorical",
                    "unique_values": train_df[target_col].nunique(),
                }

                # Check for class imbalance
                if (
                    characteristics["target"]["type"] == "categorical"
                    or characteristics["target"]["unique_values"] < 20
                ):
                    value_counts = train_df[target_col].value_counts()
                    characteristics["target"]["imbalance_ratio"] = (
                        value_counts.max() / value_counts.min()
                    )

        return characteristics

    def __call__(self, state: KaggleState) -> KaggleState:
        """Formulate competition strategy based on data analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with strategy recommendations
        """
        print("Strategy Agent: Analyzing competition and formulating approach...")

        try:
            # Handle both dict and dataclass state access
            train_data_path = (
                state.get("train_data_path", "")
                if isinstance(state, dict)
                else state.train_data_path
            )

            # Load training data
            train_df = pd.read_csv(train_data_path)

            # Analyze data characteristics
            data_chars = self.analyze_data_characteristics(train_df)

            # Use LLM with structured output to generate strategy
            llm_with_structure = self.llm.with_structured_output(CompetitionStrategy)

            system_msg = SystemMessage(
                content="""You are an expert Kaggle strategist. Based on the competition details and data characteristics,
                provide a comprehensive strategy for achieving top performance. Consider:
                - Dataset size and type
                - Feature distributions and cardinality
                - Appropriate models for the problem
                - Feature engineering techniques
                - Cross-validation strategy
                - Ensemble approaches

                Be specific and actionable in your recommendations."""
            )

            # Build detailed prompt with all information
            characteristics_str = "\n".join(
                [f"- {k}: {v}" for k, v in data_chars.items()]
            )
            data_insights = (
                state.get("data_insights", [])
                if isinstance(state, dict)
                else state.data_insights
            )
            insights_str = (
                "\n".join([f"- {insight}" for insight in data_insights])
                if data_insights
                else "No insights yet"
            )

            # Handle more state access for competition details
            competition_name = (
                state.get("competition_name", "")
                if isinstance(state, dict)
                else state.competition_name
            )
            competition_type = (
                state.get("competition_type", "")
                if isinstance(state, dict)
                else state.competition_type
            )
            metric = (
                state.get("metric", "") if isinstance(state, dict) else state.metric
            )

            human_msg = HumanMessage(
                content=f"""Competition: {competition_name}
Type: {competition_type}
Metric: {metric}

Data Characteristics:
{characteristics_str}

EDA Insights:
{insights_str}

Based on this information, provide a comprehensive strategy including:
1. Which models to prioritize
2. Feature engineering techniques to apply
3. Encoding strategies for categorical variables
4. Cross-validation approach
5. Whether to use ensembling and which technique
6. Any other critical considerations"""
            )

            strategy = llm_with_structure.invoke([system_msg, human_msg])

            # Store strategy in state
            strategy_dict = {
                "problem_characteristics": strategy.problem_characteristics,
                "recommended_models": strategy.recommended_models,
                "feature_engineering_priorities": strategy.feature_engineering_priorities,
                "validation_strategy": strategy.validation_strategy,
                "encoding_strategy": {
                    "low_cardinality": strategy.encoding_low_cardinality,
                    "high_cardinality": strategy.encoding_high_cardinality,
                },
                "scaling_required": strategy.scaling_required,
                "ensemble_strategy": strategy.ensemble_strategy,
                "data_characteristics": data_chars,
            }

            # Handle messages state access
            messages = (
                state.get("messages", []) if isinstance(state, dict) else state.messages
            )
            messages.append(
                HumanMessage(
                    content=f"""Strategy formulated:
- Recommended models: {", ".join(strategy.recommended_models)}
- Validation: {strategy.validation_strategy}
- Feature engineering: {", ".join(strategy.feature_engineering_priorities[:3])}
- Ensemble: {strategy.ensemble_strategy}"""
                )
            )

            print(
                f"Strategy Agent: Formulated strategy with {len(strategy.recommended_models)} priority models"
            )

            return {"eda_summary": {"strategy": strategy_dict}}

        except Exception as e:
            error_msg = f"Strategy formulation failed: {str(e)}"
            print(f"Strategy Agent ERROR: {error_msg}")
            # Return state with error appended, don't lose existing state
            errors = (
                state.get("errors", []) if isinstance(state, dict) else state.errors
            )
            return {"errors": errors + [error_msg]}
