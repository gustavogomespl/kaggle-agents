"""
Utility functions for developer agent.

Provides helper methods for dataset info extraction and code parsing.
"""

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ...core.state import KaggleState


class DeveloperUtilsMixin:
    """Mixin providing utility methods."""

    def _get_dataset_info(self, working_dir: Path, state: "KaggleState" = None) -> str:
        """
        Read dataset columns and basic info to provide to LLM.

        Args:
            working_dir: Working directory containing train.csv
            state: Current state (optional)

        Returns:
            Formatted string with dataset information
        """
        try:
            import pandas as pd

            train_path = working_dir / "train.csv"

            if not train_path.exists():
                return "Dataset info not available (file not found)"

            df = pd.read_csv(train_path, nrows=5)

            columns = df.columns.tolist()
            dtypes = df.dtypes.to_dict()
            target_col = "UNKNOWN"

            if state and state.get("target_col"):
                target_col = state["target_col"]
            else:
                target_candidates = [
                    c
                    for c in columns
                    if c.lower()
                    in [
                        "target",
                        "label",
                        "y",
                        "class",
                        "loan_paid_back",
                        "survived",
                        "price",
                        "sales",
                    ]
                ]
                target_col = target_candidates[0] if target_candidates else "UNKNOWN"

            numeric_cols = [c for c, dtype in dtypes.items() if dtype in ["int64", "float64"]]
            categorical_cols = [c for c, dtype in dtypes.items() if dtype == "object"]

            return f"""
            **CRITICAL**: Use these EXACT column names from the dataset:

            Target Column: {target_col}
            Total Columns: {len(columns)}

            Numeric Columns ({len(numeric_cols)}): {", ".join(numeric_cols[:10])}{"..." if len(numeric_cols) > 10 else ""}
            Categorical Columns ({len(categorical_cols)}): {", ".join(categorical_cols[:10])}{"..." if len(categorical_cols) > 10 else ""}

            All Columns: {", ".join(columns)}

            IMPORTANT: Always use target_col='{target_col}' in your code!
            """

        except Exception as e:
            return f"Dataset info not available (error: {e!s})"

    def _get_domain_template(self, domain: str, component_type: str) -> str:
        """Get domain-specific code template.

        Args:
            domain: Competition domain (e.g., 'image_classification', 'text_classification')
            component_type: Component type (e.g., 'model', 'preprocessing')

        Returns:
            Domain-specific code template string (empty - templates removed in refactoring)
        """
        # Domain-specific templates removed in favor of agentic approach
        # Agent uses SOTA solutions and feedback instead of hardcoded templates
        return ""

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response

        return code.strip()
