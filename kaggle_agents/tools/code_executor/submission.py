"""
Submission validation for code execution.

Contains methods for validating submission format and extracting metrics.
"""

from __future__ import annotations

import re
from pathlib import Path


class SubmissionValidationMixin:
    """Mixin providing submission validation methods."""

    # Constant for gating pixel-level submissions
    MAX_ROWS_FOR_VALIDATION = 100_000

    def _should_validate_submission(
        self,
        component_type: str,
        sample_submission_path: Path | None,
    ) -> bool:
        """
        Determine if submission validation should run.
        Gated to avoid bottlenecks in pixel-level tasks.

        Args:
            component_type: Type of component ('model', 'ensemble', etc.)
            sample_submission_path: Path to sample_submission.csv

        Returns:
            True if validation should run, False otherwise
        """
        # Only validate model/ensemble components
        if component_type not in ("model", "ensemble"):
            return False

        # Skip if sample_submission doesn't exist
        if not sample_submission_path or not sample_submission_path.exists():
            return False

        # Skip if pixel-level (too many rows)
        try:
            with open(sample_submission_path) as f:
                row_count = sum(1 for _ in f) - 1  # -1 for header
            if row_count > self.MAX_ROWS_FOR_VALIDATION:
                print(f"   ⏭️ Skipping submission validation: {row_count} rows (pixel-level)")
                return False
        except Exception:
            return False

        return True

    def _detect_problem_type(self, sample_submission_path: Path) -> str:
        """
        Detect problem type from sample_submission structure.

        Args:
            sample_submission_path: Path to sample_submission.csv

        Returns:
            'multiclass', 'multilabel', 'binary', or 'regression'
        """
        import numpy as np

        from kaggle_agents.utils.csv_utils import read_csv_auto

        sample_df = read_csv_auto(sample_submission_path)
        pred_cols = sample_df.columns[1:].tolist()

        if len(pred_cols) == 1:
            # Single column: regression or binary
            values = sample_df[pred_cols[0]].dropna()
            if values.dtype in ["int64", "int32"] and set(values.unique()).issubset({0, 1}):
                return "binary"
            return "regression"

        # Multiple columns
        # Multilabel: values are 0/1 independent (don't sum to 1)
        # Multiclass: probabilities (sum to ~1)
        row_sums = sample_df[pred_cols].sum(axis=1)
        if np.allclose(row_sums, 1.0, atol=0.1):
            return "multiclass"
        return "multilabel"

    def validate_submission_format(
        self,
        submission_path: Path,
        sample_submission_path: Path,
        component_type: str | None = None,
        problem_type: str | None = None,
    ) -> tuple[bool, str]:
        """
        Validate submission matches expected format exactly.

        Performs gated validation with problem-type-aware checks.

        Args:
            submission_path: Path to generated submission.csv
            sample_submission_path: Path to sample_submission.csv
            component_type: Type of component (for gating)
            problem_type: Override problem type detection

        Returns:
            Tuple of (is_valid, message)
        """
        import numpy as np
        import pandas as pd

        from kaggle_agents.utils.csv_utils import read_csv_auto

        # Gating check
        if not self._should_validate_submission(component_type, sample_submission_path):
            return True, "Validation skipped (gated)"

        # Read files with auto-delimiter detection
        try:
            sub_df = pd.read_csv(submission_path)  # Submission always uses comma
            sample_df = read_csv_auto(sample_submission_path)  # Sample may use non-standard delimiter
        except Exception as e:
            return False, f"Failed to read files: {e}"

        # Auto-detect problem type if not provided
        if problem_type is None:
            problem_type = self._detect_problem_type(sample_submission_path)

        # Check 1: Columns match exactly (order matters!)
        if list(sub_df.columns) != list(sample_df.columns):
            return False, (
                f"Column mismatch!\n"
                f"  Expected: {sample_df.columns.tolist()}\n"
                f"  Got: {sub_df.columns.tolist()}"
            )

        # Check 2: Row count matches
        if len(sub_df) != len(sample_df):
            return False, f"Row count mismatch: expected {len(sample_df)}, got {len(sub_df)}"

        # Check 3: ID column matches exactly
        id_col = sample_df.columns[0]
        if not sub_df[id_col].equals(sample_df[id_col]):
            if set(sub_df[id_col]) == set(sample_df[id_col]):
                return False, "ID values present but in WRONG ORDER"
            return False, "ID column values don't match sample_submission"

        # Check 4: No NaN values in prediction columns
        pred_cols = sample_df.columns[1:].tolist()
        nan_cols = sub_df[pred_cols].isna().any()
        if nan_cols.any():
            bad_cols = nan_cols[nan_cols].index.tolist()
            return False, f"NaN values in columns: {bad_cols}"

        # Check 5: No Inf values
        inf_mask = ~np.isfinite(sub_df[pred_cols].values)
        if inf_mask.any():
            return False, "Inf values detected in predictions"

        # Check 6: Probabilities sum to ~1 ONLY for multiclass
        if problem_type == "multiclass" and len(pred_cols) > 1:
            row_sums = sub_df[pred_cols].sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=0.01):
                bad_rows = (~np.isclose(row_sums, 1.0, atol=0.01)).sum()
                return False, f"{bad_rows} rows don't sum to 1.0 (multiclass probabilities)"

        return True, f"✅ Submission format validated ({problem_type})"

    def extract_performance_metric(self, stdout: str):
        """
        Extracts validation performance score from code output (MLE-STAR pattern).

        Args:
            stdout: Standard output from code execution

        Returns:
            Performance score if found, None otherwise
        """
        for line in stdout.splitlines():
            if "Final Validation Performance:" in line:
                try:
                    # Extract score after the colon
                    score_str = line.split(":")[-1].strip()
                    # Remove any non-numeric characters except decimal point and minus
                    score_str = re.sub(r"[^\d.\-]", "", score_str)
                    return float(score_str)
                except (ValueError, IndexError):
                    continue
        return None
