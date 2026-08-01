"""
Submission validation for code execution.

Contains methods for validating submission format and extracting metrics.
"""

from __future__ import annotations

import re
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_agents.utils.csv_utils import read_csv_auto


class SubmissionValidationMixin:
    """Mixin providing submission validation methods."""

    CSV_VALIDATION_CHUNK_ROWS = 50_000

    def _should_validate_submission(
        self,
        component_type: str,
        sample_submission_path: Path | None,
    ) -> bool:
        """
        Determine if submission validation should run.

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
        return bool(
            sample_submission_path and sample_submission_path.exists()
        )

    def _detect_problem_type(
        self,
        sample_submission_path: Path,
        target_cols: list[str] | None = None,
    ) -> str:
        """
        Detect problem type from sample_submission structure.

        Args:
            sample_submission_path: Path to sample_submission.csv

        Returns:
            'multiclass', 'multilabel', 'binary', or 'regression'
        """
        from kaggle_agents.agents.ensemble.submission import prediction_positions

        sample_df = read_csv_auto(sample_submission_path, nrows=2048)
        pred_cols = [
            sample_df.columns[position]
            for position in prediction_positions(sample_df, target_cols)
        ]
        numeric_cols = [
            column
            for column in pred_cols
            if pd.api.types.is_numeric_dtype(sample_df[column])
        ]
        if not numeric_cols:
            # The positional slice landed entirely on echoed test input. Fall
            # back to whichever columns the template actually asks for numbers
            # in, so a template whose prediction comes first still resolves.
            pred_cols = [
                str(column)
                for column in sample_df.columns
                if pd.api.types.is_numeric_dtype(sample_df[column])
            ] or pred_cols
        else:
            pred_cols = numeric_cols

        if len(pred_cols) == 1:
            # Single column: regression or binary
            values = sample_df[pred_cols[0]].dropna()
            if values.dtype in ["int64", "int32"] and set(values.unique()).issubset({0, 1}):
                return "binary"
            return "regression"

        # Multiple columns
        # Multilabel: values are 0/1 independent (don't sum to 1)
        # Multiclass: probabilities (sum to ~1)
        # Templates that echo test input alongside the prediction carry text
        # columns; summing those raises instead of describing the task.
        numeric = sample_df[pred_cols].apply(pd.to_numeric, errors="coerce")
        row_sums = numeric.sum(axis=1)
        if np.allclose(row_sums, 1.0, atol=0.1):
            return "multiclass"
        return "multilabel"

    def validate_submission_format(  # noqa: PLR0911, PLR0912
        self,
        submission_path: Path,
        sample_submission_path: Path,
        component_type: str | None = None,
        problem_type: str | None = None,
        target_cols: list[str] | None = None,
        require_normalized_rows: bool = True,
    ) -> tuple[bool, str]:
        """
        Validate submission matches expected format exactly.

        Performs streaming validation with problem-type-aware checks.

        Args:
            submission_path: Path to generated submission.csv
            sample_submission_path: Path to sample_submission.csv
            component_type: Type of component (for gating)
            problem_type: Override problem type detection
            target_cols: Resolved prediction column names. Without them the
                first column is assumed to identify rows, which misreads
                templates whose first column is the prediction.
            require_normalized_rows: Whether multiclass rows must sum to 1.
                Only true when the graded metric reads a row as a probability
                vector. Under a column-wise ranking metric the grader accepts
                and scores these predictions unchanged, so rejecting them
                discards a valid submission to enforce a property nobody
                measures.

        Returns:
            Tuple of (is_valid, message)
        """
        # Gating check
        if not self._should_validate_submission(component_type, sample_submission_path):
            return True, "Validation skipped (gated)"

        # Read only schema/role samples up front. Full content validation below
        # is chunked so pixel-level files cannot bypass the contract.
        try:
            sub_header = pd.read_csv(submission_path, nrows=0)
            sample_header = read_csv_auto(sample_submission_path, nrows=0)
            sample_preview = read_csv_auto(sample_submission_path, nrows=2048)
        except Exception as e:
            return False, f"Failed to read files: {e}"

        # Check 1: Columns match exactly (order matters!)
        if list(sub_header.columns) != list(sample_header.columns):
            return False, (
                f"Column mismatch!\n"
                f"  Expected: {sample_header.columns.tolist()}\n"
                f"  Got: {sub_header.columns.tolist()}"
            )

        # Auto-detect problem type if not provided
        if problem_type is None:
            problem_type = self._detect_problem_type(
                sample_submission_path, target_cols
            )
        normalized_problem_type = (
            str(problem_type or "").strip().lower().replace("-", "_")
        )
        is_multiclass = "multiclass" in normalized_problem_type

        # Roles, not positions: a template may put the prediction first and echo
        # the test input after it. Comparing the prediction column against the
        # template's placeholders would reject every real submission, and
        # running numeric checks over echoed text raises.
        from kaggle_agents.agents.ensemble.submission import prediction_positions

        predicted = {
            sample_header.columns[position]
            for position in prediction_positions(sample_preview, target_cols)
        }
        pred_cols = [
            column for column in sample_header.columns if column in predicted
        ]
        echo_cols = [
            column for column in sample_header.columns if column not in predicted
        ]
        label_prediction_format = (
            len(pred_cols) == 1
            and (
                is_multiclass
                or not pd.api.types.is_numeric_dtype(
                    sample_preview[pred_cols[0]]
                )
            )
        )
        try:
            submission_chunks = pd.read_csv(
                submission_path,
                chunksize=self.CSV_VALIDATION_CHUNK_ROWS,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
            )
            sample_chunks = read_csv_auto(
                sample_submission_path,
                chunksize=self.CSV_VALIDATION_CHUNK_ROWS,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
            )
            submission_rows = 0
            sample_rows = 0
            template_unchanged = True
            for sub_chunk, sample_chunk in zip_longest(
                submission_chunks,
                sample_chunks,
            ):
                if sub_chunk is None:
                    remaining = len(sample_chunk) if sample_chunk is not None else 0
                    sample_rows += remaining
                    return False, (
                        "Row count mismatch: expected at least "
                        f"{sample_rows}, got {submission_rows}"
                    )
                if sample_chunk is None:
                    submission_rows += len(sub_chunk)
                    return False, (
                        f"Row count mismatch: expected {sample_rows}, "
                        f"got at least {submission_rows}"
                    )
                submission_rows += len(sub_chunk)
                sample_rows += len(sample_chunk)
                if len(sub_chunk) != len(sample_chunk):
                    return False, (
                        f"Row count mismatch: expected {sample_rows}, "
                        f"got {submission_rows}"
                    )

                # Every template-supplied column, not only the first ID, is an
                # ordered sequence contract.
                for echo_col in echo_cols:
                    if not sub_chunk[echo_col].equals(sample_chunk[echo_col]):
                        if set(sub_chunk[echo_col]) == set(
                            sample_chunk[echo_col]
                        ):
                            return (
                                False,
                                f"'{echo_col}' values present but in WRONG ORDER",
                            )
                        return False, (
                            f"'{echo_col}' does not match sample_submission. "
                            "This column is supplied by the template and must "
                            f"be returned unchanged; predictions belong in "
                            f"{pred_cols}. Write the submission with "
                            "write_submission(test_preds) instead of choosing "
                            "columns by position."
                        )

                if label_prediction_format:
                    if sub_chunk[pred_cols[0]].eq("").any():
                        return False, (
                            f"Blank label values in column: {pred_cols[0]}"
                        )
                    if template_unchanged:
                        template_unchanged = sub_chunk[pred_cols].equals(
                            sample_chunk[pred_cols]
                        )
                    numeric_values = None
                else:
                    numeric_predictions = sub_chunk[pred_cols].apply(
                        pd.to_numeric,
                        errors="coerce",
                    )
                    invalid = numeric_predictions.isna().any()
                    if invalid.any():
                        bad_cols = invalid[invalid].index.tolist()
                        return (
                            False,
                            f"NaN or non-numeric values in columns: {bad_cols}",
                        )
                    numeric_values = numeric_predictions.to_numpy(dtype=float)
                    if bool(np.isinf(numeric_values).any()):
                        return False, "Inf values detected in predictions"

                    sample_predictions = sample_chunk[pred_cols].apply(
                        pd.to_numeric,
                        errors="coerce",
                    )
                    if template_unchanged:
                        template_unchanged = np.array_equal(
                            numeric_values,
                            sample_predictions.to_numpy(dtype=float),
                            equal_nan=True,
                        )

                if (
                    is_multiclass
                    and require_normalized_rows
                    and len(pred_cols) > 1
                    and numeric_values is not None
                ):
                    row_sums = numeric_values.sum(axis=1)
                    if not np.allclose(row_sums, 1.0, atol=0.01):
                        bad_rows = int(
                            (~np.isclose(row_sums, 1.0, atol=0.01)).sum()
                        )
                        return False, (
                            f"{bad_rows} rows don't sum to 1.0 "
                            "(multiclass probabilities)"
                        )
        except Exception as e:
            return False, f"Failed to stream validation files: {e}"

        if submission_rows != sample_rows:
            return False, (
                f"Row count mismatch: expected {sample_rows}, "
                f"got {submission_rows}"
            )
        if submission_rows and template_unchanged:
            return False, (
                "Prediction columns are unchanged from sample_submission; "
                "this is the sample submission template"
            )

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
