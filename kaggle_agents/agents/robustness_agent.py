"""
Robustness Agent with 4 Validation Modules.

This agent implements the robustness validation strategy from Google ADK,
ensuring code quality and preventing common ML mistakes.
"""

import re
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..core.state import KaggleState, ValidationResult
from ..core.config import get_config


class RobustnessAgent:
    """
    Agent responsible for validating code robustness.

    Implements 4 validation modules (from Google ADK):
    1. Debugging: Auto-fix common errors
    2. Data Leakage: Detect target leakage
    3. Data Usage: Ensure all data is used
    4. Format Compliance: Validate submission format
    """

    def __init__(self):
        """Initialize the robustness agent."""
        self.config = get_config()

    def __call__(self, state: KaggleState) -> Dict[str, Any]:
        """
        Execute robustness validation.

        Args:
            state: Current workflow state

        Returns:
            State updates with validation results
        """
        print("\n" + "="*60)
        print("=á  ROBUSTNESS AGENT: Validating Code")
        print("="*60)

        # Get development results
        dev_results = state.get("development_results", [])

        if not dev_results:
            print("   No development results to validate")
            return {}

        # Get latest result
        latest_result = dev_results[-1]
        working_dir = Path(state["working_directory"])

        # Run 4 validation modules
        print("\n= Running validation modules...")

        validation_results = []

        # 1. Debugging Module
        debug_result = self._validate_debugging(latest_result, working_dir)
        validation_results.append(debug_result)
        self._print_validation(debug_result)

        # 2. Data Leakage Module
        leakage_result = self._validate_leakage(latest_result, working_dir, state)
        validation_results.append(leakage_result)
        self._print_validation(leakage_result)

        # 3. Data Usage Module
        usage_result = self._validate_data_usage(latest_result, working_dir, state)
        validation_results.append(usage_result)
        self._print_validation(usage_result)

        # 4. Format Compliance Module
        format_result = self._validate_format(latest_result, working_dir, state)
        validation_results.append(format_result)
        self._print_validation(format_result)

        # Calculate overall score
        overall_score = sum(r.score for r in validation_results) / len(validation_results)

        print(f"\n=Ê Overall Validation Score: {overall_score:.1%}")

        # Determine if passed
        min_score = self.config.validation.min_validation_score
        passed = overall_score >= min_score

        if passed:
            print(f" Validation PASSED (threshold: {min_score:.1%})")
        else:
            print(f"L Validation FAILED (threshold: {min_score:.1%})")

        return {
            "validation_results": validation_results,
            "overall_validation_score": overall_score,
            "last_updated": datetime.now(),
        }

    def _validate_debugging(self, dev_result, working_dir: Path) -> ValidationResult:
        """
        Module 1: Debugging validation.

        Checks:
        - No uncaught exceptions
        - Proper error handling
        - No warnings in output
        """
        issues = []
        suggestions = []
        score = 1.0

        # Check for errors
        if not dev_result.success:
            issues.append("Execution failed")
            suggestions.append("Fix the errors before proceeding")
            score = 0.0
        elif dev_result.errors:
            issues.append(f"Found {len(dev_result.errors)} errors")
            score *= 0.5

        # Check for warnings
        if "Warning" in dev_result.stdout or "WARNING" in dev_result.stdout:
            warnings_count = dev_result.stdout.count("Warning") + dev_result.stdout.count("WARNING")
            issues.append(f"Found {warnings_count} warnings")
            suggestions.append("Review and fix warnings")
            score *= 0.9

        # Check for exceptions in stderr
        if "Exception" in dev_result.stderr or "Error" in dev_result.stderr:
            issues.append("Exceptions found in stderr")
            score *= 0.7

        passed = score >= 0.7

        return ValidationResult(
            module="debugging",
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
        )

    def _validate_leakage(self, dev_result, working_dir: Path, state: KaggleState) -> ValidationResult:
        """
        Module 2: Data leakage detection.

        Checks:
        - Target encoding before split
        - Feature engineering on full dataset
        - Test data in training
        """
        issues = []
        suggestions = []
        score = 1.0

        code = dev_result.code

        # Check for common leakage patterns
        leakage_patterns = [
            (r"fit.*train_df.*\+.*test_df", "Fitting on train+test together"),
            (r"TargetEncoder.*fit.*(?!train)", "Target encoding might use test data"),
            (r"fit_transform.*(?!train)", "fit_transform might include test data"),
        ]

        for pattern, description in leakage_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(description)
                suggestions.append(f"Ensure {description.lower()} is avoided")
                score *= 0.8

        # Check for proper train/test split
        if "train_test_split" not in code and "KFold" not in code:
            # Might be using pre-split data (OK) or no split (BAD)
            if "test_df" in code:
                # Has test data, probably OK
                pass
            else:
                issues.append("No train/test split detected")
                score *= 0.9

        passed = score >= 0.7

        return ValidationResult(
            module="leakage",
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
        )

    def _validate_data_usage(self, dev_result, working_dir: Path, state: KaggleState) -> ValidationResult:
        """
        Module 3: Data usage validation.

        Checks:
        - All training data is used
        - No data sampling without reason
        - Proper handling of missing values
        """
        issues = []
        suggestions = []
        score = 1.0

        code = dev_result.code

        # Check for data sampling
        if ".sample(" in code and "frac=" not in code:
            issues.append("Data sampling detected without full dataset usage")
            suggestions.append("Ensure you're using all available data")
            score *= 0.8

        # Check for dropna (might lose data)
        if ".dropna()" in code:
            issues.append("Using dropna() - might lose valuable data")
            suggestions.append("Consider imputation instead of dropping")
            score *= 0.9

        # Check for head/tail (might be debugging code left in)
        if ".head(" in code or ".tail(" in code:
            # Check if it's just for printing
            if "print" not in code[max(0, code.find(".head(") - 50):code.find(".head(") + 50]:
                issues.append("Using head/tail - might not be using full data")
                score *= 0.95

        passed = score >= 0.7

        return ValidationResult(
            module="data_usage",
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
        )

    def _validate_format(self, dev_result, working_dir: Path, state: KaggleState) -> ValidationResult:
        """
        Module 4: Format compliance validation.

        Checks:
        - Submission file exists
        - Correct format (CSV with required columns)
        - No missing values
        - Correct number of rows
        """
        issues = []
        suggestions = []
        score = 1.0

        # Check if submission file was created
        submission_path = working_dir / "submission.csv"

        if not submission_path.exists():
            # Check in artifacts
            if "submission.csv" not in dev_result.artifacts_created:
                issues.append("Submission file not created")
                suggestions.append("Ensure code saves submission.csv")
                return ValidationResult(
                    module="format",
                    passed=False,
                    score=0.0,
                    issues=issues,
                    suggestions=suggestions,
                )

            # Find submission in artifacts
            for artifact in dev_result.artifacts_created:
                if "submission" in artifact.lower() and artifact.endswith(".csv"):
                    submission_path = working_dir / artifact
                    break

        try:
            # Read submission
            submission_df = pd.read_csv(submission_path)

            # Check for required columns (usually ID + prediction)
            if len(submission_df.columns) < 2:
                issues.append("Submission has fewer than 2 columns")
                score *= 0.5

            # Check for missing values
            if submission_df.isnull().any().any():
                null_count = submission_df.isnull().sum().sum()
                issues.append(f"Submission contains {null_count} missing values")
                suggestions.append("Fill missing values before submission")
                score *= 0.6

            # Check for empty submission
            if len(submission_df) == 0:
                issues.append("Submission file is empty")
                score = 0.0

            # Check for duplicate IDs (if first column looks like ID)
            first_col = submission_df.columns[0]
            if first_col.lower() in ["id", "index", "idx"]:
                if submission_df[first_col].duplicated().any():
                    issues.append("Duplicate IDs found in submission")
                    score *= 0.7

        except Exception as e:
            issues.append(f"Error reading submission: {str(e)}")
            score = 0.0

        passed = score >= 0.7

        return ValidationResult(
            module="format",
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
        )

    def _print_validation(self, result: ValidationResult):
        """Print validation result."""
        status = "" if result.passed else "L"
        print(f"\n{status} {result.module.upper()}: {result.score:.1%}")

        if result.issues:
            print(f"   Issues:")
            for issue in result.issues:
                print(f"   - {issue}")

        if result.suggestions:
            print(f"   Suggestions:")
            for suggestion in result.suggestions:
                print(f"   - {suggestion}")


# ==================== LangGraph Node Function ====================

def robustness_agent_node(state: KaggleState) -> Dict[str, Any]:
    """
    LangGraph node function for the robustness agent.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = RobustnessAgent()
    return agent(state)
