"""
Robustness Agent with 4 Validation Modules.

This agent implements the robustness validation strategy from Google ADK,
ensuring code quality and preventing common ML mistakes.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.config import get_config
from ..core.state import KaggleState, ValidationResult
from ..utils.llm_utils import get_text_content


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

    def __call__(self, state: KaggleState) -> dict[str, Any]:
        """
        Execute robustness validation.

        Args:
            state: Current workflow state

        Returns:
            State updates with validation results
        """
        print("\n" + "="*60)
        print("=  ROBUSTNESS AGENT: Validating Code")
        print("="*60)

        # Get development results
        dev_results = state.get("development_results", [])

        if not dev_results:
            print("  No development results to validate")
            return {}

        # Initialize LLM (supports OpenAI and Anthropic)
        from ..core.config import get_llm
        self.llm = get_llm()

        # Get latest result
        latest_result = dev_results[-1]
        working_dir = Path(state["working_directory"])

        # Run 4 validation modules
        print("\nRunning validation modules...")

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

        print(f"\n= Overall Validation Score: {overall_score:.1%}")

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

        # LLM-based Leakage Check (Enhanced with ADK-style structured output)
        import json

        from langchain_core.messages import HumanMessage

        prompt = f"""You are a data science expert reviewing code for data leakage.

CRITICAL: You MUST respond with valid JSON only. No markdown, no extra text.

Code to analyze:
```python
{code[:5000]}
```

# Your task
Analyze if this code has data leakage issues:

1. **Training Leakage**: Are validation or test samples used during model training?
2. **Preprocessing Leakage**: Are scalers/encoders/transformers fitted on validation/test data?
3. **Target Leakage**: Is the target variable used in feature engineering before split?
4. **Temporal Leakage**: Is future information used to predict the past?

# Common leakage patterns to check:
- `fit()` or `fit_transform()` on combined train+test
- `TargetEncoder.fit()` on validation/test data
- Preprocessing fitted globally before train/test split
- Training data statistics calculated from validation/test

# Response Format
Respond with this EXACT JSON format (no markdown, no extra text):

{{
    "leakage_status": "YES" or "NO",
    "code_block": "exact lines of code with leakage (empty string if NO)",
    "line_numbers": "approximate line range like '45-52' (empty string if NO)",
    "explanation": "brief 1-sentence explanation of the issue (or why it's clean)"
}}

IMPORTANT:
- If leakage detected: leakage_status = "YES"
- If no leakage: leakage_status = "NO"
- Always provide code_block and line_numbers if YES
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = get_text_content(response.content).strip()
            # Handle potential markdown wrapping
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            leakage_status = result.get("leakage_status", "NO").upper()
            code_block = result.get("code_block", "")
            line_numbers = result.get("line_numbers", "")
            explanation = result.get("explanation", "No explanation provided")

            if leakage_status == "YES":
                print("   ❌ Data Leakage Detected!")
                print(f"      Lines: {line_numbers}")
                print(f"      Issue: {explanation}")

                if code_block:
                    print("      Problematic Code:")
                    print("      ```python")
                    # Show first 300 chars of code block
                    code_preview = code_block[:300] + "..." if len(code_block) > 300 else code_block
                    for line in code_preview.split('\n'):
                        print(f"      {line}")
                    print("      ```")

                issues.append(f"Data Leakage ({line_numbers}): {explanation}")
                suggestions.append("Fix the code block identified above to prevent leakage")
                score = 0.0
                passed = False

                # Store structured leakage details for potential auto-correction
                leakage_details = {
                    "leakage_code_block": code_block,
                    "line_numbers": line_numbers,
                    "explanation": explanation,
                }
            else:
                print(f"   ✅ No Data Leakage: {explanation}")
                passed = True
                leakage_details = {}

        except Exception as e:
            print(f"   ⚠️  Warning: LLM Leakage Check failed: {e}")
            leakage_details = {}
            # Fallback to regex
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
            passed = score >= 0.7

        return ValidationResult(
            module="leakage",
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
            details=leakage_details,  # Structured information for auto-correction
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
            artifact_candidates = [
                artifact
                for artifact in dev_result.artifacts_created
                if "submission" in artifact.lower() and artifact.endswith(".csv")
            ]
            if not artifact_candidates:
                issues.append("Submission file not created")
                suggestions.append("Ensure code saves submission.csv")
                return ValidationResult(
                    module="format",
                    passed=False,
                    score=0.0,
                    issues=issues,
                    suggestions=suggestions,
                )

            submission_path = working_dir / artifact_candidates[0]

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
            issues.append(f"Error reading submission: {e!s}")
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
            print("   Issues:")
            for issue in result.issues:
                print(f"   - {issue}")

        if result.suggestions:
            print("   Suggestions:")
            for suggestion in result.suggestions:
                print(f"   - {suggestion}")


# ==================== LangGraph Node Function ====================

def robustness_agent_node(state: KaggleState) -> dict[str, Any]:
    """
    LangGraph node function for the robustness agent.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = RobustnessAgent()
    return agent(state)
