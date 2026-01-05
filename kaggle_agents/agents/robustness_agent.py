"""
Robustness Agent with 5 Validation Modules.

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


# ==================== Hyperparameter Validation Prompt ====================

HYPERPARAMETER_VALIDATION_PROMPT = """You are an expert ML engineer reviewing code and execution logs for hyperparameter issues.

## Code
```python
{code}
```

## Execution Logs
**STDOUT (last 2000 chars)**:
```
{stdout}
```

**STDERR (last 1000 chars)**:
```
{stderr}
```

## Your Task
Analyze the code and logs to detect hyperparameter configuration issues:

1. **Tree Model Issues**: LightGBM, XGBoost, CatBoost, Random Forest
   - min_child_samples/min_data_in_leaf too restrictive
   - num_leaves/max_depth misconfiguration
   - Split failures ("best gain: -inf", "no valid split")

2. **Neural Network Issues**: PyTorch, TensorFlow, Keras
   - Learning rate problems (too high/low)
   - Batch size issues (OOM, instability)
   - Gradient issues (exploding/vanishing)

3. **General Issues**:
   - Memory problems
   - Class imbalance not handled
   - Convergence warnings

## Response Format
Return a JSON object:
{{
    "issues": [
        "Issue 1 description",
        "Issue 2 description"
    ],
    "suggestions": [
        "Suggestion 1",
        "Suggestion 2"
    ],
    "severity": "critical" | "warning" | "info",
    "score": 0.0 to 1.0 (1.0 = no issues, 0.7 = warnings, <0.7 = critical),
    "details": {{
        "key": "value for any relevant extracted parameters"
    }}
}}

If no issues found:
{{
    "issues": [],
    "suggestions": [],
    "severity": "info",
    "score": 1.0,
    "details": {{}}
}}

Be specific and actionable. Reference exact parameter values when possible."""


class RobustnessAgent:
    """
    Agent responsible for validating code robustness.

    Implements 5 validation modules (from Google ADK):
    1. Debugging: Auto-fix common errors
    2. Data Leakage: Detect target leakage
    3. Data Usage: Ensure all data is used
    4. Format Compliance: Validate submission format
    5. Hyperparameters: Detect model configuration issues
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
        print("\n" + "=" * 60)
        print("=  ROBUSTNESS AGENT: Validating Code")
        print("=" * 60)

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

        # 5. Hyperparameter Sanity Module (warning-only)
        hyperparam_result = self._validate_hyperparameters(latest_result, working_dir, state)
        validation_results.append(hyperparam_result)
        self._print_validation(hyperparam_result)

        # 6. Data Shapes Validation (FAIL-FAST on duplicates)
        shapes_result = self._validate_data_shapes(working_dir, state)
        validation_results.append(shapes_result)
        self._print_validation(shapes_result)

        # 7. Model Performance Gap Detection (triggers debug loops)
        gap_result = self._check_model_performance_gap(state)
        validation_results.append(gap_result)
        self._print_validation(gap_result)

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

    def _validate_leakage(
        self, dev_result, working_dir: Path, state: KaggleState
    ) -> ValidationResult:
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
                    for line in code_preview.split("\n"):
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

    def _validate_data_usage(
        self, dev_result, working_dir: Path, state: KaggleState
    ) -> ValidationResult:
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
            if "print" not in code[max(0, code.find(".head(") - 50) : code.find(".head(") + 50]:
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

    def _validate_format(
        self, dev_result, working_dir: Path, state: KaggleState
    ) -> ValidationResult:
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

    def _validate_hyperparameters(
        self, dev_result, working_dir: Path, state: KaggleState
    ) -> ValidationResult:
        """
        Module 5: LLM-driven hyperparameter sanity validation.

        Uses LLM to analyze code and execution logs for hyperparameter
        issues across all ML frameworks (LightGBM, XGBoost, CatBoost,
        sklearn, PyTorch, TensorFlow).

        Note: This is warning-only (does not block validation).
        """
        import json

        from langchain_core.messages import HumanMessage

        code = dev_result.code or ""
        stdout = (dev_result.stdout or "")[-2000:]  # Last 2000 chars
        stderr = (dev_result.stderr or "")[-1000:]  # Last 1000 chars

        # Prepare code summary (first 50 lines + last 20 lines if long)
        code_lines = code.split("\n")
        if len(code_lines) > 80:
            code_summary = "\n".join(code_lines[:50]) + "\n...\n" + "\n".join(code_lines[-20:])
        else:
            code_summary = code
        code_summary = code_summary[:4000]  # Limit token usage

        # Build prompt
        prompt = HYPERPARAMETER_VALIDATION_PROMPT.format(
            code=code_summary,
            stdout=stdout,
            stderr=stderr,
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = get_text_content(response.content).strip()

            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            issues = result.get("issues", [])
            suggestions = result.get("suggestions", [])
            severity = result.get("severity", "info")
            score = result.get("score", 1.0)
            details = result.get("details", {})

            # Ensure score is in valid range
            score = max(0.0, min(1.0, float(score)))

            # Print summary if issues found
            if issues:
                print(f"   ⚠️  Hyperparameter Analysis ({severity}):")
                for issue in issues[:3]:
                    print(f"      - {issue}")

        except Exception as e:
            print(f"   ⚠️  LLM hyperparameter analysis failed: {e}")
            # Fallback to no issues
            issues = []
            suggestions = []
            score = 1.0
            details = {"llm_error": str(e)}

        # Warning-only: ensure minimum score of 0.7 (doesn't block validation)
        score = max(score, 0.7)
        passed = True  # Warning-only module

        return ValidationResult(
            module="hyperparameters",
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
            details=details,
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

    def _validate_data_shapes(
        self, working_dir: Path, state: KaggleState
    ) -> ValidationResult:
        """
        Module 6: Data shapes validation (FAIL-FAST on duplicates).

        Validates engineered data shapes match original data.
        DOES NOT auto-fix with drop_duplicates - forces code rewrite.
        """
        issues = []
        suggestions = []
        score = 1.0
        details = {}

        # Check train shapes
        train_orig = working_dir / "train.csv"
        train_eng = working_dir / "train_engineered.csv"

        if train_orig.exists() and train_eng.exists():
            try:
                orig_df = pd.read_csv(train_orig)
                eng_df = pd.read_csv(train_eng)

                # Row count validation
                if len(eng_df) != len(orig_df):
                    issues.append(
                        f"CRITICAL: train_engineered.csv has {len(eng_df)} rows, "
                        f"but train.csv has {len(orig_df)} rows"
                    )
                    suggestions.append(
                        "REWRITE feature engineering code - DO NOT use drop_duplicates()"
                    )
                    score = 0.0
                    details["train_row_mismatch"] = {
                        "original": len(orig_df),
                        "engineered": len(eng_df),
                    }

                # Duplicate ID validation (FAIL-FAST - no auto-fix)
                if "id" in eng_df.columns and eng_df["id"].duplicated().any():
                    dup_count = eng_df["id"].duplicated().sum()
                    dup_ids = eng_df[eng_df["id"].duplicated()]["id"].head(5).tolist()
                    issues.append(
                        f"CRITICAL: {dup_count} duplicate IDs in train_engineered.csv! "
                        f"Examples: {dup_ids}"
                    )
                    suggestions.append(
                        "FIX the feature engineering code that creates duplicates. "
                        "DO NOT use drop_duplicates() - this corrupts data!"
                    )
                    score = 0.0
                    details["train_duplicate_ids"] = dup_count
                    details["action"] = "REWRITE_CODE"

                # Target column validation
                target_candidates = ["target", "species", "label", "class"]
                has_target = any(col in eng_df.columns for col in target_candidates)
                orig_target = [col for col in target_candidates if col in orig_df.columns]

                if orig_target and not has_target:
                    issues.append(
                        f"CRITICAL: Target column '{orig_target[0]}' missing in train_engineered.csv"
                    )
                    suggestions.append(
                        "Preserve target column when creating engineered features"
                    )
                    score = min(score, 0.3)
                    details["missing_target"] = orig_target[0]

                # Feature count warning (not critical)
                if len(eng_df.columns) < len(orig_df.columns) * 0.5:
                    issues.append(
                        f"WARNING: train_engineered.csv has {len(eng_df.columns)} columns, "
                        f"original has {len(orig_df.columns)} - features may be lost"
                    )
                    suggestions.append("Preserve original features when engineering new ones")
                    score = min(score, 0.7)

            except Exception as e:
                issues.append(f"Error validating train data shapes: {e}")
                score = min(score, 0.5)

        # Check test shapes
        test_orig = working_dir / "test.csv"
        test_eng = working_dir / "test_engineered.csv"

        if test_orig.exists() and test_eng.exists():
            try:
                orig_df = pd.read_csv(test_orig)
                eng_df = pd.read_csv(test_eng)

                if len(eng_df) != len(orig_df):
                    issues.append(
                        f"CRITICAL: test_engineered.csv has {len(eng_df)} rows, "
                        f"but test.csv has {len(orig_df)} rows"
                    )
                    suggestions.append(
                        "REWRITE feature engineering code - DO NOT use drop_duplicates()"
                    )
                    score = 0.0
                    details["test_row_mismatch"] = {
                        "original": len(orig_df),
                        "engineered": len(eng_df),
                    }

                if "id" in eng_df.columns and eng_df["id"].duplicated().any():
                    dup_count = eng_df["id"].duplicated().sum()
                    issues.append(
                        f"CRITICAL: {dup_count} duplicate IDs in test_engineered.csv!"
                    )
                    score = 0.0
                    details["test_duplicate_ids"] = dup_count

            except Exception as e:
                issues.append(f"Error validating test data shapes: {e}")
                score = min(score, 0.5)

        passed = score >= 0.7

        return ValidationResult(
            module="data_shapes",
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
            details=details,
        )

    def _check_model_performance_gap(self, state: KaggleState) -> ValidationResult:
        """
        Module 7: Model performance gap detection.

        Detects when one model performs drastically worse than others,
        triggering dedicated debug loops.
        """
        issues = []
        suggestions = []
        score = 1.0
        details = {}

        # Extract model scores from development results
        # DevelopmentResult doesn't have component metadata - get it from ablation_plan
        dev_results = state.get("development_results", [])
        ablation_plan = state.get("ablation_plan", [])
        model_scores = {}

        for i, result in enumerate(dev_results):
            # Get component metadata from ablation_plan (not from result)
            component = ablation_plan[i] if i < len(ablation_plan) else None
            component_type = component.component_type if component else None
            component_name = component.name if component else f"component_{i}"

            if component_type == "model":
                # Try to extract score from stdout
                stdout = getattr(result, "stdout", "") or ""

                # Look for common score patterns (re is imported at module level)
                patterns = [
                    r"(?:CV|Validation|Val|OOF).*?(?:Score|Loss|logloss|LogLoss|RMSE|MAE|AUC).*?:\s*([\d.]+)",
                    r"(?:Score|Loss|logloss|LogLoss).*?:\s*([\d.]+)",
                    r"Final.*?(?:Score|Loss).*?:\s*([\d.]+)",
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, stdout, re.IGNORECASE)
                    if matches:
                        try:
                            model_scores[component_name] = float(matches[-1])
                            break
                        except ValueError:
                            continue

        details["model_scores"] = model_scores

        if len(model_scores) >= 2:
            scores = list(model_scores.values())
            max_gap = max(scores) - min(scores)
            details["max_gap"] = max_gap

            # For logloss (lower is better), gap > 1.0 is HUGE
            if max_gap > 1.0:
                worst_model = max(model_scores, key=model_scores.get)
                best_model = min(model_scores, key=model_scores.get)

                issues.append(
                    f"PERFORMANCE GAP: {worst_model} (score {model_scores[worst_model]:.4f}) "
                    f"is {max_gap:.2f} worse than {best_model} ({model_scores[best_model]:.4f})"
                )
                suggestions.extend([
                    f"TRIGGER DEBUG LOOP for {worst_model}",
                    "Check if label encoding is consistent between models",
                    "Verify class_weight='balanced' is appropriate for this metric",
                    "Compare data preprocessing between models",
                    "Check if same train/val splits are used",
                ])
                score = 0.5
                details["trigger_debug"] = True
                details["worst_model"] = worst_model
                details["best_model"] = best_model
                details["action"] = "DEBUG_WORST_MODEL"

            elif max_gap > 0.5:
                # Moderate gap - warning only
                issues.append(
                    f"Moderate performance gap ({max_gap:.2f}) between models"
                )
                suggestions.append("Consider investigating model consistency")
                score = 0.8

        passed = score >= 0.7

        return ValidationResult(
            module="performance_gap",
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
            details=details,
        )


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
