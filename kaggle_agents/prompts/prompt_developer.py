"""Prompts for the Developer agent."""

PROMPT_DEVELOPER_TASK = """# YOUR TASK #
Write executable, syntactically correct Python code for the "{phase_name}" phase.

PRIORITIES (in order):
1. SYNTACTICALLY VALID - zero syntax errors
2. CONCISE - avoid long comments, get straight to implementation
3. FUNCTIONAL - code must run successfully
4. FOLLOWS PLAN - implement the specified steps
5. SAVES RESULTS - write outputs to appropriate files
"""

PROMPT_DEVELOPER = """# CONTEXT #
You are implementing code for the "{phase_name}" phase.

## Current State ##
{state_info}

## Plan to Implement ##
{plan}

## Available Tools ##
{tools}

## Data Information ##
{data_info}

## Previous Experience ##
{experience}

{task}

# CRITICAL CODE REQUIREMENTS #

**SYNTAX (HIGHEST PRIORITY):**
- ✓ ALWAYS close strings (single, double, and triple quotes)
- ✓ ALWAYS close parentheses (), brackets [], braces {{}}
- ✓ ALWAYS complete try blocks with except or finally
- ✓ ALWAYS finish loops and conditionals properly
- ✓ Use multiline strings CAREFULLY - prefer single-line strings when possible
- ✓ Avoid breaking strings across lines without proper escaping
- ✓ Test your syntax mentally before returning the code

**FUNCTIONALITY:**
1. Import libraries at top (use only: pandas, numpy, matplotlib, sklearn, pathlib, pickle, joblib)
2. Load data from specified paths
3. Implement each plan step DIRECTLY and CONCISELY
4. Save outputs to: {restore_dir}
5. Print brief progress messages
6. Use simple try/except for error handling

**PHASE-SPECIFIC REQUIREMENTS:**
- For "Model Building" phase: MUST create and save:
  * Trained model file(s) in models/ directory (use pickle or joblib)
  * Submission CSV file in submissions/ directory (matching sample_submission.csv format)
  * Predictions on test data
- For "Feature Engineering" phase: MUST save:
  * processed_train.csv with engineered features
  * processed_test.csv with same features applied
- For "EDA" phases: MUST save:
  * Analysis results as JSON or TXT
  * Plots/visualizations as PNG files

**CODE STYLE:**
- CONCISE: comment only what's essential
- DIRECT: get to the point, no fluff
- FUNCTIONAL: prioritize code that RUNS over "pretty" code
- PRACTICAL: use simple solutions, avoid over-engineering

# RESPONSE FORMAT #
Return ONLY the executable Python code block:

```python
# {phase_name} - Concise Implementation
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
COMP_DIR = Path("{restore_dir}").parent
OUTPUT_DIR = Path("{restore_dir}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# For Model Building phase, also ensure these directories exist
MODELS_DIR = COMP_DIR / "models"
SUBMISSIONS_DIR = COMP_DIR / "submissions"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Your implementation here - DIRECT AND FUNCTIONAL

if __name__ == "__main__":
    print("[START] {phase_name}")
    # Main implementation
    print("[DONE] {phase_name}")
```

⚠️ CRITICAL: Verify syntax before returning. Code with syntax errors is UNACCEPTABLE.
⚠️ KEEP IT CONCISE: If running out of space, prioritize correct, working code over comments.
⚠️ FOR MODEL BUILDING: You MUST save model files and create submission.csv - this is mandatory!
"""

PROMPT_EXTRACT_TOOLS = """# TASK #
Extract the relevant tool names from the plan that should be used in this phase.

# DOCUMENT #
{document}

# ALL AVAILABLE TOOL NAMES #
{all_tool_names}

# INSTRUCTIONS #
Analyze the plan and identify which tools from the available tools list are needed.
Return only the tool names that are explicitly mentioned or clearly required by the plan.

# RESPONSE FORMAT #
```json
{{
  "tool_names": ["tool1", "tool2"]
}}
```
"""

PROMPT_FIX_CODE = """# TASK #
Fix the code based on the error messages and test results.

## Previous Code ##
{code}

## Error Messages ##
{errors}

## Test Results ##
{test_results}

# INSTRUCTIONS #
1. Analyze the errors carefully
2. Identify the root cause
3. Fix the code while maintaining functionality
4. Ensure the fix addresses the specific error
5. Don't introduce new bugs

# RESPONSE FORMAT #
```python
# Fixed code

[Your corrected code here]
```

# EXPLANATION #
Briefly explain what was wrong and how you fixed it.
"""

PROMPT_MODEL_BUILDING_TEMPLATE = """
# CRITICAL: For Model Building phase, your code MUST include these steps:

## 1. Load Data
```python
train = pd.read_csv(COMP_DIR / "data" / "processed_train.csv")
test = pd.read_csv(COMP_DIR / "data" / "processed_test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data" / "sample_submission.csv")
```

## 2. Prepare Features and Target
```python
target_col = sample_sub.columns[1]  # Usually second column
X = train.drop(columns=[target_col, 'id'] if 'id' in train.columns else [target_col])
y = train[target_col]
X_test = test.drop(columns=['id'] if 'id' in test.columns else [])
```

## 3. Train Model
```python
from sklearn.ensemble import RandomForestRegressor  # or appropriate model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
```

## 4. MANDATORY: Save Model
```python
import joblib
model_path = MODELS_DIR / "model.pkl"
joblib.dump(model, model_path)
print(f"Model saved to: {{model_path}}")
```

## 5. MANDATORY: Create Submission
```python
predictions = model.predict(X_test)
submission = sample_sub.copy()
submission[target_col] = predictions
submission_path = SUBMISSIONS_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to: {{submission_path}}")
print(f"Submission shape: {{submission.shape}}")
```

⚠️ CRITICAL: Steps 4 and 5 are MANDATORY for Model Building phase!
"""

PROMPT_FEATURE_ENGINEERING_TEMPLATE = """
# CRITICAL: For Feature Engineering phase, your code MUST save processed data:

## Load Original Data
```python
train = pd.read_csv(COMP_DIR / "data" / "train.csv")
test = pd.read_csv(COMP_DIR / "data" / "test.csv")
```

## Apply Feature Engineering
```python
# Your feature engineering code here
# Create new features, transform existing ones, etc.
```

## MANDATORY: Save Processed Data
```python
processed_train_path = COMP_DIR / "data" / "processed_train.csv"
processed_test_path = COMP_DIR / "data" / "processed_test.csv"

train.to_csv(processed_train_path, index=False)
test.to_csv(processed_test_path, index=False)

print(f"Processed train saved to: {{processed_train_path}}")
print(f"Processed test saved to: {{processed_test_path}}")
print(f"Train shape: {{train.shape}}, Test shape: {{test.shape}}")
```

⚠️ CRITICAL: Model Building phase needs processed_train.csv and processed_test.csv!
"""

PROMPT_DEBUG_CODE = """# DEBUGGING TASK #
The code execution failed. Fix it quickly and efficiently.

## Code ##
```python
{code}
```

## Error Output ##
```
{error_output}
```

## Error Type ##
{error_type}

# FIX STRATEGY #
1. Identify exact line causing error
2. Check common issues:
   - Unclosed strings/quotes (triple quotes, double quotes, single quotes)
   - Unclosed brackets/parentheses/braces
   - Incomplete try/except blocks
   - Missing imports (especially seaborn - DO NOT USE IT)
   - Type mismatches
   - File path issues
3. Return ONLY the corrected code

# CRITICAL RULES #
- If error mentions seaborn or ModuleNotFoundError, REMOVE all seaborn imports
- Use matplotlib.pyplot instead of seaborn
- Verify ALL strings are properly closed
- Verify ALL brackets/parentheses are balanced
- Keep the fix MINIMAL - only change what's broken

# RESPONSE FORMAT #
Return ONLY the fixed code block:

```python
# Fixed code - syntax verified
[Your corrected code here]
```

⚠️ Do NOT add explanations. Return ONLY executable code.
"""
