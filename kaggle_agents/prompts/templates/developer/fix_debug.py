"""
Fix, debug, and refinement prompt templates for the Developer Agent.

Contains prompts used for error recovery and code improvement.
"""

# ==================== Fix and Debug Prompts ====================

FIX_CODE_PROMPT = """Fix this code error.

## Code
```python
{code}
```

## Error
{error}

## Error Type
{error_type}

## Meta-Feedback (use this to fix root cause)
{meta_feedback}

## Correct Data Paths (USE EXACTLY - DO NOT HARDCODE)
{paths}

## CRITICAL REQUIREMENTS (DO NOT REMOVE):
1. MUST preserve `print(f"Final Validation Performance: {{score:.6f}}")` - Meta-Evaluator depends on this exact string
2. MUST preserve soft-deadline pattern with `_check_deadline()` calls
3. MUST keep all OOF prediction saving (np.save)
4. For FileNotFoundError: Use the EXACT paths from "Correct Data Paths" section above. DO NOT hardcode 'train.csv' or 'test.csv'.

Fix the issue while preserving the component's intent. Return complete fixed code."""


DEBUG_CODE_PROMPT = """Debug this code that failed.

## Code
```python
{code}
```

## Issue
{issue}

## Stdout (last lines)
{stdout}

## Stderr
{stderr}

## Meta-Feedback (if available)
{meta_feedback}

## Correct Data Paths (USE EXACTLY - DO NOT HARDCODE)
{paths}

## CRITICAL REQUIREMENTS (DO NOT REMOVE):
1. MUST preserve `print(f"Final Validation Performance: {{score:.6f}}")` - Meta-Evaluator depends on this exact string
2. MUST preserve soft-deadline pattern with `_check_deadline()` calls
3. MUST keep all OOF prediction saving (np.save)
4. For FileNotFoundError or path issues: Use the EXACT paths from "Correct Data Paths" section above. DO NOT hardcode 'train.csv' or 'test.csv'.

Analyze the output, fix logic errors or missing imports, and return the complete debugged code."""


# ==================== Refinement Prompt ====================

REFINEMENT_WITH_FEEDBACK_PROMPT = """Refine this model based on training feedback.

## Current Score
CV: {current_score}

## Training Feedback
{training_feedback}

## Current Code
```python
{current_code}
```

## Improvement Guidelines
Based on the feedback:
- High variance (std > 0.02): Increase regularization, reduce depth
- Overfitting (train >> val): Add dropout, increase subsample
- Underfitting (low score): Decrease regularization, add features
- Optuna best params: Use as starting point

Keep the same [LOG:*] format for the feedback loop.
Return the complete improved code."""
