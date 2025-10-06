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
1. Import libraries at top (use only: pandas, numpy, matplotlib, sklearn, pathlib)
2. Load data from specified paths
3. Implement each plan step DIRECTLY and CONCISELY
4. Save outputs to: {restore_dir}
5. Print brief progress messages
6. Use simple try/except for error handling

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

# Your implementation here - DIRECT AND FUNCTIONAL

if __name__ == "__main__":
    print("[START] {phase_name}")
    # Main implementation
    print("[DONE] {phase_name}")
```

⚠️ CRITICAL: Verify syntax before returning. Code with syntax errors is UNACCEPTABLE.
⚠️ KEEP IT CONCISE: If you're running out of space, prioritize correct, working code over comments.
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
