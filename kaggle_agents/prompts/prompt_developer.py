"""Prompts for the Developer agent."""

PROMPT_DEVELOPER_TASK = """# YOUR TASK #
Implement the plan for the "{phase_name}" phase by writing production-quality Python code.

Your code should:
1. Follow the plan's specifications exactly
2. Use the recommended tools and libraries
3. Include proper error handling
4. Be well-documented with comments
5. Save results to appropriate files
6. Be runnable as a standalone script
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

# CODE REQUIREMENTS #
1. Import all necessary libraries at the top
2. Load data from the specified paths
3. Implement each step from the plan
4. Save outputs to files in the restore directory: {restore_dir}
5. Print progress messages for debugging
6. Handle errors gracefully

# RESPONSE FORMAT #
```python
# {phase_name} Implementation
# Generated code following the plan

import pandas as pd
import numpy as np
# ... other imports

# [Your implementation here]

if __name__ == "__main__":
    # Main execution
    pass
```
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
The code execution failed. Analyze the error and provide a fix.

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

# DEBUGGING STEPS #
1. Identify the exact line and cause of the error
2. Check for common issues:
   - Missing imports
   - Incorrect variable names
   - Type mismatches
   - File path issues
   - Logic errors
3. Provide corrected code

# RESPONSE FORMAT #
```python
# Corrected code
[Your fixed code here]
```
"""
