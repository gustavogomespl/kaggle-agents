"""Base prompts shared across agents."""

AGENT_ROLE_TEMPLATE = """# YOUR ROLE #
You are an AI agent with the role: {agent_role}

"""

PROMPT_DATA_PREVIEW = """# TASK #
Please provide a concise preview of the data including:
1. Target variable(s)
2. Key features and their types
3. Data shape and size
4. Notable characteristics (missing values, imbalances, etc.)

# DATA #
{data}

# RESPONSE FORMAT #
```markdown
# Data Preview

## Target Variable
[Target variable name and description]

## Features
[List key features with types]

## Data Characteristics
- Shape: [rows x columns]
- Missing values: [summary]
- Notable patterns: [any observations]
```
"""

PROMPT_FEATURE_INFO = """# FEATURE INFORMATION #

Target Variable: {target_variable}

Features Before: {features_before}

Features After: {features_after}

Please analyze the feature changes.
"""

PROMPT_EACH_EXPERIENCE_WITH_SUGGESTION = """
## Experience #{index} ##
Previous Result:
{experience}

Reviewer Suggestion:
{suggestion}

Score: {score}/5

"""

PROMPT_REORGANIZE_JSON = """# TASK #
Please reorganize the information you provided into a structured JSON format.

Extract the key information and format it as valid JSON that can be parsed.

# RESPONSE FORMAT #
```json
{{
  "final_answer": {{
    [Your structured data here]
  }}
}}
```
"""

PROMPT_REORGANIZE_EXTRACT_TOOLS = """# TASK #
Extract the tool names from the following information and format as JSON.

# INFORMATION #
{information}

# RESPONSE FORMAT #
```json
{{
  "tool_names": ["tool1", "tool2", "tool3"]
}}
```
"""
