"""Prompts for the Planner agent."""

PROMPT_PLANNER_TASK = """# YOUR TASK #
Create a comprehensive plan for the "{phase_name}" phase of the Kaggle competition workflow.

Your plan should:
1. Break down the phase into clear, actionable steps
2. Consider the competition type and evaluation metric
3. Leverage previous phase results
4. Utilize available tools and libraries
5. Be specific and implementable

Focus on practical steps that can be directly translated into code.
"""

PROMPT_PLANNER = """# CONTEXT #
You are planning the "{phase_name}" phase in a Kaggle competition workflow.

## Workflow Phases for Context ##
{phases_in_context}

## Current State ##
{state_info}

## User Rules ##
{user_rules}

## Background Information ##
{background_info}

{task}

Please provide an initial plan considering:
- The competition objective and metric
- Data characteristics from background
- Best practices for this phase
- Logical progression of steps
"""

PROMPT_PLANNER_TOOLS = """# AVAILABLE TOOLS #

The following pre-defined ML tools are available for this phase:

{tools}

Tool Names: {tool_names}

# TASK #
Refine your plan to incorporate these tools where appropriate.

For each tool you plan to use:
1. Explain why it's suitable for the task
2. Specify the parameters you'll use
3. Describe expected outputs

Update your plan with these tool specifications.
"""

PROMPT_PLANNER_REORGANIZE_IN_MARKDOWN = """# TASK #
Please reorganize your plan into a clean, structured Markdown format.

Use this structure:

# RESPONSE FORMAT #
```markdown
# {Phase Name} Plan

## Objective
[Clear statement of what this phase aims to achieve]

## Approach
[High-level strategy]

## Detailed Steps

### Step 1: [Step Name]
- **Description**: [What this step does]
- **Method**: [How it will be done]
- **Tools/Libraries**: [What will be used]
- **Expected Output**: [What this step produces]

### Step 2: [Step Name]
[Continue for all steps]

## Success Criteria
[How to know the phase was successful]

## Notes
[Any important considerations or warnings]
```
"""

PROMPT_PLANNER_REORGANIZE_IN_JSON = """# TASK #
Convert your plan into a structured JSON format for programmatic processing.

# RESPONSE FORMAT #
```json
{
  "final_answer": {
    "phase": "Phase Name",
    "objective": "Clear objective statement",
    "approach": "High-level approach description",
    "steps": [
      {
        "step_number": 1,
        "name": "Step Name",
        "description": "What this step does",
        "method": "How it will be done",
        "tools": ["tool1", "library1"],
        "expected_output": "What this produces"
      }
    ],
    "success_criteria": ["criterion1", "criterion2"],
    "notes": ["note1", "note2"]
  }
}
```
"""
