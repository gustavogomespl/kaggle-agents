"""Prompts for the Planner agent."""

PROMPT_PLANNER_TASK = """# YOUR TASK #
Create a clear, actionable plan for the "{phase_name}" phase.

Your plan should:
1. Break down into 3-5 concrete, implementable steps
2. Consider competition type and evaluation metric
3. Leverage previous phase results
4. Be SPECIFIC - avoid vague statements
5. Be CODE-READY - steps should translate directly to code

Focus on PRACTICAL steps, not theoretical explanations.
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

# PLANNING GUIDELINES #
- Competition objective and metric
- Data characteristics from background
- Best practices for this phase
- Logical progression of steps
- KEEP IT CONCISE - 3-5 clear steps maximum

⚠️ Focus on WHAT to do, not lengthy explanations of WHY.
"""

PROMPT_PLANNER_TOOLS = """# AVAILABLE TOOLS #

The following pre-defined ML tools are available for this phase:

{tools}

Tool Names: {tool_names}

# TASK #
Refine your plan to incorporate these tools where appropriate.

For each tool you plan to use:
1. State which tool to use
2. Specify key parameters
3. Note expected output

Keep tool specifications BRIEF and ACTIONABLE.
"""

PROMPT_PLANNER_REORGANIZE_IN_MARKDOWN = """# TASK #
Reorganize your plan into a clean, structured Markdown format.

# RESPONSE FORMAT #
```markdown
# {Phase Name} Plan

## Objective
[Clear, one-sentence objective]

## Approach
[2-3 sentences on strategy]

## Steps

### 1. [Step Name]
- **Action**: [What to do]
- **Tools**: [What to use]
- **Output**: [What is produced]

### 2. [Step Name]
[Continue for 3-5 steps total]

## Success Criteria
- [Criterion 1]
- [Criterion 2]
```

⚠️ Keep it CONCISE - avoid lengthy explanations.
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
