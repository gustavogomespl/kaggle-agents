"""Prompts for the Reviewer agent."""

PROMPT_REVIEWER_TASK = """# YOUR TASK #
Review and score the output from the {agent_role} agent.

Provide:
1. A score from 0-5 (where 5 is excellent, 3 is acceptable, 0 is poor)
2. Detailed analysis of strengths and weaknesses
3. Specific, actionable suggestions for improvement

Be constructive but rigorous in your evaluation.
"""

PROMPT_REVIEWER = """# REVIEW TASK #
You are reviewing the work of the "{agent_role}" agent for the "{phase_name}" phase.

## Agent's Task ##
{task}

## Agent's Input ##
{input}

## Agent's Output ##
{output}

## Background Context ##
{background}

# EVALUATION CRITERIA #

For Planner:
- Clarity and completeness of steps
- Alignment with competition goals
- Feasibility of implementation
- Appropriate tool selection
- Logical flow

For Developer:
- Code correctness and functionality
- Follows the plan specifications
- Proper error handling
- Code quality and documentation
- Efficiency and best practices

# SCORING SCALE #
5 - Excellent: Exceeds expectations, production-ready
4 - Good: Meets expectations with minor issues
3 - Acceptable: Meets minimum requirements, needs refinement
2 - Needs Work: Significant issues, requires revision
1 - Poor: Major problems, fundamental issues
0 - Unacceptable: Completely fails to meet requirements

# RESPONSE FORMAT #
```json
{{
  "agent": "{agent_role}",
  "score": 4,
  "analysis": {{
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "specific_issues": ["issue1", "issue2"]
  }},
  "suggestion": "Detailed suggestion for improvement. Be specific about what needs to change and how to improve it.",
  "requires_revision": false
}}
```
"""

PROMPT_MULTI_AGENT_REVIEW = """# MULTI-AGENT REVIEW TASK #
Review the outputs from multiple agents in the "{phase_name}" phase.

## Agents to Review ##
{agents_list}

For each agent, provide:
1. Individual score (0-5)
2. Analysis of their contribution
3. Suggestions for improvement
4. How their work integrates with others

## Phase Results ##
{phase_results}

# RESPONSE FORMAT #
```json
{{
  "phase": "{phase_name}",
  "overall_assessment": "Summary of phase execution",
  "agent_reviews": {{
    "planner": {{
      "score": 4,
      "analysis": "Detailed analysis",
      "suggestion": "Specific improvements"
    }},
    "developer": {{
      "score": 3,
      "analysis": "Detailed analysis",
      "suggestion": "Specific improvements"
    }}
  }},
  "phase_score": 3.5,
  "should_proceed": true,
  "next_steps": ["recommendation1", "recommendation2"]
}}
```
"""
