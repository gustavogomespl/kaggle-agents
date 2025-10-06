"""Prompts for the Reviewer agent."""

PROMPT_REVIEWER_TASK = """# YOUR TASK #
Review and score the output from the {agent_role} agent.

Provide:
1. Score from 0-5 (5=excellent, 3=acceptable, 0=poor)
2. Brief analysis (2-3 bullet points)
3. Specific, actionable suggestion (1-2 sentences)

Be constructive and CONCISE.
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

**For Planner:**
- Clear, actionable steps
- Aligned with competition goals
- Implementable with available tools

**For Developer:**
- Code runs without errors
- Follows plan specifications
- Handles errors appropriately
- Produces required outputs

# SCORING SCALE #
5 - Excellent: Production-ready, exceeds expectations
4 - Good: Meets expectations, minor issues
3 - Acceptable: Minimum requirements met
2 - Needs Work: Significant issues
1 - Poor: Major problems
0 - Unacceptable: Fails completely

# RESPONSE FORMAT #
Return CONCISE JSON evaluation:

```json
{{
  "agent": "{agent_role}",
  "score": 4,
  "analysis": {{
    "strengths": ["Concise strength 1", "Concise strength 2"],
    "weaknesses": ["Concise weakness 1", "Concise weakness 2"],
    "specific_issues": ["Issue 1", "Issue 2"]
  }},
  "suggestion": "One concrete, actionable suggestion (1-2 sentences max).",
  "requires_revision": false
}}
```

⚠️ Keep analysis brief - focus on actionable insights, not lengthy explanations.
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
