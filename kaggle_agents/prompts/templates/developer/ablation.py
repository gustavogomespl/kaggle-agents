"""
Ablation study prompt templates for the Developer Agent.

Contains prompts used for ablation analysis and improvement planning.
"""

# ==================== Ablation Study Prompts ====================

ABLATION_STUDY_PROMPT = """Analyze the impact of component changes through ablation study.

## Baseline Code
```python
{baseline_code}
```

## Modified Code
```python
{modified_code}
```

## Component Being Tested
{component_name}

Compare baseline vs modified performance. Return analysis in JSON format:
{{"component": "{component_name}", "baseline_score": float, "modified_score": float, "delta": float, "recommendation": "keep|remove|modify"}}"""


ABLATION_STUDY_SEQUENTIAL_PROMPT = """Perform sequential ablation study.

## Current Best Code
```python
{current_code}
```

## Components to Test
{components}

Test each component's impact sequentially. Return results for each."""


SUMMARIZE_ABLATION_PROMPT = """Summarize ablation study results.

## Results
{results}

Provide:
1. Most impactful components (positive delta)
2. Harmful components (negative delta)
3. Recommended final configuration"""


EXTRACT_IMPROVEMENT_PLAN_PROMPT = """Extract improvement plan from ablation results.

## Ablation Results
{results}

## Current Score
{current_score}

Create prioritized list of improvements based on ablation findings."""


EXTRACT_IMPROVEMENT_PLAN_SEQUENTIAL_PROMPT = """Extract sequential improvement plan.

## Sequential Results
{results}

## Target Score
{target_score}

Create ordered plan to reach target score."""


PLAN_REFINEMENT_PROMPT = """Refine improvement plan based on actual results.

## Original Plan
{original_plan}

## Actual Results
{actual_results}

## Gap Analysis
{gap_analysis}

Update plan based on what worked and what didn't."""


IMPLEMENT_PLAN_PROMPT = """Implement the improvement plan.

## Current Code
```python
{current_code}
```

## Improvement Plan
{plan}

## Priority
{priority}

Generate improved code implementing the plan."""
