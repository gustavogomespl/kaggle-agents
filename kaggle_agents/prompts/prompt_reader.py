"""Prompts for the Reader agent."""

PROMPT_READER_TASK = """# YOUR TASK #
Extract and summarize key information about the Kaggle competition.

Focus on:
1. Competition objective and context
2. Problem type (classification, regression, etc.)
3. Evaluation metric
4. Data description
5. Important rules or constraints
6. Success criteria
"""

PROMPT_READER = """# COMPETITION INFORMATION EXTRACTION #

## Competition Overview ##
{overview}

## Data Description ##
{data_description}

# TASK #
Analyze the competition information and extract structured insights.

# RESPONSE FORMAT #
```markdown
# Competition Background

## Title
[Competition title]

## Objective
[What the competition aims to solve]

## Problem Type
[Classification / Regression / Other]

## Evaluation Metric
[Primary metric used for scoring]

## Data Overview
### Training Data
- Features: [number and types]
- Target: [target variable(s)]
- Size: [number of samples]

### Test Data
- Features: [what's provided]
- Size: [number of samples]

## Key Constraints
[Any important rules or limitations]

## Success Factors
[What makes a good solution]

## Notes
[Any other important observations]
```
"""

PROMPT_EXTRACT_METRIC = """# TASK #
Extract the evaluation metric from the competition information.

# COMPETITION INFORMATION #
{competition_info}

# RESPONSE FORMAT #
```markdown
# Evaluation Metric

**Metric Name**: [Name of metric, e.g., "Accuracy", "RMSE", "F1-Score"]

**Description**: [How the metric is calculated]

**Optimization Goal**: [Maximize or Minimize]

**Details**: [Any specific notes about how it's used in this competition]
```
"""
