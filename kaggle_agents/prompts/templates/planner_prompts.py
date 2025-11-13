"""
Prompt templates for the Planner Agent.

These templates guide the agent in creating ablation plans
for systematic improvement of Kaggle solutions.
"""

# Base system prompt for the planner
PLANNER_SYSTEM_PROMPT = """You are an expert Machine Learning Engineer specializing in Kaggle competitions.

Your role is to create ABLATION PLANS that systematically identify and test high-impact components
of a machine learning solution. You follow the "Ablation-Driven Optimization" strategy:

1. Identify independent, testable components (features, models, preprocessing, ensembles)
2. Estimate the impact of each component (0-1 scale, where 0.1 = 10% improvement)
3. Prioritize components by estimated impact
4. Create a plan that focuses on high-impact components first

Your plans should be:
- SPECIFIC: Each component has clear code boundaries
- TESTABLE: Components can be independently enabled/disabled
- IMPACTFUL: Focus on components likely to improve scores
- DIVERSE: Cover different aspect types (features, models, etc.)
"""

# Template for creating initial ablation plan
CREATE_ABLATION_PLAN_PROMPT = """Given the following competition information and SOTA solutions, create an ablation plan.

## Competition Information
{competition_info}

## Domain
{domain}

## SOTA Solutions Summary
{sota_summary}

## Your Task
Create a list of 5-10 ablation components. For each component:

1. **Name**: Short descriptive name
2. **Type**: One of [feature_engineering, model, preprocessing, ensemble]
3. **Description**: What this component does
4. **Estimated Impact**: Float 0-1 (e.g., 0.15 = 15% expected improvement)
5. **Rationale**: Why you think this will help
6. **Code Outline**: Brief pseudocode or description

## Output Format
Return a JSON list of components:

```json
[
  {{
    "name": "target_encoding",
    "component_type": "feature_engineering",
    "description": "Apply target encoding to high-cardinality categorical features",
    "estimated_impact": 0.12,
    "rationale": "SOTA solutions show target encoding improved scores by 8-15%",
    "code_outline": "Use category_encoders.TargetEncoder on ['col1', 'col2']"
  }},
  ...
]
```

Focus on components with high estimated impact (>0.05) and diversity across types.
"""

# Template for refining ablation plan based on results
REFINE_ABLATION_PLAN_PROMPT = """You previously created an ablation plan. Now refine it based on actual results.

## Previous Plan
{previous_plan}

## Test Results
{test_results}

## Current Best Score
{current_score}

## Your Task
Analyze what worked and what didn't. Create a NEW refined plan that:

1. Keeps components that showed positive impact
2. Removes or modifies components with no/negative impact
3. Adds NEW components inspired by successful patterns
4. Re-estimates impacts based on actual data

Focus on:
- Components that actually moved the score
- Combinations of successful components
- New ideas not yet tested

Return the refined plan in the same JSON format.
"""

# Template for explaining a component
EXPLAIN_COMPONENT_PROMPT = """Explain the following ML component in detail:

## Component
{component}

## Context
Competition: {competition_name}
Domain: {domain}
Current Approach: {current_approach}

## Your Task
Provide:

1. **Detailed Description**: How it works technically
2. **Implementation Steps**: Step-by-step guide
3. **Expected Impact**: Why it helps (with examples)
4. **Potential Risks**: What could go wrong
5. **Code Example**: Minimal working code

Be specific and actionable.
"""

# Template for SOTA analysis
ANALYZE_SOTA_PROMPT = """Analyze the following SOTA solutions and extract key patterns.

## SOTA Solutions
{sota_solutions}

## Your Task
Identify:

1. **Common Models**: Which models appear most frequently?
2. **Feature Engineering Patterns**: What feature techniques are popular?
3. **Ensemble Strategies**: How do winners combine models?
4. **Unique Tricks**: Any novel approaches?
5. **Success Factors**: What separates top solutions?

Return analysis as structured JSON:

```json
{{
  "common_models": ["XGBoost", "LightGBM"],
  "feature_patterns": ["Target encoding", "Polynomial features"],
  "ensemble_strategies": ["Stacking with linear meta-learner"],
  "unique_tricks": ["Feature interaction mining"],
  "success_factors": ["Deep feature engineering", "Careful CV strategy"]
}}
```
"""

# Template for component prioritization
PRIORITIZE_COMPONENTS_PROMPT = """Given these potential components, prioritize them by expected ROI.

## Components
{components}

## Constraints
- Time budget: {time_budget} hours
- Compute budget: {compute_budget}
- Current score: {current_score}
- Target score: {target_score}

## Your Task
Rank components by ROI considering:
1. Estimated impact / implementation time
2. Risk (probability of success)
3. Dependencies (what must be done first)
4. Compute cost

Return prioritized list with scores:

```json
[
  {{
    "component": "target_encoding",
    "priority_rank": 1,
    "roi_score": 0.85,
    "implementation_time_hours": 2,
    "risk_level": "low",
    "dependencies": []
  }},
  ...
]
```

Order by priority_rank (1 = highest priority).
"""

# Domain-specific prompts
DOMAIN_SPECIFIC_PROMPTS = {
    "tabular": """
For tabular competitions, prioritize:
- Feature engineering (target encoding, feature interactions)
- Gradient boosting models (XGBoost, LightGBM, CatBoost)
- Careful cross-validation (stratified, group-based)
- Feature selection and importance analysis
- Ensemble methods (stacking, blending)
""",
    "computer_vision": """
For computer vision competitions, prioritize:
- Transfer learning (pre-trained models like ResNet, EfficientNet)
- Data augmentation strategies (rotation, crop, color jitter)
- Test-time augmentation (TTA)
- Ensemble of different architectures
- Image preprocessing (normalization, resizing strategies)
""",
    "nlp": """
For NLP competitions, prioritize:
- Pre-trained transformers (BERT, RoBERTa, GPT)
- Fine-tuning strategies (learning rate, epochs)
- Data augmentation (back-translation, synonym replacement)
- Ensemble of different models
- Text preprocessing (cleaning, tokenization)
""",
    "time_series": """
For time series competitions, prioritize:
- Lag features and rolling statistics
- Seasonality and trend decomposition
- Time-based cross-validation
- Forecasting models (ARIMA, Prophet, LSTM)
- Feature engineering for temporal patterns
""",
}


def get_domain_guidance(domain: str) -> str:
    """
    Get domain-specific guidance for the planner.

    Args:
        domain: Domain type

    Returns:
        Domain-specific prompt guidance
    """
    return DOMAIN_SPECIFIC_PROMPTS.get(domain, "")
