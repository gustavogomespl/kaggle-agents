"""
Prompt templates for the Planner Agent.

These templates guide the agent in creating ablation plans
for systematic improvement of Kaggle solutions.
"""

# Base system prompt for the planner
PLANNER_SYSTEM_PROMPT = """# Introduction
You are a Kaggle Grandmaster with 10+ years of competition experience and expert at Ablation Studies for Machine Learning competitions.

Your role is to create FOCUSED, HIGH-IMPACT ablation plans that systematically identify and test only the most
promising components of a machine learning solution. You prioritize QUALITY over QUANTITY.

Your Ablation-Driven Optimization Strategy:
1. Analyze SOTA solutions to identify what actually wins competitions
2. Identify 3-5 HIGH-IMPACT components only (estimated impact >0.10)
3. Ensure diversity: different models (LightGBM, XGBoost, CatBoost) for ensembling
4. Prioritize components by ROI (impact / execution time / model cost)
5. Focus on proven winners: proper feature engineering, class imbalance handling, stacking ensembles
6. Treat each component as a bandit arm: exploit top-performing arms from prior iterations, explore 1 new idea only if capacity remains

Your plans should be:
- FOCUSED: Only 3-5 components total (quality over quantity)
- DIVERSE: At least 2 different models + optional preprocessing/ensemble
- HIGH-IMPACT: Each component estimated >10% improvement (0.10+ on 0-1 scale)
- ACTIONABLE: Clear, specific implementation details
- PROVEN: Based on what works in SOTA Kaggle solutions

CRITICAL RULES:
- NEVER create more than 5 components (prefer 3-4 when refining)
- ALWAYS include at least 2 model components (for ensemble diversity)
- ALWAYS prioritize components with estimated_impact >= 0.10
- PREFER proven techniques over experimental ideas; drop redundant variants
- USE reward signals from prior CV/LB scores: keep top-2 arms, replace lowest ROI arm with a new variant only if needed
- CONTROL cost: reserve expensive models for planner/critic phases; choose cheaper-but-solid models for bulk developer runs
- ENSURE each component is significantly different from others
"""

# Template for creating initial ablation plan
CREATE_ABLATION_PLAN_PROMPT = """Given the following competition information and SOTA solutions, create an ablation plan.

## Competition Information
{competition_info}

## Domain
{domain}

## SOTA Solutions Summary
{sota_summary}

## CRITICAL COMPONENT TYPE RULES

### preprocessing
- Data cleaning, missing value handling, basic scaling/encoding
- **NO MODEL TRAINING** - only prepare data
- Examples: StandardScaler, SimpleImputer, basic feature selection

### feature_engineering
- Create NEW features from existing ones
- **NO MODEL TRAINING** - only create features
- Examples: polynomial features, interaction terms, domain-specific features

### model
- **MUST TRAIN A MODEL and GENERATE PREDICTIONS**
- **MUST CREATE submission.csv**
- Use simple, fast models (LightGBM, XGBoost, RandomForest)
- Examples: LGBMClassifier with simple hyperparameters

### ensemble
- Combine predictions from multiple models
- Create weighted average or stacking ensemble

## Your Task
Create a list of 3-5 HIGH-QUALITY ablation components. **REQUIREMENTS:**
- **QUALITY OVER QUANTITY** - Focus on most impactful components only
- **AT LEAST 2 components MUST be type "model"** (to generate diverse predictions for ensembling)
- Include: 0-1 preprocessing + 2-3 models + 0-1 ensemble
- Prioritize ONLY high-impact components (>0.10 estimated impact)
- Each component must be significantly different from others

For each component provide:

1. **Name**: Short descriptive name (e.g., "xgboost_baseline", "missing_value_imputation")
2. **Type**: One of [feature_engineering, model, preprocessing, ensemble]
3. **Description**: What this component does (be specific)
4. **Estimated Impact**: Float 0-1 (e.g., 0.15 = 15% expected improvement)
5. **Rationale**: Why you think this will help based on SOTA
6. **Code Outline**: Brief pseudocode or description

## Output Format
Return ONLY a valid JSON list (no markdown, no explanation). Example for a tabular regression competition:

```json
[
  {{
    "name": "advanced_feature_engineering",
    "component_type": "feature_engineering",
    "description": "Create polynomial features (degree 2), feature interactions (ratio, diff, product), statistical transformations (log, sqrt), and target encoding for categorical features",
    "estimated_impact": 0.15,
    "rationale": "SOTA solutions show comprehensive feature engineering improves RMSE by 10-20%. Polynomial features capture non-linear relationships, target encoding handles high-cardinality categoricals effectively.",
    "code_outline": "Use PolynomialFeatures(degree=2, interaction_only=True), create ratio/diff/product features, apply log/sqrt transforms, use TargetEncoder with smoothing"
  }},
  {{
    "name": "lightgbm_tuned",
    "component_type": "model",
    "description": "LightGBM regression with tuned hyperparameters: n_estimators=2000, max_depth=8, learning_rate=0.03, num_leaves=63",
    "estimated_impact": 0.20,
    "rationale": "LightGBM consistently wins tabular competitions. Deeper trees (depth=8) and more leaves capture complex patterns. Lower learning rate with early stopping prevents overfitting.",
    "code_outline": "LGBMRegressor with 5-fold CV, early_stopping_rounds=100, eval_metric=rmse"
  }},
  {{
    "name": "xgboost_tuned",
    "component_type": "model",
    "description": "XGBoost regression with tuned hyperparameters: n_estimators=2000, max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8",
    "estimated_impact": 0.18,
    "rationale": "XGBoost provides different regularization than LightGBM, enabling better ensemble diversity. Mid-depth trees with subsampling prevent overfitting on tabular data.",
    "code_outline": "XGBRegressor with 5-fold CV, early_stopping_rounds=50, eval_metric=rmse"
  }},
  {{
    "name": "catboost_tuned",
    "component_type": "model",
    "description": "CatBoost regression with native categorical handling: iterations=2000, depth=7, learning_rate=0.03",
    "estimated_impact": 0.17,
    "rationale": "CatBoost handles categorical features natively without encoding, often outperforms other GBDTs. Adds diversity to ensemble.",
    "code_outline": "CatBoostRegressor with cat_features parameter, 5-fold CV, early_stopping_rounds=50"
  }},
  {{
    "name": "stacking_ensemble",
    "component_type": "ensemble",
    "description": "Stack LightGBM, XGBoost, and CatBoost predictions using Ridge regression as meta-learner with 5-fold out-of-fold predictions",
    "estimated_impact": 0.12,
    "rationale": "Stacking combines diverse models and typically improves RMSE by 5-10%. Ridge meta-learner prevents overfitting to base model predictions.",
    "code_outline": "StackingRegressor with base_estimators=[lgb, xgb, catboost], final_estimator=Ridge(alpha=10), cv=5"
  }}
]
```

**IMPORTANT**: Return ONLY the JSON array, nothing else. No markdown code blocks, no explanations.
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
5. Use bandit-style selection: top-2 previous winners stay, worst-performing arm gets replaced by 1 new idea only if ROI is low
6. Cap at 3-4 total components unless diversity requires 5; avoid duplicate model variants

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
    "image_to_image": """
For image-to-image competitions (denoising, super-resolution, style transfer, inpainting):

CRITICAL: These are PIXEL-LEVEL prediction tasks, NOT image classification!

## Architecture priorities:
- U-Net with skip connections (best for denoising, segmentation)
- Residual autoencoders (good for learning subtle transformations)
- DnCNN (denoising-specific CNN with residual learning)
- Fully Convolutional Networks (FCN) for dense prediction

## Submission format (CRITICAL - READ CAREFULLY):
- Output is NOT one prediction per image
- Output is ONE PREDICTION PER PIXEL
- Sample submission typically has MILLIONS of rows (one per pixel across all test images)
- ID format is usually: '{image_id}_{row}_{col}' or '{image_id}_{pixel_index}'
- ALWAYS read sample_submission.csv to understand exact format

## Model output requirements:
- Must output FULL IMAGE with same spatial dimensions as input (HxW or HxWxC)
- Then FLATTEN to pixel-level format for submission CSV
- Example: 420x540 image = 226,800 rows in submission per image

## Flattening code pattern:
```python
submission_rows = []
for img_path in test_images:
    img_id = img_path.stem
    pred = model(preprocess(img))  # Output: HxW
    H, W = pred.shape
    for row in range(H):
        for col in range(W):
            pixel_id = f"{img_id}_{row+1}_{col+1}"  # 1-indexed
            submission_rows.append({"id": pixel_id, "value": pred[row, col]})
```

## DO NOT USE:
- Image classifiers (EfficientNet, ResNet with FC classification head)
- Single-value regression models
- Any architecture that outputs one value per image
- Global average pooling followed by dense layers

## Training approach:
- Input: degraded/noisy image
- Target: clean/original image
- Loss: MSE, L1, or perceptual loss (VGG feature loss)
- Use paired training data (noisy -> clean pairs)
""",
    "image_segmentation": """
For image segmentation competitions:

CRITICAL: These require PIXEL-WISE classification/regression!

## Architecture priorities:
- U-Net (standard for medical and general segmentation)
- DeepLabV3+ (for semantic segmentation)
- Mask R-CNN (for instance segmentation)
- HRNet (maintains high-resolution representations)

## Submission format considerations:
- Check if RLE (Run-Length Encoding) is required
- Or pixel-level format (one row per pixel)
- Some competitions use mask images directly

## Key techniques:
- Data augmentation: rotation, flip, elastic deformation
- Multi-scale training
- Test-time augmentation (TTA)
- Post-processing: CRF, morphological operations
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
