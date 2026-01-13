"""Extended strategies for tabular competitions."""

EXTENDED_STRATEGIES_TABULAR = {
    "feature_engineering_heavy": {
        "name": "feature_engineering_heavy",
        "prompt_modifier": """
Prioritize extensive feature engineering before modeling:
- Create derived features: interactions, ratios, aggregations
- Apply target encoding with proper CV to avoid leakage
- Generate temporal features if applicable (lags, rolling stats)
- Use clustering-based features (KMeans cluster assignments)
- Save engineered features for reuse by all models
""",
        "model_preference": ["lightgbm", "xgboost"],
        "component_emphasis": ["feature_engineering"],
    },
    "neural_exploration": {
        "name": "neural_exploration",
        "prompt_modifier": """
Explore neural network approaches for tabular data:
- TabNet for interpretable neural networks
- MLP with embeddings for categorical features
- Neural network + gradient boosting ensemble
- Use proper regularization (dropout, weight decay)
""",
        "model_preference": ["tabnet", "mlp", "neural_ensemble"],
        "component_emphasis": ["model"],
    },
    "hyperparameter_variant": {
        "name": "hyperparameter_variant",
        "prompt_modifier": """
Explore hyperparameter variations of successful models:
- If LightGBM worked, try different learning_rate (0.01, 0.05, 0.1)
- Vary max_depth (4, 6, 8, 10) and num_leaves
- Test different regularization (lambda_l1, lambda_l2)
- Try different n_estimators (500, 1000, 2000)
""",
        "model_preference": ["lightgbm", "xgboost", "catboost"],
        "component_emphasis": ["model"],
        "inherit_from_best": True,
    },
    "stacking_ensemble": {
        "name": "stacking_ensemble",
        "prompt_modifier": """
Focus on advanced stacking ensembles:
- Create diverse base models (GBM, RF, Linear)
- Use OOF predictions as meta-features
- Add second-level meta-learner (Ridge, LightGBM)
- Ensure proper CV alignment to avoid leakage
""",
        "model_preference": ["stacking", "blending"],
        "component_emphasis": ["ensemble"],
    },
}
