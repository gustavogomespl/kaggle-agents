"""
Feature engineering instruction builder.
"""


def build_feature_engineering_instructions() -> list[str]:
    """Build feature engineering instructions."""
    return [
        "\n🔧 FEATURE ENGINEERING REQUIREMENTS:",
        "  - Create NEW features from existing ones",
        "  - IMPLEMENT SOTA TECHNIQUES:",
        "    - Target Encoding: MUST be done inside Cross-Validation (fit on train folds, transform val fold) to prevent leakage.",
        "    - Frequency Encoding: Map categorical features to their frequency/count.",
        "    - Aggregations: Mean/Count of numeric features grouped by categorical features.",
        "  - Save engineered features to file for model components",
        "  - NO model training in this component",
        "  - Print feature importance or correlation metrics",
        "\nFEATURE SELECTION (CRITICAL):",
        "  - After creating new features, perform selection to remove noise:",
        "  1. Train a quick LightGBM/XGBoost on the new feature set.",
        "  2. Calculate feature importance (gain/split).",
        "  3. Drop features with 0 importance or very low importance (< 1e-4).",
        "  4. Save ONLY the selected features to 'train_engineered.csv' and 'test_engineered.csv'.",
        "  5. Print list of dropped features.",
    ]
