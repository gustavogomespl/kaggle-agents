"""
Cross-validation and OOF (Out-of-Fold) instruction builders.
"""


def build_cv_instructions(working_dir: str, component_name: str) -> list[str]:
    """Build cross-validation instructions."""
    return [
        "\n🔄 CONSISTENT CROSS-VALIDATION (CRITICAL):",
        f"  - Check if '{working_dir}/folds.csv' exists.",
        "  - IF EXISTS: Load it and use the 'fold' column for splitting.",
        "    ```python",
        "    folds = pd.read_csv('folds.csv')",
        "    # Assuming X is aligned with folds (reset_index if needed)",
        "    for fold in sorted(folds['fold'].unique()):",
        "        val_idx = folds[folds['fold'] == fold].index",
        "        train_idx = folds[folds['fold'] != fold].index",
        "        # ... train/val split ...",
        "    ```",
        "  - IF NOT EXISTS: Use StratifiedKFold(n_splits=int(os.getenv('KAGGLE_AGENTS_CV_FOLDS','5')), shuffle=True, random_state=42)",
        f"  - CRITICAL: MUST save Out-of-Fold (OOF) predictions during CV to models/oof_{component_name}.npy",
        "  - OOF predictions enable proper stacking ensemble (meta-model trained on OOF)",
        "  - MUST print 'Final Validation Performance: {score}'",
        "  - If metric is NaN/inf, replace with 0.0 before printing Final Validation Performance",
        "  - MUST handle class imbalance with class_weight='balanced'",
    ]


def build_stacking_oof_instructions(working_dir: str, component_name: str) -> list[str]:
    """Build stacking/OOF instructions."""
    return [
        "\nSTACKING & OOF REQUIREMENTS (CRITICAL):",
        "  1. Initialize `oof_preds` array of zeros with length of train set.",
        "  2. Initialize `test_preds` array of zeros with length of test set.",
        "  3. During CV loop:",
        "     - Fill `oof_preds[val_idx]` with predictions for validation fold.",
        "     - Predict on test set and accumulate:",
        "       - Binary: `test_preds += model.predict_proba(X_test)[:, 1] / n_folds`",
        "       - Multi-class: `test_preds += model.predict_proba(X_test) / n_folds`",
        "       - Regression: `test_preds += model.predict(X_test) / n_folds`",
        f"  4. Save OOF predictions: `np.save(str(Path('{working_dir}') / 'models' / 'oof_{component_name}.npy'), oof_preds)`",
        f"  5. Save Test predictions: `np.save(str(Path('{working_dir}') / 'models' / 'test_{component_name}.npy'), test_preds)`",
        "  6. Ensemble will ONLY run if BOTH oof_{name}.npy AND test_{name}.npy exist for at least 2 models.",
        "  7. This enables the Ensemble Agent to use Stacking later.",
    ]
