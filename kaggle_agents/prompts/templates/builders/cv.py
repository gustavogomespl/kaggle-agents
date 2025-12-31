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
        "  - CRITICAL: Fit preprocessing/scalers INSIDE each CV fold (fit on X_train, transform X_val/X_test)",
        f"  - CRITICAL: MUST save Out-of-Fold (OOF) predictions during CV to models/oof_{component_name}.npy",
        "  - OOF predictions enable proper stacking ensemble (meta-model trained on OOF)",
        "  - MUST print 'Final Validation Performance: {score}'",
        "  - If metric is NaN/inf, replace with 0.0 before printing Final Validation Performance",
        "  - Multiclass log_loss: after clipping, renormalize rows to sum to 1",
        "  - If OOF rows are empty due to early stop, compute log_loss on rows with sum>0",
        "  - MUST handle class imbalance with class_weight='balanced'",
    ]


def build_stacking_oof_instructions(working_dir: str, component_name: str) -> list[str]:
    """Build stacking/OOF instructions."""
    return [
        "\nSTACKING & OOF REQUIREMENTS (CRITICAL):",
        "  1. Initialize `oof_preds` array of zeros with shape (n_train, n_classes) for multi-class.",
        "  2. Initialize `test_preds` array of zeros with shape (n_test, n_classes) for multi-class.",
        "  3. **CLASS ORDER ALIGNMENT (CRITICAL FOR ENSEMBLE - PREVENTS DEGRADATION)**:",
        "     - Read sample_submission.csv to get canonical class order:",
        "       ```python",
        "       sample_sub = pd.read_csv(sample_submission_path)",
        "       class_order = sample_sub.columns[1:].tolist()  # Canonical order from submission",
        "       ```",
        "     - After fitting LabelEncoder, compute reordering indices:",
        "       ```python",
        "       le_classes = le.classes_.tolist()  # Model's LabelEncoder order",
        "       reorder_idx = [le_classes.index(c) for c in class_order]  # Map to submission order",
        "       ```",
        "     - BEFORE saving, reorder predictions to match sample_submission:",
        "       ```python",
        "       oof_preds_aligned = oof_preds[:, reorder_idx]",
        "       test_preds_aligned = test_preds[:, reorder_idx]",
        "       ```",
        "     - Save canonical class order for validation:",
        f"       `np.save(str(Path('{working_dir}') / 'models' / 'class_order.npy'), class_order)`",
        "  4. During CV loop:",
        "     - Fill `oof_preds[val_idx]` with predictions for validation fold.",
        "     - Predict on test set and accumulate:",
        "       - Binary: `test_preds += model.predict_proba(X_test)[:, 1] / n_folds`",
        "       - Multi-class: `test_preds += model.predict_proba(X_test) / n_folds`",
        "       - Regression: `test_preds += model.predict(X_test) / n_folds`",
        f"  5. Save OOF predictions (AFTER reordering): `np.save(str(Path('{working_dir}') / 'models' / 'oof_{component_name}.npy'), oof_preds_aligned)`",
        f"  6. Save Test predictions (AFTER reordering): `np.save(str(Path('{working_dir}') / 'models' / 'test_{component_name}.npy'), test_preds_aligned)`",
        "  7. Ensemble will ONLY run if BOTH oof_{name}.npy AND test_{name}.npy exist for at least 2 models.",
        "  8. This enables the Ensemble Agent to use Stacking later.",
    ]
