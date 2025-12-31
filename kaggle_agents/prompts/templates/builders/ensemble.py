"""
Ensemble instruction builder.
"""


def build_ensemble_instructions(target_col: str = "target") -> list[str]:
    """Build ensemble instructions."""
    return [
        "\nENSEMBLE REQUIREMENTS:",
        "  - Combine predictions from multiple models",
        "  - PREFERRED STRATEGY: Stacking Ensemble (best performance)",
        "    - Load OOF predictions from models/oof_*.npy files",
        "    - Match test predictions by name: models/test_{name}.npy for each oof_{name}.npy",
        "    - DO NOT hardcode filenames like test_preds_mlp.npy; discover pairs dynamically",
        "    - Stack OOF predictions: oof_stack = np.column_stack([oof1, oof2, ...])",
        "    - Train meta-model (LogisticRegression/Ridge) on stacked OOF",
        "    - Load test predictions from each model and stack them",
        "    - Use meta-model to predict on stacked test predictions",
        "  - FALLBACK: Weighted average if OOF files missing",
        "    - Load submission files from each model",
        "    - Combine with weights: final = w1*pred1 + w2*pred2 + ...",
        "  - Generate final submission.csv",
        f"  - CRITICAL: Use target_col from dataset info when sample_submission has 2 columns (target_col='{target_col}' if available)",
        "  - CRITICAL: submission columns MUST match sample_submission columns and order",
        "    - If sample has 2 cols: fill columns[1] only",
        "    - If sample has >2 cols: fill ALL target columns in order (columns[1:])",
        "  - Print which models were used and their contribution/weights",
    ]
