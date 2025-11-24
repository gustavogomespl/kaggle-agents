"""
Cross-validation utilities for consistent evaluation.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def generate_folds(
    train_path: str, target_col: str, output_path: str, n_folds: int = 5, seed: int = 42
) -> str:
    """
    Generate fixed cross-validation folds and save to CSV.

    Args:
        train_path: Path to training data CSV
        target_col: Name of target column
        output_path: Path to save folds CSV
        n_folds: Number of folds
        seed: Random seed

    Returns:
        Path to saved folds file
    """
    print(f"   🔄 Generating {n_folds} fixed folds...")

    df = pd.read_csv(train_path)


    df["fold"] = -1



    n_unique = df[target_col].nunique()
    is_classification = n_unique < 20

    if is_classification:
        print(
            f"      Detected classification (unique targets: {n_unique}) -> Using StratifiedKFold"
        )
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        y = df[target_col]
        for fold, (train_idx, val_idx) in enumerate(kf.split(df, y)):
            df.loc[val_idx, "fold"] = fold
    else:
        print("      Detected regression -> Using KFold")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, "fold"] = fold









    df.to_csv(output_path, index=False)
    print(f"      ✅ Saved fixed folds to: {output_path}")

    return output_path
