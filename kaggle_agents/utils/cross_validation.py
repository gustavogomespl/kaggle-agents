"""
Cross-validation utilities for consistent evaluation.
"""

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def generate_folds(
    train_path: str,
    target_col: str,
    output_path: str,
    n_folds: int = 5,
    seed: int = 42
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

    # Create 'fold' column
    df['fold'] = -1

    # Determine problem type for splitting strategy
    # Simple heuristic: if target has few unique values -> classification
    n_unique = df[target_col].nunique()
    is_classification = n_unique < 20

    if is_classification:
        print(f"      Detected classification (unique targets: {n_unique}) -> Using StratifiedKFold")
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        y = df[target_col]
        for fold, (_train_idx, val_idx) in enumerate(kf.split(df, y)):
            df.loc[val_idx, 'fold'] = fold
    else:
        print("      Detected regression -> Using KFold")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = fold

    # Save only index (if needed) or full dataframe?
    # Saving full dataframe with 'fold' column is safest for alignment
    # But to save space/time, maybe just save the fold column?
    # Actually, user requested "folds.csv". Let's save the whole dataframe with the fold column
    # so agents can just load this instead of train.csv if they want, OR join it.
    # BETTER: Save just the 'fold' column aligned with original index, or the full file.
    # Saving full file is easiest for agents to use: "Load folds.csv instead of train.csv"

    df.to_csv(output_path, index=False)
    print(f"      ✅ Saved fixed folds to: {output_path}")

    return output_path
