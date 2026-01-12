"""
Text Normalization utilities for fast baseline and hybrid approaches.

This module implements the lookup-first strategy that handles 80%+ of tokens
with deterministic rules, reserving neural models only for ambiguous cases.

Key classes:
- LookupBaseline: Frequency-based lookup table for (class, before) -> after mappings
- HybridPipeline: Combines lookup baseline with neural model for ambiguous cases
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Deterministic semiotic classes (rule-based, no neural needed)
# These classes have near-deterministic mappings that can be handled by lookup/rules
DETERMINISTIC_CLASSES = {
    "PLAIN",      # Keep as-is
    "PUNCT",      # Keep as-is
    "VERBATIM",   # Keep as-is
    "LETTERS",    # Spell out: "ABC" -> "a b c"
    "ELECTRONIC", # URLs, emails - special formatting
    "TRANS",      # Transliteration - usually keep as-is
}

# Ambiguous classes that benefit from neural model
# These classes have context-dependent or multiple valid transformations
AMBIGUOUS_CLASSES = {
    "CARDINAL",   # 123 -> "one hundred twenty three" OR "one two three"
    "ORDINAL",    # 1st -> "first"
    "DATE",       # 1/2/2023 -> many possible formats
    "TIME",       # 10:30 -> "ten thirty" or "ten thirty a m"
    "MEASURE",    # 5kg -> "five kilograms" or "five k g"
    "MONEY",      # $5 -> "five dollars" or "five bucks"
    "DECIMAL",    # 3.14 -> "three point one four"
    "FRACTION",   # 1/2 -> "one half" or "a half"
    "TELEPHONE",  # Phone number formatting
    "ADDRESS",    # Address formatting
}

# Max steps guard for neural training to prevent timeout
DEFAULT_MAX_STEPS_FAST = 2000
DEFAULT_MAX_STEPS_FULL = 10000


class LookupBaseline:
    """
    Frequency-based lookup baseline for text normalization.

    For each (class, before) pair, stores the most frequent 'after' value
    from training data. Achieves 80%+ accuracy on deterministic classes.

    This is the first component in the hybrid pipeline and handles most tokens
    with zero inference cost.
    """

    def __init__(self):
        self.lookup: dict[tuple[str, str], str] = {}
        self.class_fallbacks: dict[str, str] = {}
        self.stats: dict[str, int] = defaultdict(int)

    def fit(
        self,
        df: pd.DataFrame,
        class_col: str = "class",
        before_col: str = "before",
        after_col: str = "after",
    ) -> "LookupBaseline":
        """
        Build lookup table from training data.

        Args:
            df: Training DataFrame with class, before, after columns
            class_col: Name of class column
            before_col: Name of input column
            after_col: Name of target column

        Returns:
            self for chaining
        """
        # Count (class, before) -> after frequencies using vectorized groupby
        # This is 50-100x faster than iterrows for large datasets
        counts = df.groupby(
            [class_col, before_col, after_col], as_index=False
        ).size()
        counts.columns = [class_col, before_col, after_col, 'count']

        # Get the most frequent 'after' for each (class, before) pair
        idx = counts.groupby([class_col, before_col])['count'].idxmax()
        best_mappings = counts.loc[idx]

        # Build lookup dictionary
        for _, row in best_mappings.iterrows():
            key = (str(row[class_col]), str(row[before_col]))
            self.lookup[key] = str(row[after_col])
            self.stats["lookup_entries"] += 1

        # Build class-level fallbacks
        self._build_class_fallbacks(df, class_col, before_col, after_col)

        print(f"[LookupBaseline] Built lookup with {len(self.lookup):,} entries")
        return self

    def _build_class_fallbacks(
        self,
        df: pd.DataFrame,
        class_col: str,
        before_col: str,
        after_col: str,
    ):
        """Build fallback rules per semiotic class for unseen tokens."""
        # PLAIN: keep as-is
        self.class_fallbacks["PLAIN"] = "<self>"

        # PUNCT: keep as-is
        self.class_fallbacks["PUNCT"] = "<self>"

        # VERBATIM: keep as-is
        self.class_fallbacks["VERBATIM"] = "<self>"

        # LETTERS: spell out (will be handled in predict)
        self.class_fallbacks["LETTERS"] = "<spell>"

        # ELECTRONIC: usually keep as-is
        self.class_fallbacks["ELECTRONIC"] = "<self>"

        # TRANS: keep as-is
        self.class_fallbacks["TRANS"] = "<self>"

        # For other classes, use most common transformation pattern
        for cls in df[class_col].unique():
            cls_str = str(cls)
            if cls_str in self.class_fallbacks:
                continue
            cls_df = df[df[class_col] == cls]
            if len(cls_df) > 0:
                # Most common 'after' value for this class
                most_common = cls_df[after_col].value_counts().idxmax()
                self.class_fallbacks[cls_str] = str(most_common)

    def predict(self, class_val: str, before_val: str) -> tuple[str, bool]:
        """
        Predict 'after' value for a (class, before) pair.

        Args:
            class_val: Semiotic class
            before_val: Input token

        Returns:
            Tuple of (prediction, is_confident)
            is_confident=False indicates fallback was used (may need neural refinement)
        """
        key = (str(class_val), str(before_val))

        # Try exact lookup first
        if key in self.lookup:
            self.stats["lookup_hits"] += 1
            return self.lookup[key], True

        # Try class-level fallback
        fallback = self.class_fallbacks.get(str(class_val))
        self.stats["fallback_used"] += 1

        if fallback == "<self>":
            # Deterministic: keep as-is
            return before_val, True
        elif fallback == "<spell>":
            # Spell out letters: "ABC" -> "a b c"
            spelled = " ".join(before_val.lower())
            return spelled, True
        elif fallback:
            # Used class-level fallback, may need neural refinement
            is_ambiguous = str(class_val) in AMBIGUOUS_CLASSES
            return fallback, not is_ambiguous
        else:
            # Unknown class, keep as-is
            return before_val, False

    def predict_batch(
        self,
        df: pd.DataFrame,
        class_col: str = "class",
        before_col: str = "before",
    ) -> pd.DataFrame:
        """
        Predict for entire DataFrame using vectorized operations.

        Args:
            df: DataFrame with class and before columns

        Returns:
            DataFrame with predictions and confidence flags
        """
        # Vectorized approach - 50-100x faster than iterrows
        class_str = df[class_col].astype(str)
        before_str = df[before_col].astype(str)

        # Create lookup keys as tuples
        keys = list(zip(class_str, before_str))

        # Vectorized lookup using Series.map()
        predictions = pd.Series(keys, index=df.index).map(self.lookup)

        # Track which had direct lookup hits
        is_lookup_hit = predictions.notna()
        self.stats["lookup_hits"] += int(is_lookup_hit.sum())

        # Get fallback values for misses
        fallback_values = class_str.map(self.class_fallbacks)
        needs_fallback = ~is_lookup_hit

        # Handle <self> fallback: keep as-is
        is_self_fallback = needs_fallback & (fallback_values == "<self>")

        # Handle <spell> fallback: spell out letters
        is_spell_fallback = needs_fallback & (fallback_values == "<spell>")
        spelled_out = before_str.apply(lambda x: " ".join(x.lower()))

        # Handle unknown class (fallback is NaN): keep as-is, mark not confident
        is_unknown_class = needs_fallback & fallback_values.isna()

        # Handle other known fallbacks (not <self>, not <spell>, not unknown)
        is_other_fallback = needs_fallback & ~is_self_fallback & ~is_spell_fallback & ~is_unknown_class

        # Apply fallbacks
        predictions = predictions.where(~is_self_fallback, before_str)
        predictions = predictions.where(~is_spell_fallback, spelled_out)
        predictions = predictions.where(~is_other_fallback, fallback_values)
        predictions = predictions.where(~is_unknown_class, before_str)  # Unknown class: keep as-is

        # Track fallback usage
        self.stats["fallback_used"] += int(needs_fallback.sum())

        # Calculate confidence:
        # - Lookup hit = confident
        # - <self> or <spell> fallback = confident (deterministic rules)
        # - Other fallback for non-ambiguous class = confident
        # - Other fallback for ambiguous class = not confident
        # - Unknown class = NOT confident (must go to neural)
        is_ambiguous_class = class_str.isin(AMBIGUOUS_CLASSES)
        is_confident = (
            is_lookup_hit |
            is_self_fallback |
            is_spell_fallback |
            (is_other_fallback & ~is_ambiguous_class)
        )
        # Unknown class is explicitly NOT confident (handled by not being in any of the above)

        # Mark as needing neural if not confident AND class is ambiguous
        needs_neural = ~is_confident & is_ambiguous_class

        result = df.copy()
        result["prediction"] = predictions
        result["is_confident"] = is_confident
        result["needs_neural"] = needs_neural

        return result

    def save(self, path: str | Path):
        """Save lookup table to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tuple keys to strings for JSON
        lookup_serializable = {f"{k[0]}|||{k[1]}": v for k, v in self.lookup.items()}

        with open(path, "w") as f:
            json.dump({
                "lookup": lookup_serializable,
                "class_fallbacks": self.class_fallbacks,
                "stats": dict(self.stats),
            }, f)

        print(f"[LookupBaseline] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LookupBaseline":
        """Load lookup table from file."""
        with open(path) as f:
            data = json.load(f)

        instance = cls()
        # Convert string keys back to tuples
        instance.lookup = {
            tuple(k.split("|||")): v for k, v in data["lookup"].items()
        }
        instance.class_fallbacks = data["class_fallbacks"]
        instance.stats = defaultdict(int, data.get("stats", {}))

        print(f"[LookupBaseline] Loaded {len(instance.lookup):,} entries from {path}")
        return instance

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about lookup usage."""
        return {
            "total_entries": len(self.lookup),
            "lookup_hits": self.stats.get("lookup_hits", 0),
            "fallback_used": self.stats.get("fallback_used", 0),
            "hit_rate": (
                self.stats.get("lookup_hits", 0) /
                max(1, self.stats.get("lookup_hits", 0) + self.stats.get("fallback_used", 0))
            ),
        }


def get_neural_training_config(
    n_ambiguous_samples: int,
    fast_mode: bool = True,
    timeout_s: int = 1800,
) -> dict[str, Any]:
    """
    Get training configuration for neural seq2seq model.

    Enforces max_steps guard to prevent runaway training.

    Args:
        n_ambiguous_samples: Number of samples needing neural prediction
        fast_mode: Whether in fast mode
        timeout_s: Available timeout in seconds

    Returns:
        Dict with training configuration
    """
    # Estimate steps needed
    batch_size = 32
    estimated_steps_per_epoch = max(1, n_ambiguous_samples // batch_size)

    # Time-based max steps (assume ~0.5s per step on GPU)
    time_based_max = int(timeout_s * 0.6 / 0.5)  # Use 60% of timeout for training

    if fast_mode:
        max_steps = min(DEFAULT_MAX_STEPS_FAST, time_based_max, estimated_steps_per_epoch * 3)
        num_epochs = 1
    else:
        max_steps = min(DEFAULT_MAX_STEPS_FULL, time_based_max, estimated_steps_per_epoch * 5)
        num_epochs = 3

    return {
        "model_name": "t5-small",  # NOT t5-base for speed
        "max_steps": max_steps,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size * 2,
        "learning_rate": 3e-4,
        "warmup_ratio": 0.1,
        "fp16": True,
        "eval_steps": min(500, max(100, max_steps // 4)),
        "save_steps": min(500, max(100, max_steps // 4)),
        "logging_steps": 50,
    }


def create_hybrid_pipeline(
    train_df: pd.DataFrame,
    fast_mode: bool = True,
    timeout_s: int = 1800,
    class_col: str = "class",
    before_col: str = "before",
    after_col: str = "after",
) -> dict[str, Any]:
    """
    Create hybrid lookup + neural pipeline for text normalization.

    Strategy:
    1. Build lookup baseline from training data
    2. Identify ambiguous samples that need neural prediction
    3. Configure neural model only for ambiguous subset

    Args:
        train_df: Training DataFrame
        fast_mode: Whether in fast mode
        timeout_s: Available timeout
        class_col: Name of class column
        before_col: Name of input column
        after_col: Name of target column

    Returns:
        Dict with pipeline components:
        - lookup: LookupBaseline instance
        - ambiguous_df: DataFrame of samples needing neural model
        - neural_config: Training config for neural model (or None if not needed)
        - stats: Coverage statistics
    """
    # Step 1: Build lookup baseline
    lookup = LookupBaseline().fit(train_df, class_col, before_col, after_col)

    # Step 2: Identify samples needing neural prediction
    predictions = lookup.predict_batch(train_df, class_col, before_col)
    ambiguous_df = predictions[predictions["needs_neural"]]
    n_ambiguous = len(ambiguous_df)
    n_total = len(train_df)

    coverage_pct = 100 * (n_total - n_ambiguous) / n_total
    print(f"[HybridPipeline] Lookup coverage: {n_total - n_ambiguous:,} / {n_total:,} ({coverage_pct:.1f}%)")
    print(f"[HybridPipeline] Samples for neural: {n_ambiguous:,} ({100 - coverage_pct:.1f}%)")

    # Step 3: Configure neural model
    if n_ambiguous > 0:
        neural_config = get_neural_training_config(n_ambiguous, fast_mode, timeout_s)
        print(f"[HybridPipeline] Neural config: model={neural_config['model_name']}, max_steps={neural_config['max_steps']}")
    else:
        neural_config = None
        print("[HybridPipeline] No neural model needed - full coverage by lookup!")

    return {
        "lookup": lookup,
        "ambiguous_df": ambiguous_df,
        "ambiguous_indices": ambiguous_df.index.tolist(),
        "neural_config": neural_config,
        "stats": {
            "total_samples": n_total,
            "lookup_coverage": n_total - n_ambiguous,
            "neural_samples": n_ambiguous,
            "coverage_pct": coverage_pct,
        },
    }


def apply_hybrid_predictions(
    test_df: pd.DataFrame,
    lookup: LookupBaseline,
    neural_predictions: np.ndarray | None = None,
    neural_indices: list[int] | None = None,
    class_col: str = "class",
    before_col: str = "before",
) -> np.ndarray:
    """
    Apply hybrid predictions: lookup first, then neural for ambiguous cases.

    Args:
        test_df: Test DataFrame
        lookup: Fitted LookupBaseline
        neural_predictions: Predictions from neural model for ambiguous indices
        neural_indices: Indices in test_df where neural predictions should be used
        class_col: Name of class column
        before_col: Name of input column

    Returns:
        Array of final predictions
    """
    # Get lookup predictions for all
    predictions = lookup.predict_batch(test_df, class_col, before_col)
    final_preds = predictions["prediction"].values.copy()

    # Override with neural predictions where available
    if neural_predictions is not None and neural_indices is not None:
        for i, idx in enumerate(neural_indices):
            if i < len(neural_predictions):
                final_preds[idx] = neural_predictions[i]

    return final_preds
