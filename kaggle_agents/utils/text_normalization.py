"""
Text normalization utilities for data-driven lookup and hybrid approaches.

Rules and routing decisions are learned from the supplied training data. This
module deliberately contains no benchmark-specific class taxonomy.

Key classes:
- LookupBaseline: Frequency-based lookup for resolved context/source/target roles
- HybridPipeline: Combines lookup baseline with neural model for ambiguous cases
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data_contract import infer_seq2seq_columns


# Backward-compatible exports. Class membership is now learned by
# ``LookupBaseline.fit`` and exposed on each fitted instance.
DETERMINISTIC_CLASSES: frozenset[str] = frozenset()
AMBIGUOUS_CLASSES: frozenset[str] = frozenset()

_GLOBAL_CONTEXT = "__global_context__"
_CONTEXT_COL = "__context__"
_SOURCE_COL = "__source__"
_TARGET_COL = "__target__"

# Max steps guard for neural training to prevent timeout
DEFAULT_MAX_STEPS_FAST = 2000
DEFAULT_MAX_STEPS_FULL = 10000


class LookupBaseline:
    """
    Frequency-based lookup baseline for text normalization.

    For each context/input pair, stores the most frequent target observed in
    training data. Confidence requires enough observations and high empirical
    purity. Class-level identity, character-spelling, or constant-output rules
    are enabled only when the same data-driven confidence test passes.
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.98,
        min_lookup_count: int = 2,
        min_rule_count: int = 3,
    ):
        if not 0.5 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0.5, 1.0]")
        if min_lookup_count < 1 or min_rule_count < 1:
            raise ValueError("minimum observation counts must be positive")

        self.confidence_threshold = confidence_threshold
        self.min_lookup_count = min_lookup_count
        self.min_rule_count = min_rule_count
        self.lookup: dict[tuple[str, str], str] = {}
        self.confident_lookup_keys: set[tuple[str, str]] = set()
        self.class_fallbacks: dict[str, str] = {}
        self.deterministic_classes: set[str] = set()
        self.ambiguous_classes: set[str] = set()
        self.class_rule_metrics: dict[str, dict[str, float | int | str | None]] = {}
        self.stats: dict[str, int] = defaultdict(int)
        self.class_col: str | None = None
        self.before_col: str | None = None
        self.after_col: str | None = None

    def fit(
        self,
        df: pd.DataFrame,
        class_col: str | None = None,
        before_col: str | None = None,
        after_col: str | None = None,
        *,
        test_df: pd.DataFrame | None = None,
        sample_submission: str | Path | pd.DataFrame | None = None,
        column_contract: Mapping[str, Any] | None = None,
    ) -> LookupBaseline:
        """
        Build lookup table from training data.

        Args:
            df: Training DataFrame
            class_col: Optional context/type column
            before_col: Name of input column
            after_col: Name of target column
            test_df: Test data used to derive train-only target/schema roles
            sample_submission: Submission schema used to confirm the output
            column_contract: Canonical column-role metadata

        Returns:
            self for chaining
        """
        resolved = infer_seq2seq_columns(
            df,
            test_df,
            target_col=after_col,
            source_col=before_col,
            class_col=class_col,
            sample_submission=sample_submission,
            contract=column_contract,
        )
        class_col = resolved["class_col"]
        before_col = resolved["source_col"]
        after_col = resolved["target_col"]

        required_columns = {before_col, after_col}
        if class_col is not None:
            required_columns.add(class_col)
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing normalization columns: {sorted(missing_columns)}")
        if df.empty:
            raise ValueError("Cannot fit lookup baseline on an empty DataFrame")

        self.class_col = class_col
        self.before_col = before_col
        self.after_col = after_col
        self.lookup.clear()
        self.confident_lookup_keys.clear()
        self.class_fallbacks.clear()
        self.deterministic_classes.clear()
        self.ambiguous_classes.clear()
        self.class_rule_metrics.clear()
        self.stats.clear()

        context = (
            df[class_col].fillna("").astype(str)
            if class_col is not None
            else pd.Series(_GLOBAL_CONTEXT, index=df.index, dtype=object)
        )
        normalized = pd.DataFrame(
            {
                _CONTEXT_COL: context,
                _SOURCE_COL: df[before_col].fillna("").astype(str),
                _TARGET_COL: df[after_col].fillna("").astype(str),
            },
            index=df.index,
        )

        # Count context/input -> target frequencies.
        counts = normalized.groupby(
            [_CONTEXT_COL, _SOURCE_COL, _TARGET_COL], as_index=False,
            dropna=False,
        ).size()
        counts.columns = [_CONTEXT_COL, _SOURCE_COL, _TARGET_COL, "count"]
        counts["total"] = counts.groupby([_CONTEXT_COL, _SOURCE_COL], dropna=False)[
            "count"
        ].transform("sum")
        counts["purity"] = counts["count"] / counts["total"]

        # Get the most frequent target for each context/input pair.
        idx = counts.groupby([_CONTEXT_COL, _SOURCE_COL], dropna=False)["count"].idxmax()
        best_mappings = counts.loc[idx]

        for _, row in best_mappings.iterrows():
            key = (str(row[_CONTEXT_COL]), str(row[_SOURCE_COL]))
            self.lookup[key] = str(row[_TARGET_COL])
            self.stats["lookup_entries"] += 1
            if (
                int(row["total"]) >= self.min_lookup_count
                and float(row["purity"]) >= self.confidence_threshold
            ):
                self.confident_lookup_keys.add(key)

        self._build_class_fallbacks(
            normalized,
            _CONTEXT_COL,
            _SOURCE_COL,
            _TARGET_COL,
        )

        print(
            "[LookupBaseline] Built "
            f"{len(self.lookup):,} mappings "
            f"({len(self.confident_lookup_keys):,} confidence-qualified)"
        )
        return self

    def _build_class_fallbacks(
        self,
        df: pd.DataFrame,
        class_col: str,
        before_col: str,
        after_col: str,
    ):
        """Infer conservative fallback rules from observed transformations."""
        for class_value, class_df in df.groupby(class_col, dropna=False):
            class_name = str(class_value)
            before = class_df[before_col].fillna("").astype(str)
            after = class_df[after_col].fillna("").astype(str)
            sample_count = len(class_df)

            identity_rate = float((before == after).mean())
            spelled = before.map(lambda value: " ".join(value.lower()))
            spell_rate = float((spelled == after).mean())
            value_counts = after.value_counts(dropna=False)
            constant_output = str(value_counts.index[0]) if not value_counts.empty else None
            constant_rate = (
                float(value_counts.iloc[0] / sample_count) if sample_count else 0.0
            )

            selected_rule: str | None = None
            if sample_count >= self.min_rule_count:
                if identity_rate >= self.confidence_threshold:
                    selected_rule = "<self>"
                elif spell_rate >= self.confidence_threshold:
                    selected_rule = "<spell>"
                elif constant_output is not None and constant_rate >= self.confidence_threshold:
                    selected_rule = constant_output

            if selected_rule is not None:
                self.class_fallbacks[class_name] = selected_rule
                self.deterministic_classes.add(class_name)
            else:
                self.ambiguous_classes.add(class_name)

            self.class_rule_metrics[class_name] = {
                "sample_count": sample_count,
                "identity_rate": identity_rate,
                "spell_rate": spell_rate,
                "constant_rate": constant_rate,
                "selected_rule": selected_rule,
            }

    def predict(self, class_val: str | None, before_val: str) -> tuple[str, bool]:
        """
        Predict a target string for a (context, source) pair.

        Args:
            class_val: Optional context/type value
            before_val: Source text

        Returns:
            Tuple of (prediction, is_confident)
            is_confident=False indicates fallback was used (may need neural refinement)
        """
        context_value = (
            _GLOBAL_CONTEXT if self.class_col is None else str(class_val)
        )
        key = (context_value, str(before_val))

        # Try exact lookup first
        if key in self.lookup:
            self.stats["lookup_hits"] += 1
            return self.lookup[key], key in self.confident_lookup_keys

        # Try class-level fallback
        fallback = self.class_fallbacks.get(context_value)
        self.stats["fallback_used"] += 1

        if fallback == "<self>":
            # Deterministic: keep as-is
            return before_val, True
        if fallback == "<spell>":
            # Spell out letters: "ABC" -> "a b c"
            spelled = " ".join(before_val.lower())
            return spelled, True
        if fallback:
            return fallback, True
        # Unknown class, keep as-is
        return before_val, False

    def predict_batch(
        self,
        df: pd.DataFrame,
        class_col: str | None = None,
        before_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Predict for entire DataFrame using vectorized operations.

        Args:
            df: DataFrame with the fitted source and optional context columns

        Returns:
            DataFrame with predictions and confidence flags
        """
        class_col = self.class_col if class_col is None else class_col
        before_col = self.before_col if before_col is None else before_col
        if before_col is None:
            raise ValueError(
                "Source column is unavailable; fit with before_col or a column contract"
            )
        if before_col not in df.columns:
            raise ValueError(f"Source column '{before_col}' is missing from prediction data")
        if class_col is not None and class_col not in df.columns:
            raise ValueError(f"Context column '{class_col}' is missing from prediction data")

        # Vectorized approach - 50-100x faster than iterrows
        class_str = (
            df[class_col].fillna("").astype(str)
            if class_col is not None
            else pd.Series(_GLOBAL_CONTEXT, index=df.index, dtype=object)
        )
        before_str = df[before_col].fillna("").astype(str)

        # Create lookup keys as tuples.
        keys = list(zip(class_str, before_str, strict=True))

        # Vectorized lookup using Series.map().
        predictions = pd.Series(keys, index=df.index).map(self.lookup)

        # A memorized mapping is only trusted when its observation count and
        # empirical purity passed the thresholds during fit.
        is_lookup_hit = predictions.notna()
        is_confident_lookup = pd.Series(
            [key in self.confident_lookup_keys for key in keys],
            index=df.index,
        )
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

        # Handle learned constant-output fallbacks.
        is_other_fallback = needs_fallback & ~is_self_fallback & ~is_spell_fallback & ~is_unknown_class

        # Apply fallbacks
        predictions = predictions.where(~is_self_fallback, before_str)
        predictions = predictions.where(~is_spell_fallback, spelled_out)
        predictions = predictions.where(~is_other_fallback, fallback_values)
        predictions = predictions.where(~is_unknown_class, before_str)  # Unknown class: keep as-is

        # Track fallback usage
        self.stats["fallback_used"] += int(needs_fallback.sum())

        # Every class fallback was learned with the configured confidence and
        # sample-count thresholds. Unknown and low-purity mappings fail closed.
        is_confident = (
            is_confident_lookup
            | is_self_fallback
            | is_spell_fallback
            | is_other_fallback
        )
        needs_neural = ~is_confident
        self.stats["uncertain_predictions"] += int(needs_neural.sum())

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

        confident_keys = [f"{key[0]}|||{key[1]}" for key in self.confident_lookup_keys]
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "lookup": lookup_serializable,
                    "confident_lookup_keys": confident_keys,
                    "class_fallbacks": self.class_fallbacks,
                    "deterministic_classes": sorted(self.deterministic_classes),
                    "ambiguous_classes": sorted(self.ambiguous_classes),
                    "class_rule_metrics": self.class_rule_metrics,
                    "confidence_threshold": self.confidence_threshold,
                    "min_lookup_count": self.min_lookup_count,
                    "min_rule_count": self.min_rule_count,
                    "column_contract": {
                        "class_col": self.class_col,
                        "source_col": self.before_col,
                        "target_col": self.after_col,
                    },
                    "stats": dict(self.stats),
                },
                f,
            )

        print(f"[LookupBaseline] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> LookupBaseline:
        """Load lookup table from file."""
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(
            confidence_threshold=float(data.get("confidence_threshold", 0.98)),
            min_lookup_count=int(data.get("min_lookup_count", 2)),
            min_rule_count=int(data.get("min_rule_count", 3)),
        )
        # Convert string keys back to tuples
        instance.lookup = {
            tuple(k.split("|||")): v for k, v in data["lookup"].items()
        }
        serialized_confident_keys = data.get("confident_lookup_keys")
        if serialized_confident_keys is None:
            # Old artifacts did not record confidence; keep their mappings
            # usable while requiring new fits to follow the stricter contract.
            instance.confident_lookup_keys = set(instance.lookup)
        else:
            instance.confident_lookup_keys = {
                tuple(key.split("|||")) for key in serialized_confident_keys
            }
        instance.class_fallbacks = data["class_fallbacks"]
        instance.deterministic_classes = set(data.get("deterministic_classes", []))
        instance.ambiguous_classes = set(data.get("ambiguous_classes", []))
        instance.class_rule_metrics = data.get("class_rule_metrics", {})
        saved_contract = data.get("column_contract") or {}
        instance.class_col = saved_contract.get("class_col")
        instance.before_col = saved_contract.get("source_col")
        instance.after_col = saved_contract.get("target_col")
        instance.stats = defaultdict(int, data.get("stats", {}))

        print(f"[LookupBaseline] Loaded {len(instance.lookup):,} entries from {path}")
        return instance

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about lookup usage."""
        return {
            "total_entries": len(self.lookup),
            "lookup_hits": self.stats.get("lookup_hits", 0),
            "fallback_used": self.stats.get("fallback_used", 0),
            "confidence_qualified_entries": len(self.confident_lookup_keys),
            "uncertain_predictions": self.stats.get("uncertain_predictions", 0),
            "learned_deterministic_classes": sorted(self.deterministic_classes),
            "learned_ambiguous_classes": sorted(self.ambiguous_classes),
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
    class_col: str | None = None,
    before_col: str | None = None,
    after_col: str | None = None,
    validation_folds: int = 3,
    *,
    test_df: pd.DataFrame | None = None,
    sample_submission: str | Path | pd.DataFrame | None = None,
    column_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create hybrid lookup + neural pipeline for text normalization.

    Strategy:
    1. Estimate routing confidence out of fold.
    2. Fit the final lookup baseline on all supplied training rows.
    3. Configure a neural model only for rows not covered confidently OOF.

    Args:
        train_df: Training DataFrame
        fast_mode: Whether in fast mode
        timeout_s: Available timeout
        class_col: Optional context/type column
        before_col: Name of input column
        after_col: Name of target column
        validation_folds: Number of deterministic OOF routing folds
        test_df: Test data used for schema-role resolution
        sample_submission: Submission schema used to confirm the output
        column_contract: Canonical column-role metadata

    Returns:
        Dict with pipeline components:
        - lookup: LookupBaseline instance
        - ambiguous_df: DataFrame of samples needing neural model
        - neural_config: Training config for neural model (or None if not needed)
        - stats: Coverage statistics
    """
    if train_df.empty:
        raise ValueError("Cannot create hybrid pipeline from an empty DataFrame")

    resolved = infer_seq2seq_columns(
        train_df,
        test_df,
        target_col=after_col,
        source_col=before_col,
        class_col=class_col,
        sample_submission=sample_submission,
        contract=column_contract,
    )
    class_col = resolved["class_col"]
    before_col = resolved["source_col"]
    after_col = resolved["target_col"]

    required_columns = {before_col, after_col}
    if class_col is not None:
        required_columns.add(class_col)
    missing_columns = required_columns - set(train_df.columns)
    if missing_columns:
        raise ValueError(f"Missing normalization columns: {sorted(missing_columns)}")

    n_total = len(train_df)

    # Estimate routing out of fold so unique training keys cannot masquerade
    # as generalizable lookup coverage.
    n_splits = min(max(2, validation_folds), n_total) if n_total >= 2 else 1
    oof_confident = np.zeros(n_total, dtype=bool)
    oof_predictions = np.full(n_total, "", dtype=object)

    if n_splits >= 2:
        rng = np.random.default_rng(0)
        shuffled_positions = rng.permutation(n_total)
        fold_by_position = np.empty(n_total, dtype=int)
        fold_by_position[shuffled_positions] = np.arange(n_total) % n_splits

        for fold in range(n_splits):
            validation_positions = np.flatnonzero(fold_by_position == fold)
            training_positions = np.flatnonzero(fold_by_position != fold)
            fold_lookup = LookupBaseline().fit(
                train_df.iloc[training_positions],
                class_col=class_col,
                before_col=before_col,
                after_col=after_col,
            )
            validation_df = train_df.iloc[validation_positions]
            fold_predictions = fold_lookup.predict_batch(
                validation_df,
                class_col=class_col,
                before_col=before_col,
            )
            oof_confident[validation_positions] = fold_predictions["is_confident"].to_numpy()
            oof_predictions[validation_positions] = fold_predictions["prediction"].to_numpy()
        routing_evaluation = "out_of_fold"
    else:
        singleton_lookup = LookupBaseline().fit(
            train_df,
            class_col=class_col,
            before_col=before_col,
            after_col=after_col,
        )
        singleton_predictions = singleton_lookup.predict_batch(
            train_df,
            class_col=class_col,
            before_col=before_col,
        )
        oof_confident[:] = singleton_predictions["is_confident"].to_numpy()
        oof_predictions[:] = singleton_predictions["prediction"].to_numpy()
        routing_evaluation = "in_sample_singleton"

    ambiguous_df = train_df.iloc[np.flatnonzero(~oof_confident)].copy()
    n_ambiguous = len(ambiguous_df)
    coverage_count = int(oof_confident.sum())
    coverage_pct = 100 * coverage_count / n_total
    target_text = train_df[after_col].fillna("").astype(str).to_numpy()
    confident_correct = (oof_predictions.astype(str) == target_text) & oof_confident
    confident_accuracy = (
        float(confident_correct.sum() / coverage_count) if coverage_count else None
    )

    # Final inference artifact uses every public training row, while routing
    # statistics and neural selection remain OOF.
    lookup = LookupBaseline().fit(
        train_df,
        class_col=class_col,
        before_col=before_col,
        after_col=after_col,
    )

    print(
        "[HybridPipeline] OOF confident coverage: "
        f"{coverage_count:,} / {n_total:,} ({coverage_pct:.1f}%)"
    )
    print(f"[HybridPipeline] Samples for neural: {n_ambiguous:,}")

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
        "column_contract": {
            "class_col": class_col,
            "source_col": before_col,
            "target_col": after_col,
        },
        "stats": {
            "total_samples": n_total,
            "lookup_coverage": coverage_count,
            "neural_samples": n_ambiguous,
            "coverage_pct": coverage_pct,
            "confident_accuracy": confident_accuracy,
            "routing_evaluation": routing_evaluation,
        },
    }


def apply_hybrid_predictions(
    test_df: pd.DataFrame,
    lookup: LookupBaseline,
    neural_predictions: np.ndarray | None = None,
    neural_indices: list[int] | None = None,
    class_col: str | None = None,
    before_col: str | None = None,
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
