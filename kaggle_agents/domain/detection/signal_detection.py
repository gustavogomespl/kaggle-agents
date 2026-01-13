"""
Signal detection methods for specific domains.

Contains methods for detecting time series, object detection, segmentation,
multimodal, and metric-based domain signals.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .constants import BBOX_COLUMNS, CLASSIFICATION_METRICS, REGRESSION_METRICS, TIME_TARGETS


if TYPE_CHECKING:
    from ...core.state import CompetitionInfo


class SignalDetectionMixin:
    """Mixin providing signal detection methods for specific domains."""

    def _detect_time_series(self, train_path: Path | None) -> tuple[bool, float, dict[str, Any]]:
        """Detect time series domain by datetime columns and target names."""
        if not train_path or not train_path.exists():
            return False, 0.0, {}

        try:
            df = pd.read_csv(train_path, nrows=100)
        except Exception:
            return False, 0.0, {}

        cols_lower = [c.lower() for c in df.columns]

        strong_time_cols = {"date", "datetime", "timestamp", "time"}
        weak_time_cols = {"year", "month", "day", "week"}

        present_strong = [c for c in df.columns if c.lower() in strong_time_cols]
        present_weak = [c for c in df.columns if c.lower() in weak_time_cols]

        # Signal 1: Typical time series targets
        has_time_target = any(tt in cols_lower for tt in TIME_TARGETS)

        # Signal 2: Try to parse column as datetime (favor strong time names)
        date_parseable = False
        date_col_found = None
        parsed_unique = 0
        for col in (present_strong or present_weak):
            try:
                parsed = pd.to_datetime(df[col].head(50), errors="coerce")
            except Exception:
                continue
            non_null = int(parsed.notna().sum())
            if non_null < max(5, int(len(parsed) * 0.6)):
                continue
            unique_vals = int(parsed.dropna().nunique())
            if unique_vals < 3:
                continue
            date_parseable = True
            date_col_found = col
            parsed_unique = unique_vals
            break

        strong_signal = bool(present_strong)

        if date_parseable and (strong_signal or has_time_target):
            confidence = 0.70
            if strong_signal:
                confidence += 0.10
            if has_time_target:
                confidence += 0.10
            confidence = min(0.90, confidence)
            return True, confidence, {
                "date_col": date_col_found,
                "time_target": has_time_target,
                "strong_time_col": strong_signal,
                "weak_time_cols": len(present_weak),
                "parsed_unique": parsed_unique,
            }

        return False, 0.0, {}

    def _detect_object_detection(
        self, sample_sub_path: Path | None, data_dir: Path
    ) -> tuple[bool, float, dict[str, Any]]:
        """Detect object detection domain by bbox columns or COCO JSON."""
        metadata: dict[str, Any] = {}

        # Signal 1: Bbox columns in sample_submission
        if sample_sub_path and sample_sub_path.exists():
            try:
                df = pd.read_csv(sample_sub_path, nrows=5)
                cols_lower = [c.lower() for c in df.columns]
                bbox_matches = sum(1 for bc in BBOX_COLUMNS if bc in cols_lower)
                if bbox_matches >= 2:
                    return True, 0.92, {"bbox_cols": bbox_matches}

                # PredictionString format (common in object detection)
                if "predictionstring" in cols_lower:
                    return True, 0.90, {"format": "prediction_string"}
            except Exception:
                pass

        # Signal 2: COCO JSON annotations
        if data_dir.exists():
            coco_patterns = ["annotations.json", "train.json"]
            for pattern in coco_patterns:
                matches = list(data_dir.glob(f"**/{pattern}"))
                if matches:
                    return True, 0.88, {"coco_file": str(matches[0])}
            # Also check for instances_*.json
            instances_matches = list(data_dir.glob("**/instances_*.json"))
            if instances_matches:
                return True, 0.88, {"coco_file": str(instances_matches[0])}

        return False, 0.0, {}

    def _detect_segmentation(
        self, sample_sub_path: Path | None
    ) -> tuple[bool, float, dict[str, Any]]:
        """Detect segmentation domain by RLE/mask columns."""
        if not sample_sub_path or not sample_sub_path.exists():
            return False, 0.0, {}

        try:
            df = pd.read_csv(sample_sub_path, nrows=10)
            cols = df.columns.tolist()
            cols_lower = [c.lower() for c in cols]

            # EncodedPixels is a very strong indicator
            if "EncodedPixels" in cols or "encodedpixels" in cols_lower:
                return True, 0.95, {"format": "rle_encoded"}

            # Other indicators
            if any("rle" in c or "mask" in c for c in cols_lower):
                return True, 0.90, {"format": "mask_column"}
        except Exception:
            pass

        return False, 0.0, {}

    def _detect_multimodal(
        self, data_dir: Path, train_path: Path | None
    ) -> tuple[bool, float, dict[str, Any]]:
        """Detect multimodal domain: images + rich tabular features."""
        has_images = self._has_image_assets(data_dir)

        if not has_images or not train_path or not train_path.exists():
            return False, 0.0, {}

        try:
            df = pd.read_csv(train_path, nrows=50)
            n_cols = len(df.columns)
            numeric_cols = len(df.select_dtypes(include="number").columns)

            # Images + many numeric features = multimodal
            if n_cols >= 8 and numeric_cols >= 5:
                return True, 0.85, {"n_features": n_cols, "n_numeric": numeric_cols}
        except Exception:
            pass

        return False, 0.0, {}

    def _detect_from_metric(
        self, competition_info: CompetitionInfo
    ) -> tuple[str | None, float, dict[str, Any]]:
        """Detect classification/regression by evaluation metric."""
        metric = (competition_info.evaluation_metric or "").lower()

        # Regression metrics
        if any(m in metric for m in REGRESSION_METRICS):
            return "regression", 0.88, {"metric": metric, "inferred": "regression"}

        # Classification metrics
        if any(m in metric for m in CLASSIFICATION_METRICS):
            return "classification", 0.85, {"metric": metric, "inferred": "classification"}

        return None, 0.0, {}
