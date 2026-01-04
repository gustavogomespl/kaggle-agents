"""
Domain Detection Module for Kaggle Competitions.

Uses LLM to classify competition domain based on description and file metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from ..core.state import CompetitionInfo, DomainType, SubmissionFormatType
from ..utils.llm_utils import get_text_content


# =============================================================================
# DETECTION SIGNAL DATACLASS
# =============================================================================

@dataclass
class DetectionSignal:
    """A single detection signal with source and confidence."""

    domain: str  # DomainType or partial (e.g., "classification", "regression")
    confidence: float
    source: Literal["structural", "submission", "metric", "llm", "sota", "fallback"]
    metadata: dict[str, Any] = field(default_factory=dict)


# Image extensions for format detection
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif"}

# =============================================================================
# EXTENDED CONSTANTS FOR MULTI-SIGNAL DETECTION
# =============================================================================

# Expanded patterns for image-to-image detection
CLEAN_DIR_PATTERNS = [
    # Existing patterns
    "train_cleaned", "train_clean", "clean", "cleaned",
    "gt", "ground_truth", "train_gt", "target", "targets", "train_target",
    # New patterns for denoising, super-resolution, etc.
    "hr", "high_res", "high_resolution", "hq", "high_quality",
    "output", "outputs", "label", "labels",
    "denoised", "restored", "enhanced", "original",
]

# Columns indicating time series data
TIME_COLUMNS = ["date", "datetime", "timestamp", "time", "day", "week", "month", "year"]
TIME_TARGETS = ["sales", "demand", "volume", "price", "count", "visitors", "quantity"]

# Columns indicating object detection
BBOX_COLUMNS = ["x", "y", "width", "height", "x_min", "y_min", "x_max", "y_max", "bbox", "confidence"]

# Regression metrics
REGRESSION_METRICS = ["rmse", "mae", "mse", "mape", "r2", "rmsle", "medae"]

# Classification metrics
CLASSIFICATION_METRICS = ["logloss", "auc", "accuracy", "f1", "precision", "recall", "map@k", "mrr"]

# Source weights for signal fusion
SOURCE_WEIGHTS = {
    "structural": 1.0,    # Structural heuristics - most reliable
    "submission": 0.9,    # Submission format analysis
    "metric": 0.85,       # Evaluation metric signal
    "llm": 0.7,           # LLM inference
    "sota": 0.6,          # SOTA tags from notebooks
    "fallback": 0.5,      # Fallback generic
}


class DomainDetector:
    """
    Detects the domain/type of a Kaggle competition using LLM.

    Supports granular domain classification for various competition types.
    """

    DOMAINS = [
        # Image-based
        "image_classification",
        "image_regression",
        "image_to_image",
        "image_segmentation",
        "object_detection",
        # Text-based
        "text_classification",
        "seq_to_seq",
        "text_regression",
        # Audio-based
        "audio_classification",
        "audio_regression",
        # Tabular
        "tabular_classification",
        "tabular_regression",
        # Time series
        "time_series_forecasting",
        # Multi-modal
        "multi_modal",
    ]

    PROMPT = """Classify this Kaggle competition into exactly ONE category.

Categories:
- image_classification: Classify images into categories (dog breeds, cancer detection, plant diseases)
- image_regression: Predict continuous values from images (age estimation, severity scores)
- image_to_image: Transform images (denoising, super-resolution, style transfer) or pixel-level (one row per pixel) if it's an image-to-image task and the sample submission has pixel-level format in csv
- image_segmentation: Pixel-wise classification of images
- object_detection: Locate and classify objects in images
- text_classification: Classify text (sentiment, toxicity, spam, author identification)
- seq_to_seq: Sequence to sequence (translation, text normalization, summarization)
- text_regression: Predict continuous values from text
- audio_classification: Classify audio signals (speaker, music genre, species by sound)
- audio_regression: Predict continuous values from audio
- tabular_classification: Classify rows in structured CSV data
- tabular_regression: Predict continuous values from structured CSV data
- time_series_forecasting: Predict future values from temporal sequences
- multi_modal: Combination of multiple data types (images + text + tabular)

Competition Name: {name}
Description: {description}
Data Files: {files}

IMPORTANT CLUES FOR DETECTION:
- Directories ending with "/" containing .jpg/.png files → image_* domain
- Directories with .wav/.mp3 files → audio_* domain
- Directories with .txt files → text_* domain
- Only .csv/.parquet files with no directories → tabular_* domain

Respond with ONLY the category name, nothing else. Example: image_classification"""

    DESCRIPTIONS = {
        "image_classification": "Classify images into discrete categories",
        "image_regression": "Predict continuous values from images",
        "image_to_image": "Transform images (denoising, super-resolution, style transfer) or pixel-level (one row per pixel)",
        "image_segmentation": "Pixel-wise classification of images",
        "object_detection": "Locate and classify objects in images",
        "text_classification": "Classify text into categories",
        "seq_to_seq": "Sequence to sequence transformation (translation, normalization)",
        "text_regression": "Predict continuous values from text",
        "audio_classification": "Classify audio signals",
        "audio_regression": "Predict continuous values from audio",
        "tabular_classification": "Classify rows in structured tabular data",
        "tabular_regression": "Predict continuous values from tabular data",
        "time_series_forecasting": "Predict future values from temporal data",
        "multi_modal": "Combination of multiple data types",
    }

    def __init__(self, llm: BaseChatModel | None = None):
        """
        Initialize the domain detector.

        Args:
            llm: LangChain LLM client. If None, defaults to tabular domain.
        """
        self.llm = llm

    def _detect_from_structure(
        self, competition_info: CompetitionInfo, data_dir: Path
    ) -> tuple[DomainType, float]:
        """Heuristic detection from local files when no LLM is available."""
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        text_exts = {".txt", ".json"}
        tabular_exts = {".csv", ".parquet"}

        def classify(counts: dict[str, int], total: int) -> tuple[DomainType, float] | None:
            if total == 0:
                return None

            image_ratio = sum(counts.get(ext, 0) for ext in image_exts) / total
            audio_ratio = sum(counts.get(ext, 0) for ext in audio_exts) / total
            text_ratio = sum(counts.get(ext, 0) for ext in text_exts) / total
            tabular_ratio = sum(counts.get(ext, 0) for ext in tabular_exts) / total

            if image_ratio >= 0.3:
                return ("image_classification", 0.90)
            if audio_ratio >= 0.3:
                return ("audio_classification", 0.85)
            if text_ratio >= 0.3:
                # Use regression hint if problem_type mentions it
                if "regression" in (competition_info.problem_type or "").lower():
                    return ("text_regression", 0.80)
                return ("text_classification", 0.80)
            if tabular_ratio >= 0.5:
                if "regression" in (competition_info.problem_type or "").lower():
                    return ("tabular_regression", 0.80)
                return ("tabular_classification", 0.80)

            return None

        def analyze_dir(dir_path: Path) -> tuple[dict[str, int], int]:
            counts: dict[str, int] = {}
            total = 0
            for i, file_path in enumerate(dir_path.rglob("*")):
                if i >= 600:
                    break
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext:
                        counts[ext] = counts.get(ext, 0) + 1
                        total += 1
            return counts, total

        # Prefer train/test folders if present
        candidate_dirs = (
            [
                p
                for p in data_dir.iterdir()
                if p.is_dir() and p.name.lower().startswith(("train", "test"))
            ]
            if data_dir.exists()
            else []
        )

        # Image-to-image heuristic: look for paired train + clean/target directories
        if data_dir.exists():
            dir_map = {p.name.lower(): p for p in data_dir.iterdir() if p.is_dir()}
            # Use expanded CLEAN_DIR_PATTERNS constant
            train_dir = None
            for name in ("train", "training", "train_images"):
                if name in dir_map:
                    train_dir = dir_map[name]
                    break
            clean_dir = None
            for name in CLEAN_DIR_PATTERNS:
                if name in dir_map:
                    clean_dir = dir_map[name]
                    break

            if train_dir and clean_dir:
                train_counts, train_total = analyze_dir(train_dir)
                clean_counts, clean_total = analyze_dir(clean_dir)
                train_result = classify(train_counts, train_total)
                clean_result = classify(clean_counts, clean_total)
                if train_result and clean_result:
                    if train_result[0].startswith("image_") and clean_result[0].startswith(
                        "image_"
                    ):
                        return ("image_to_image", 0.92)

        for dir_path in candidate_dirs:
            counts, total = analyze_dir(dir_path)
            result = classify(counts, total)
            if result:
                return result

        # Fall back to scanning the whole directory tree
        if data_dir.exists():
            counts, total = analyze_dir(data_dir)
            result = classify(counts, total)
            if result:
                return result

        # Default tabular guess
        if "regression" in (competition_info.problem_type or "").lower():
            return ("tabular_regression", 0.50)
        return ("tabular_classification", 0.50)

    def _detect_data_type(self, data_dir: Path) -> str:
        """
        Detect the primary data type from file structure.

        Returns one of: "image", "audio", "text", "tabular"
        """
        if not data_dir.exists():
            return "tabular"

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif"}
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        text_exts = {".txt", ".json"}

        counts: dict[str, int] = {"image": 0, "audio": 0, "text": 0, "tabular": 0}

        # Check direct files and subdirectories
        for path in data_dir.iterdir():
            if path.is_file():
                ext = path.suffix.lower()
                if ext in image_exts:
                    counts["image"] += 1
                elif ext in audio_exts:
                    counts["audio"] += 1
                elif ext in text_exts:
                    counts["text"] += 1
                elif ext in {".csv", ".parquet"}:
                    counts["tabular"] += 1
            elif path.is_dir():
                # Sample first 100 files in subdirectory
                for i, subfile in enumerate(path.rglob("*")):
                    if i >= 100:
                        break
                    if subfile.is_file():
                        ext = subfile.suffix.lower()
                        if ext in image_exts:
                            counts["image"] += 10  # Weight directories higher
                        elif ext in audio_exts:
                            counts["audio"] += 10
                        elif ext in text_exts:
                            counts["text"] += 10

        # Return dominant type
        if counts["image"] > max(counts["audio"], counts["text"], counts["tabular"]):
            return "image"
        if counts["audio"] > max(counts["text"], counts["tabular"]):
            return "audio"
        if counts["text"] > counts["tabular"]:
            return "text"
        return "tabular"

    def _find_train_file(self, data_dir: Path) -> Path | None:
        """Find a train.csv/train.parquet file in the data directory or one level below."""
        candidates = [data_dir / "train.csv", data_dir / "train.parquet"]
        for path in candidates:
            if path.exists():
                return path

        if data_dir.exists():
            for child in data_dir.iterdir():
                if not child.is_dir():
                    continue
                for name in ("train.csv", "train.parquet"):
                    path = child / name
                    if path.exists():
                        return path
        return None

    def _has_image_assets(self, data_dir: Path) -> bool:
        """Return True if the data directory contains image assets."""
        if not data_dir.exists():
            return False

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif"}
        candidate_dirs = [
            p
            for p in data_dir.iterdir()
            if p.is_dir() and p.name.lower().startswith(("train", "test", "images"))
        ]
        if not candidate_dirs:
            candidate_dirs = [data_dir]

        for dir_path in candidate_dirs:
            for i, file_path in enumerate(dir_path.rglob("*")):
                if i >= 200:
                    break
                if file_path.is_file() and file_path.suffix.lower() in image_exts:
                    return True

        return False

    def _detect_tabular_from_csv(
        self, competition_info: CompetitionInfo, data_dir: Path
    ) -> tuple[DomainType, float] | None:
        """
        Heuristic override: if train.csv is wide with many numeric columns,
        treat the competition as tabular when no strong image signal exists.
        """
        if self._has_image_assets(data_dir):
            return None

        train_path = self._find_train_file(data_dir)
        if not train_path:
            return None

        try:
            if train_path.suffix.lower() == ".parquet":
                if train_path.stat().st_size > 200 * 1024 * 1024:
                    return None
                df = pd.read_parquet(train_path)
            else:
                df = pd.read_csv(train_path, nrows=50)
        except Exception:
            return None

        if df.empty:
            return None

        n_cols = len(df.columns)
        if n_cols < 8:
            return None

        numeric_cols = df.select_dtypes(include="number").columns
        numeric_ratio = len(numeric_cols) / max(n_cols, 1)

        # Strong signal: many numeric feature columns
        if len(numeric_cols) >= 10 or (n_cols >= 12 and numeric_ratio >= 0.5):
            is_regression = "regression" in (competition_info.problem_type or "").lower()
            return ("tabular_regression", 0.92) if is_regression else ("tabular_classification", 0.92)

        return None

    # =========================================================================
    # NEW MULTI-SIGNAL DETECTION METHODS
    # =========================================================================

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

    def _refine_domain_with_metric(
        self, base_domain: str, signals: list[DetectionSignal]
    ) -> str:
        """Refine base domain by combining with metric signal."""
        # Domains that need classification/regression suffix
        base_domains = ["tabular", "image", "text", "audio"]

        # Preserve specific domains that should not be overridden by metric
        if base_domain in {
            "object_detection",
            "image_segmentation",
            "image_to_image",
            "time_series_forecasting",
            "multi_modal",
        }:
            return base_domain

        # Check if base domain needs refinement
        domain_base = base_domain.split("_")[0]  # "tabular_classification" -> "tabular"
        if domain_base not in base_domains:
            return base_domain  # Already specific (e.g., time_series_forecasting)

        # If already has suffix, keep it
        if "_classification" in base_domain or "_regression" in base_domain:
            return base_domain

        # Look for metric signal
        metric_signal = next((s for s in signals if s.source == "metric"), None)
        if metric_signal:
            suffix = metric_signal.domain  # "classification" or "regression"
            return f"{domain_base}_{suffix}"

        # Default: classification
        return f"{domain_base}_classification"

    def _extract_sota_tags(self, state: dict[str, Any] | None) -> str | None:
        """Extract keywords from SOTA solutions for LLM context."""
        if not state:
            return None

        sota_solutions = state.get("sota_solutions", [])
        if not sota_solutions:
            return None

        # Collect models and strategies mentioned
        models: list[str] = []
        strategies: list[str] = []
        for sol in sota_solutions[:5]:  # Top 5 solutions
            models.extend(sol.get("models_used", []))
            strategies.extend(sol.get("strategies", []))

        # Remove duplicates and limit
        models = list(set(models))[:5]
        strategies = list(set(strategies))[:3]

        if not models and not strategies:
            return None

        tags = []
        if models:
            tags.append(f"Models: {', '.join(models)}")
        if strategies:
            tags.append(f"Strategies: {', '.join(strategies)}")

        return " | ".join(tags)

    def _fuse_signals(self, signals: list[DetectionSignal]) -> tuple[str, float]:
        """Fuse signals using weights and boost for agreement."""
        if not signals:
            return "tabular_classification", 0.50

        # Group signals by domain and calculate weighted scores
        domain_scores: dict[str, float] = {}
        domain_signals: dict[str, list[DetectionSignal]] = {}

        for signal in signals:
            weighted_conf = signal.confidence * SOURCE_WEIGHTS.get(signal.source, 0.5)
            domain_scores[signal.domain] = domain_scores.get(signal.domain, 0) + weighted_conf
            domain_signals.setdefault(signal.domain, []).append(signal)

        # Find domain with highest score among valid domains
        valid_domains = [domain for domain in domain_scores if domain in self.DOMAINS]
        if valid_domains:
            best_domain = max(valid_domains, key=lambda d: domain_scores[d])
        else:
            metric_signal = next(
                (
                    s
                    for s in signals
                    if s.source == "metric" and s.domain in {"classification", "regression"}
                ),
                None,
            )
            if metric_signal:
                best_domain = f"tabular_{metric_signal.domain}"
            else:
                best_domain = "tabular_classification"
            domain_scores.setdefault(best_domain, 0.0)
            domain_signals.setdefault(best_domain, [])

        agreeing_signals = domain_signals.get(best_domain, [])

        # Calculate base confidence (weighted average of agreeing signals)
        total_weight = sum(SOURCE_WEIGHTS.get(s.source, 0.5) for s in agreeing_signals)
        if agreeing_signals:
            base_confidence = domain_scores[best_domain] / max(total_weight, 1)
        else:
            base_confidence = 0.50

        # Boost for agreement
        n_agreeing = len(agreeing_signals)
        n_sources = len(set(s.source for s in agreeing_signals))

        if n_agreeing >= 3 or n_sources >= 2:
            boost = 0.15 if n_sources >= 2 else 0.10
        else:
            boost = 0.0

        final_confidence = min(0.98, base_confidence + boost)
        return best_domain, final_confidence

    def _log_detection_results(
        self,
        signals: list[DetectionSignal],
        final_domain: str,
        final_confidence: float,
    ) -> None:
        """Rich logging of all signals and scores for debug."""
        print("\n" + "=" * 60)
        print("= DOMAIN DETECTION RESULTS")
        print("=" * 60)

        # Group scores by domain
        domain_scores: dict[str, list[tuple[str, float, float]]] = {}
        for s in signals:
            weighted = s.confidence * SOURCE_WEIGHTS.get(s.source, 0.5)
            domain_scores.setdefault(s.domain, []).append((s.source, s.confidence, weighted))

        # Show each domain with its signals
        print("\n📊 Signals by Domain:")
        for domain, signal_list in sorted(
            domain_scores.items(), key=lambda x: -sum(s[2] for s in x[1])
        ):
            total = sum(s[2] for s in signal_list)
            print(f"\n  {domain}: (total score: {total:.3f})")
            for source, conf, weighted in signal_list:
                weight = SOURCE_WEIGHTS.get(source, 0.5)
                print(f"    └─ {source}: conf={conf:.2f} × weight={weight:.1f} = {weighted:.3f}")

        # Final result
        print(f"\n✅ FINAL: {final_domain} (confidence: {final_confidence:.2f})")
        print("=" * 60 + "\n")

    def _call_llm_diagnostic(
        self, competition_info: CompetitionInfo, data_dir: Path
    ) -> tuple[DomainType | None, float]:
        """Call LLM with diagnostic prompt when signals are weak."""
        if not self.llm:
            return None, 0.0

        diagnostic_prompt = f"""Analyze this Kaggle competition and list the 3 most likely domains with reasons.

Competition: {competition_info.name}
Description: {(competition_info.description or "")[:1500]}
Metric: {competition_info.evaluation_metric or "unknown"}

Response format:
1. <domain>: <short reason>
2. <domain>: <short reason>
3. <domain>: <short reason>

Most likely domain (name only):"""

        try:
            response = self.llm.invoke(diagnostic_prompt)
            content = (
                get_text_content(response.content)
                if hasattr(response, "content")
                else str(response)
            )

            # Extract last domain mentioned (most likely)
            lines = content.strip().split("\n")
            last_line = lines[-1].strip().lower().replace(" ", "_")

            if last_line in self.DOMAINS:
                return last_line, 0.70  # type: ignore

            # Try to find any valid domain in the response
            for domain in self.DOMAINS:
                if domain in content.lower():
                    return domain, 0.65  # type: ignore

        except Exception:
            pass

        return None, 0.0

    def _call_llm_with_context(
        self,
        competition_info: CompetitionInfo,
        data_dir: Path,
        data_type: str,
        files: list[str],
        sota_tags: str | None = None,
    ) -> tuple[DomainType | None, float]:
        """Call LLM with enhanced context for domain detection."""
        if not self.llm:
            return None, 0.0

        # Build SOTA context
        sota_context = f"\nSOTA approaches suggest: {sota_tags}" if sota_tags else ""

        # Enhanced prompt with 1500 chars + metric + SOTA tags
        prompt = f"""Classify this Kaggle competition into exactly ONE category.

## Categories with Examples:
- image_classification: Dog breeds, plant diseases, cancer detection
- object_detection: Bounding boxes required (x,y,w,h), YOLO/RCNN tasks
- image_segmentation: Pixel-wise masks, RLE encoding, semantic/instance segmentation
- image_to_image: Denoising, super-resolution, inpainting, style transfer
- time_series_forecasting: Sales prediction, demand forecasting, temporal data
- tabular_classification/regression: Structured CSV with many numeric features
- text_classification: Sentiment, toxicity, spam detection
- seq_to_seq: Translation, summarization, text normalization
- audio_classification/regression: Audio signal classification or prediction
- multi_modal: Images + text + tabular combined (e.g., images WITH extracted features)

## Few-Shot Examples:
- "Dog Breed Identification" + train/ images → image_classification
- "Store Sales Forecasting" + date column → time_series_forecasting
- "Global Wheat Detection" + bbox columns → object_detection
- "Airbus Ship Detection" + EncodedPixels → image_segmentation
- "Leaf Classification" + images/ + 192 feature columns (margin, shape, texture) → multi_modal

## CRITICAL: Multi-modal Detection
If the competition has BOTH:
1. Image directories (train/, images/)
2. AND train.csv with many numeric features (>5 columns of extracted features)
→ This is likely MULTI_MODAL, not just image_classification!

Competition: {competition_info.name}
Description: {(competition_info.description or "")[:1500]}
Files: {files if files else ["No files found"]}
Data Type Detected: {data_type}
Evaluation Metric: {competition_info.evaluation_metric or "unknown"}{sota_context}

Think step by step:
1. What is the primary data type? {data_type}
2. Are there BOTH images AND numeric features?
3. What does the submission format require?
4. What metric is used? (regression metrics = *_regression, classification = *_classification)

Final answer (category name only):"""

        try:
            response = self.llm.invoke(prompt)
            content = (
                get_text_content(response.content)
                if hasattr(response, "content")
                else str(response)
            )
            domain = content.strip().lower().replace(" ", "_")

            # Extract last line if multiple lines
            if "\n" in domain:
                domain = domain.split("\n")[-1].strip()

            if domain in self.DOMAINS:
                return domain, 0.95  # type: ignore

        except Exception:
            pass

        return None, 0.0

    def detect(
        self,
        competition_info: CompetitionInfo,
        data_directory: Path | str,
        state: dict[str, Any] | None = None,
    ) -> tuple[DomainType, float]:
        """
        Detect the domain type using multi-signal fusion.

        Strategy:
        1. Collect ALL structural signals (time series, object detection, etc.)
        2. Collect metric signal (classification vs regression)
        3. Call LLM with enhanced context (always participates if available)
        4. If no strong signals, use diagnostic LLM fallback
        5. Fuse all signals using weighted scoring
        6. Refine domain with metric signal

        Args:
            competition_info: Competition metadata
            data_directory: Path to competition data files
            state: Optional workflow state for SOTA tags

        Returns:
            Tuple of (detected_domain, confidence_score)
        """
        data_dir = Path(data_directory) if isinstance(data_directory, str) else data_directory
        signals: list[DetectionSignal] = []

        # Find key files
        train_path = self._find_train_file(data_dir)
        sample_sub = data_dir / "sample_submission.csv"
        if not sample_sub.exists():
            for p in data_dir.glob("**/sample_submission.csv"):
                sample_sub = p
                break

        # =====================================================================
        # STEP 1: Collect ALL structural signals
        # =====================================================================

        # Time series detection
        is_ts, ts_conf, ts_meta = self._detect_time_series(train_path)
        if is_ts:
            signals.append(DetectionSignal(
                "time_series_forecasting", ts_conf, "structural", ts_meta
            ))

        # Object detection
        is_od, od_conf, od_meta = self._detect_object_detection(sample_sub, data_dir)
        if is_od:
            signals.append(DetectionSignal(
                "object_detection", od_conf, "structural", od_meta
            ))

        # Segmentation detection
        is_seg, seg_conf, seg_meta = self._detect_segmentation(sample_sub)
        if is_seg:
            signals.append(DetectionSignal(
                "image_segmentation", seg_conf, "structural", seg_meta
            ))

        # Multimodal detection
        is_mm, mm_conf, mm_meta = self._detect_multimodal(data_dir, train_path)
        if is_mm:
            signals.append(DetectionSignal(
                "multi_modal", mm_conf, "structural", mm_meta
            ))

        # Existing heuristics (tabular override, image-to-image)
        tabular_result = self._detect_tabular_from_csv(competition_info, data_dir)
        if tabular_result:
            signals.append(DetectionSignal(
                tabular_result[0], tabular_result[1], "structural"
            ))

        structural_result = self._detect_from_structure(competition_info, data_dir)
        signals.append(DetectionSignal(
            structural_result[0], structural_result[1], "structural"
        ))

        # =====================================================================
        # STEP 2: Metric signal (classification vs regression)
        # =====================================================================
        metric_type, metric_conf, metric_meta = self._detect_from_metric(competition_info)
        if metric_type:
            signals.append(DetectionSignal(
                metric_type, metric_conf, "metric", metric_meta
            ))

        # =====================================================================
        # STEP 3: LLM ALWAYS participates (if available)
        # =====================================================================
        if self.llm:
            # Detect data type for context
            data_type = self._detect_data_type(data_dir)

            # Scan files to provide context
            files = []
            if data_dir.exists():
                for path in data_dir.glob("*"):
                    if path.is_file():
                        files.append(path.name)
                    elif path.is_dir():
                        contents = list(path.glob("*"))[:100]
                        if contents:
                            extensions: dict[str, int] = {}
                            for item in contents:
                                ext = item.suffix.lower()
                                extensions[ext] = extensions.get(ext, 0) + 1
                            if extensions:
                                dominant = max(extensions.items(), key=lambda x: x[1])
                                files.append(
                                    f"{path.name}/ ({len(contents)} files, mostly {dominant[0]})"
                                )
                files = files[:20]

            # Extract SOTA tags if available
            sota_tags = self._extract_sota_tags(state)

            # Call LLM with enhanced context
            llm_domain, llm_conf = self._call_llm_with_context(
                competition_info, data_dir, data_type, files, sota_tags
            )
            if llm_domain:
                signals.append(DetectionSignal(llm_domain, llm_conf, "llm"))

        # =====================================================================
        # STEP 4: Fallback: if no strong signal, use diagnostic LLM
        # =====================================================================
        max_conf = max((s.confidence for s in signals), default=0)
        if max_conf < 0.6 and self.llm:
            diagnostic_domain, diagnostic_conf = self._call_llm_diagnostic(
                competition_info, data_dir
            )
            if diagnostic_domain:
                signals.append(DetectionSignal(
                    diagnostic_domain, diagnostic_conf, "llm"
                ))

        # =====================================================================
        # STEP 5: Fuse all signals
        # =====================================================================
        domain, confidence = self._fuse_signals(signals)

        # =====================================================================
        # STEP 6: Refine domain with metric signal
        # =====================================================================
        domain = self._refine_domain_with_metric(domain, signals)

        # =====================================================================
        # STEP 7: Log results for debug
        # =====================================================================
        self._log_detection_results(signals, domain, confidence)

        return domain, confidence  # type: ignore

    def get_domain_description(self, domain: DomainType) -> str:
        """Get a human-readable description of a domain."""
        return self.DESCRIPTIONS.get(domain, "Unknown domain type")

    def detect_submission_format(
        self,
        sample_submission_path: Path | str,
        test_dir: Path | str | None = None,
        competition_info: CompetitionInfo | None = None,
    ) -> tuple[SubmissionFormatType, dict[str, Any]]:
        """
        Detect the submission format by analyzing sample_submission.csv.

        This is critical for image-to-image tasks where submission format is
        pixel-level (one row per pixel) rather than standard (one row per sample).

        Args:
            sample_submission_path: Path to sample_submission.csv
            test_dir: Optional path to test data directory
            competition_info: Optional competition metadata

        Returns:
            Tuple of (format_type, metadata)
            metadata includes: expected_rows, id_pattern, pixel_format_detected, etc.
        """
        sample_path = Path(sample_submission_path)
        test_path = Path(test_dir) if test_dir else None

        metadata: dict[str, Any] = {
            "expected_rows": 0,
            "n_test_samples": 0,
            "id_column": "",
            "value_columns": [],
            "id_pattern": None,
            "pixel_format_detected": False,
        }

        # Read sample submission
        if not sample_path.exists():
            return "standard", metadata

        try:
            sample_sub = pd.read_csv(sample_path)
        except Exception:
            return "standard", metadata

        n_rows = len(sample_sub)
        metadata["expected_rows"] = n_rows

        if len(sample_sub.columns) == 0:
            return "standard", metadata

        id_col = sample_sub.columns[0]
        metadata["id_column"] = id_col
        metadata["value_columns"] = list(sample_sub.columns[1:])

        # Count test samples (images or files)
        n_test_samples = 0
        if test_path and test_path.exists():
            if test_path.is_dir():
                # Count test images
                test_files = list(test_path.glob("*"))
                n_test_samples = len(
                    [f for f in test_files if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
                )
                # If no images found, count all files
                if n_test_samples == 0:
                    n_test_samples = len([f for f in test_files if f.is_file()])

        metadata["n_test_samples"] = n_test_samples

        # Heuristic 1: If rows >> test_samples, likely pixel-level
        if n_test_samples > 0 and n_rows > n_test_samples * 100:
            # Check ID pattern for pixel format (e.g., "1_1_1" = image_row_col)
            sample_ids = sample_sub[id_col].astype(str).head(20).tolist()

            # Check if IDs contain underscores (common pixel format: image_row_col)
            if sample_ids and all("_" in str(id_val) for id_val in sample_ids):
                parts = str(sample_ids[0]).split("_")
                if len(parts) >= 3:
                    metadata["id_pattern"] = "image_row_col"
                    metadata["pixel_format_detected"] = True
                    metadata["estimated_pixels_per_image"] = n_rows // n_test_samples
                    return "pixel_level", metadata
                if len(parts) == 2:
                    # Could be image_pixel_index format
                    metadata["id_pattern"] = "image_pixel"
                    metadata["pixel_format_detected"] = True
                    metadata["estimated_pixels_per_image"] = n_rows // n_test_samples
                    return "pixel_level", metadata

            # Even without underscore pattern, high ratio suggests pixel-level
            ratio = n_rows / n_test_samples
            if ratio > 1000:  # More than 1000 rows per test sample
                metadata["pixel_format_detected"] = True
                metadata["estimated_pixels_per_image"] = int(ratio)
                return "pixel_level", metadata

        # Heuristic 2: Check for RLE encoding pattern (segmentation)
        if "rle" in id_col.lower() or "EncodedPixels" in sample_sub.columns:
            metadata["id_pattern"] = "rle_encoded"
            return "rle_encoded", metadata

        # Heuristic 3: Check for multi-label format (multiple rows per sample)
        if n_test_samples > 0 and n_rows > n_test_samples * 2:
            # Could be multi-label, check for repeated IDs
            sample_ids = sample_sub[id_col].head(100)
            if sample_ids.duplicated().any():
                metadata["id_pattern"] = "multi_label"
                return "multi_label", metadata

        # Default: standard format (one row per sample)
        return "standard", metadata


# ==================== Convenience Function ====================


def detect_competition_domain(
    competition_info: CompetitionInfo,
    data_directory: Path | str,
    llm: BaseChatModel | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[DomainType, float]:
    """
    Convenience function to detect competition domain using multi-signal fusion.

    Args:
        competition_info: Competition metadata
        data_directory: Path to competition data
        llm: Optional LLM client
        state: Optional workflow state for SOTA tags extraction

    Returns:
        Tuple of (domain_type, confidence)
    """
    detector = DomainDetector(llm=llm)
    return detector.detect(competition_info, data_directory, state=state)


def detect_submission_format(
    sample_submission_path: Path | str,
    test_dir: Path | str | None = None,
    competition_info: CompetitionInfo | None = None,
) -> tuple[SubmissionFormatType, dict[str, Any]]:
    """
    Convenience function to detect submission format.

    Critical for distinguishing between:
    - Standard format (one row per sample) - most competitions
    - Pixel-level format (one row per pixel) - image-to-image, segmentation

    Args:
        sample_submission_path: Path to sample_submission.csv
        test_dir: Optional path to test data directory
        competition_info: Optional competition metadata

    Returns:
        Tuple of (format_type, metadata)
    """
    detector = DomainDetector()
    return detector.detect_submission_format(sample_submission_path, test_dir, competition_info)
