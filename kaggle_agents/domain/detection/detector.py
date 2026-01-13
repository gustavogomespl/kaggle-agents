"""
Domain Detector - Main detection class.

Contains the DomainDetector class combining all detection methods.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .constants import DOMAIN_DESCRIPTIONS, DOMAINS, FORCE_TYPE_TO_DOMAIN
from .dataclasses import DetectionSignal
from .file_detection import FileDetectionMixin
from .llm_detection import LLMDetectionMixin
from .signal_detection import SignalDetectionMixin
from .signal_fusion import SignalFusionMixin
from .submission_format import SubmissionFormatMixin
from .text_tabular_detection import TextTabularDetectionMixin


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from ...core.state import CompetitionInfo, DomainType


# LLM Prompt for domain classification
DOMAIN_PROMPT = """Classify this Kaggle competition into exactly ONE category.

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


class DomainDetector(
    FileDetectionMixin,
    SignalDetectionMixin,
    TextTabularDetectionMixin,
    LLMDetectionMixin,
    SignalFusionMixin,
    SubmissionFormatMixin,
):
    """
    Detects the domain/type of a Kaggle competition using LLM.

    Supports granular domain classification for various competition types.
    """

    DOMAINS = DOMAINS
    PROMPT = DOMAIN_PROMPT
    DESCRIPTIONS = DOMAIN_DESCRIPTIONS

    def __init__(self, llm: BaseChatModel | None = None):
        """
        Initialize the domain detector.

        Args:
            llm: LangChain LLM client. If None, defaults to tabular domain.
        """
        self.llm = llm

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
        # =====================================================================
        # FORCE DOMAIN OVERRIDE - Check environment variables first
        # This takes ABSOLUTE precedence over all detection methods
        # =====================================================================
        forced_type = (
            os.getenv("KAGGLE_AGENTS_FORCE_DATA_TYPE", "")
            or os.getenv("KAGGLE_AGENTS_DATA_TYPE", "")
            or os.getenv("KAGGLE_AGENTS_FORCE_DOMAIN", "")
        ).strip().lower()

        if forced_type and forced_type in FORCE_TYPE_TO_DOMAIN:
            forced_domain = FORCE_TYPE_TO_DOMAIN[forced_type]
            print("\n" + "=" * 60)
            print("= DOMAIN DETECTION RESULTS (FORCED)")
            print("=" * 60)
            print("\n⚠️  Domain FORCED via environment variable")
            print(f"   KAGGLE_AGENTS_FORCE_DATA_TYPE={forced_type!r}")
            print(f"\n✅ FINAL: {forced_domain} (confidence: 1.00)")
            print("=" * 60 + "\n")
            return forced_domain, 1.0  # type: ignore
        # =====================================================================

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

        # Text/NLP detection (BEFORE tabular to correctly classify NLP competitions in CSV)
        text_result = self._detect_text_from_csv(competition_info, data_dir)
        if text_result:
            signals.append(DetectionSignal(
                text_result[0], text_result[1], "structural"
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
