"""
LLM-based detection methods.

Contains methods for domain detection using LLM inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...utils.llm_utils import get_text_content
from .constants import DOMAINS


if TYPE_CHECKING:
    from ...core.state import CompetitionInfo, DomainType


class LLMDetectionMixin:
    """Mixin providing LLM-based detection methods."""

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

            if last_line in DOMAINS:
                return last_line, 0.70  # type: ignore

            # Try to find any valid domain in the response
            for domain in DOMAINS:
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

            if domain in DOMAINS:
                return domain, 0.95  # type: ignore

        except Exception:
            pass

        return None, 0.0
