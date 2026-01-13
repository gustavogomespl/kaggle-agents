"""
Signal fusion and result refinement for domain detection.

Contains methods for fusing signals, refining domains with metrics, and logging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import DOMAINS, SOURCE_WEIGHTS
from .dataclasses import DetectionSignal


if TYPE_CHECKING:
    pass


class SignalFusionMixin:
    """Mixin providing signal fusion, logging, and refinement methods."""

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
        valid_domains = [domain for domain in domain_scores if domain in DOMAINS]
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
