"""
Text and tabular detection methods.

Contains methods for detecting NLP and tabular domains from CSV data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd


if TYPE_CHECKING:
    from ...core.state import CompetitionInfo, DomainType


class TextTabularDetectionMixin:
    """Mixin providing text and tabular detection methods."""

    def _detect_text_from_csv(
        self, competition_info: CompetitionInfo, data_dir: Path
    ) -> tuple[DomainType, float] | None:
        """
        Detect NLP/text domain from CSV files containing text columns.

        This should be called BEFORE tabular detection to correctly classify
        competitions like Spooky Author Identification as text_classification.

        Signals:
        1. Column named "text", "sentence", "content", "body", "comment", "review"
        2. Object columns with average string length > 100 (prose/paragraphs)
        3. Competition name/description contains NLP keywords

        Args:
            competition_info: Competition metadata
            data_dir: Path to data directory

        Returns:
            Tuple of (domain, confidence) if text detected, None otherwise
        """
        train_path = self._find_train_file(data_dir)
        if not train_path:
            return None

        try:
            if train_path.suffix.lower() == ".parquet":
                if train_path.stat().st_size > 200 * 1024 * 1024:
                    return None
                df = pd.read_parquet(train_path)
            else:
                df = pd.read_csv(train_path, nrows=20)
        except Exception:
            return None

        if df.empty:
            return None

        # Block text detection if strong numeric/tabular signal exists
        # This prevents false-positives on competitions like NYC taxi fare
        # where datetime string columns trigger text detection
        numeric_cols = df.select_dtypes(include="number").columns
        numeric_ratio = len(numeric_cols) / max(len(df.columns), 1)

        if len(numeric_cols) >= 5 and numeric_ratio > 0.5:
            print(f"   Skipping text detection: {len(numeric_cols)} numeric cols ({numeric_ratio:.0%})")
            return None

        # Signal 1: Well-known text column names
        text_column_names = {
            "text", "sentence", "content", "body", "comment", "review",
            "message", "description", "title", "question", "answer",
            "abstract", "article", "excerpt", "passage", "document",
        }
        cols_lower = {c.lower(): c for c in df.columns}

        found_text_col = None
        for text_name in text_column_names:
            if text_name in cols_lower:
                found_text_col = cols_lower[text_name]
                break

        # Signal 2: Object columns with long average string length
        long_text_col = None
        object_cols = df.select_dtypes(include="object").columns
        for col in object_cols:
            try:
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > 100:  # Average length > 100 chars = prose/paragraphs
                    long_text_col = col
                    break
            except Exception:
                continue

        # Signal 3: Competition name/description contains NLP keywords
        nlp_keywords = {
            "text", "nlp", "sentiment", "author", "classification",
            "toxic", "spam", "review", "language", "translation",
        }
        name_lower = (competition_info.name or "").lower()
        desc_lower = (competition_info.description or "").lower()[:500]
        nlp_keyword_match = any(kw in name_lower or kw in desc_lower for kw in nlp_keywords)

        # Calculate confidence based on signals
        confidence = 0.0
        signals_found = []

        if found_text_col:
            confidence += 0.40
            signals_found.append(f"text column '{found_text_col}'")

        if long_text_col:
            confidence += 0.35
            signals_found.append(f"long text in '{long_text_col}'")

        if nlp_keyword_match:
            confidence += 0.15
            signals_found.append("NLP keywords in name/description")

        # Need at least one strong signal
        if confidence >= 0.35:
            is_regression = "regression" in (competition_info.problem_type or "").lower()
            domain: DomainType = "text_regression" if is_regression else "text_classification"
            final_conf = min(0.92, confidence)

            print(f"   📝 Text domain detected: {', '.join(signals_found)} (conf={final_conf:.2f})")
            return (domain, final_conf)

        return None

    def _detect_tabular_from_csv(
        self, competition_info: CompetitionInfo, data_dir: Path
    ) -> tuple[DomainType, float] | None:
        """
        Heuristic override: if train.csv is wide with many numeric columns,
        treat the competition as tabular when no strong image signal exists.
        """
        if self._has_image_assets(data_dir):
            return None

        # Also block tabular if audio files exist
        if self._has_audio_assets(data_dir):
            print("   Audio assets detected - skipping tabular override")
            return None

        # Block tabular if text domain detected (NLP competitions in CSV format)
        text_result = self._detect_text_from_csv(competition_info, data_dir)
        if text_result:
            return None  # Let text detection take precedence

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
