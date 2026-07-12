"""
MLE-bench contamination guard - pure decision helpers.

MLE-bench forbids using solutions specific to the target competition.
These helpers decide whether a retrieved Kaggle notebook belongs to the
target competition so it can be filtered out (and audited) in mlebench mode.

Kept free of heavy imports (no Kaggle SDK) so it is unit-testable and safe
to import anywhere.
"""

from __future__ import annotations

import re


# Generic tokens that appear in many competition slugs and would cause
# false positives when matching notebook titles against slug tokens.
COMPETITION_TOKEN_STOPWORDS = {
    "challenge",
    "challenges",
    "competition",
    "classification",
    "detection",
    "identification",
    "prediction",
    "predict",
    "kaggle",
    "series",
    "playground",
    "tabular",
    "edition",
    "kernels",
    "redux",
    "with",
    "the",
    "and",
    "for",
    "language",
    "english",
    "russian",
    "2013",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
}


def competition_slug_tokens(competition: str) -> set[str]:
    """
    Distinctive tokens of a competition slug (stopwords and short tokens removed).

    Example: "aerial-cactus-identification" -> {"aerial", "cactus"}
    """
    tokens = re.split(r"[-_\s]+", competition.lower())
    return {t for t in tokens if len(t) >= 4 and t not in COMPETITION_TOKEN_STOPWORDS}


def looks_like_same_competition(text: str | None, competition: str) -> bool:
    """
    Heuristic: does this text (notebook ref/title) reference the target competition?

    Conservative by design (benchmark compliance prefers false positives):
    - full slug appears in the text (also matched with spaces/underscores), OR
    - at least 2 distinctive slug tokens appear (1 suffices when the slug only
      has a single distinctive token).
    """
    if not text or not competition:
        return False

    text_l = text.lower()
    slug = competition.lower()
    if slug in text_l or slug.replace("-", " ") in text_l or slug.replace("-", "_") in text_l:
        return True

    tokens = competition_slug_tokens(competition)
    if not tokens:
        return False

    hits = sum(1 for t in tokens if t in text_l)
    required = min(2, len(tokens))
    return hits >= required


def code_references_competition(code_text: str | None, competition: str) -> bool:
    """
    High-precision check on downloaded notebook code: kernels attached to a
    competition read data from ``../input/<slug>`` or reference the slug/URL.
    """
    if not code_text or not competition:
        return False

    code_l = code_text.lower()
    slug = competition.lower()
    patterns = (
        slug,
        f"input/{slug}",
        f"competitions/{slug}",
    )
    return any(p in code_l for p in patterns)


def is_same_competition_candidate(
    ref: str,
    title: str,
    candidate_competition: str,
    target_competition: str,
) -> bool:
    """
    Decide (from metadata only) whether a retrieved notebook belongs to the
    target competition. Used as the first filter stage; downloaded code is
    checked afterwards with :func:`code_references_competition`.
    """
    if candidate_competition and looks_like_same_competition(
        candidate_competition, target_competition
    ):
        return True
    return looks_like_same_competition(f"{ref} {title}", target_competition)
