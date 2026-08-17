"""
Target-blind retrieval guard - pure decision helpers.

This project's MLE-bench protocol keeps external retrieval active while
excluding sources that identify the target competition. These helpers enforce
that boundary and make every decision auditable.

Kept free of heavy imports (no Kaggle SDK) so it is unit-testable and safe
to import anywhere.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from typing import Any


_STRUCTURAL_DESCRIPTION_HEADINGS = {
    "acknowledgements",
    "citation",
    "data",
    "dataset",
    "description",
    "evaluation",
    "metric",
    "overview",
    "rules",
    "submission",
    "submission format",
    "task",
    "timeline",
}


def _normalize_identity(value: str) -> str:
    """Normalize a slug/title for exact phrase matching."""
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _clean_public_title(value: str) -> str:
    """Remove presentation-only Markdown/HTML from a public title."""
    title = html.unescape(value).strip()
    title = re.sub(r"<[^>]+>", " ", title)
    title = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", title)
    title = re.sub(r"\[([^\]]+)]\([^)]*\)", r"\1", title)
    title = re.sub(r"^[#=\-\s]+|[#=\-\s]+$", "", title)
    title = re.sub(r"[*_`~]+", "", title)
    return " ".join(title.split())


def derive_competition_identity_aliases(
    competition: str,
    description: str | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Derive a small, auditable target-identity set from public metadata.

    The opaque competition slug is always retained in full.  At most one
    official-looking title is added, and only when it is explicitly present as
    a public document title (YAML ``title``, HTML ``h1``, Markdown ``h1``, or
    Setext ``h1``).  Narrative text and individual title words are never
    converted into aliases.
    """
    aliases: list[str] = []
    evidence: list[dict[str, Any]] = []
    normalized_seen: set[str] = set()

    def add(alias: str, source: str, line: int | None = None) -> None:
        cleaned = _clean_public_title(alias)
        normalized = _normalize_identity(cleaned)
        if not normalized or normalized in normalized_seen:
            return
        normalized_seen.add(normalized)
        aliases.append(cleaned)
        record: dict[str, Any] = {
            "alias": cleaned,
            "normalized_alias": normalized,
            "source": source,
        }
        if line is not None:
            record["line"] = line
        evidence.append(record)

    add(competition, "competition_slug")
    if not description:
        return aliases, evidence

    lines = description.splitlines()
    title_candidate: tuple[str, str, int] | None = None

    # A front-matter title is the most explicit document-title signal.
    if lines and lines[0].strip() == "---":
        for line_number, line in enumerate(lines[1:], start=2):
            if line.strip() == "---":
                break
            match = re.match(r"^\s*title\s*:\s*(.+?)\s*$", line, re.IGNORECASE)
            if match:
                title_candidate = (
                    match.group(1).strip(" '\""),
                    "public_description_frontmatter_title",
                    line_number,
                )
                break

    if title_candidate is None:
        # HTML and Markdown H1s are explicit public headings.  Only the first
        # document-level H1 is eligible: later headings or fenced examples
        # cannot introduce new target identities.
        in_fence = False
        for line_number, line in enumerate(lines, start=1):
            if re.match(r"^\s*(```|~~~)", line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            html_match = re.search(
                r"<h1(?:\s[^>]*)?>(.*?)</h1>",
                line,
                re.IGNORECASE,
            )
            if html_match:
                title_candidate = (
                    html_match.group(1),
                    "public_description_html_h1",
                    line_number,
                )
                break
            markdown_match = re.match(r"^\s*#(?!#)\s+(.+?)\s*#*\s*$", line)
            if markdown_match:
                title_candidate = (
                    markdown_match.group(1),
                    "public_description_markdown_h1",
                    line_number,
                )
                break
            if (
                line_number < len(lines)
                and line.strip()
                and re.match(r"^\s*=+\s*$", lines[line_number])
            ):
                title_candidate = (
                    line,
                    "public_description_setext_h1",
                    line_number,
                )
                break

    if title_candidate is not None:
        candidate, source, line_number = title_candidate
        cleaned = _clean_public_title(candidate)
        # Section labels are not task identities.  This generic structural
        # deny-list prevents broad terms such as "Overview" from becoming
        # filters; no task-specific names are encoded here.
        if _normalize_identity(cleaned) not in _STRUCTURAL_DESCRIPTION_HEADINGS:
            add(cleaned, source, line_number)

    return aliases, evidence


def _competition_identities(
    competition: str,
    aliases: Iterable[str] | None = None,
) -> list[str]:
    """Return complete, normalized identities without token expansion."""
    identities: list[str] = []
    seen: set[str] = set()
    alias_values: Iterable[str]
    if isinstance(aliases, str):
        alias_values = (aliases,)
    else:
        alias_values = aliases or ()
    for value in (competition, *alias_values):
        if value is None:
            continue
        normalized = _normalize_identity(str(value))
        if normalized and normalized not in seen:
            identities.append(normalized)
            seen.add(normalized)
    return identities


def references_competition_identity(
    text: str | None,
    competition: str,
    aliases: Iterable[str] | None = None,
) -> bool:
    """Match any complete normalized target identity inside external text."""
    if not text or not competition:
        return False
    normalized_text = _normalize_identity(text)
    padded_text = f" {normalized_text} "
    return any(
        f" {identity} " in padded_text
        for identity in _competition_identities(competition, aliases)
    )


def query_references_competition(
    query: str | None,
    competition: str,
    aliases: Iterable[str] | None = None,
) -> bool:
    """Return whether an external-search query identifies the target task.

    The query is the first contamination boundary: once it is sent to a search
    provider, filtering the returned notebooks is too late. Query filtering is
    deliberately high precision, however: generic task words such as "text
    normalization" must remain searchable even when they overlap a target
    slug.
    """
    return references_competition_identity(query, competition, aliases)


def code_references_competition(
    code_text: str | None,
    competition: str,
    aliases: Iterable[str] | None = None,
) -> bool:
    """
    Check the complete downloaded source for references to the target task.

    Notebook metadata and markdown can spell a competition slug with spaces or
    underscores even when executable cells do not contain ``../input/<slug>``.
    Complete normalized-identity matching covers those forms without rejecting
    generic domain concepts shared by unrelated competitions.
    """
    if not code_text or not competition:
        return False

    return references_competition_identity(
        code_text,
        competition,
        aliases,
    )


def is_same_competition_candidate(
    ref: str,
    title: str,
    candidate_competition: str,
    target_competition: str,
    target_aliases: Iterable[str] | None = None,
) -> bool:
    """
    Decide (from metadata only) whether a retrieved notebook belongs to the
    target competition. Used as the first filter stage; downloaded code is
    checked afterwards with :func:`code_references_competition`.
    """
    if candidate_competition and references_competition_identity(
        candidate_competition, target_competition, target_aliases
    ):
        return True
    return references_competition_identity(
        f"{ref} {title}",
        target_competition,
        target_aliases,
    )
