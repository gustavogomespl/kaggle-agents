"""
Kaggle Search Tool for retrieving notebooks and discussions.

This module provides functionality to search and retrieve state-of-the-art
solutions from Kaggle competitions via the official API and web scraping.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from ..core.config import get_config
from ..core.state import SOTASolution
from ..utils.contamination import (
    code_references_competition,
    is_same_competition_candidate,
    query_references_competition,
)


@dataclass
class NotebookMetadata:
    """Metadata for a Kaggle notebook."""

    ref: str  # notebook reference (username/notebook-slug)
    title: str
    author: str
    total_votes: int
    medal_type: str | None  # gold, silver, bronze
    language: str  # python, r
    competition: str
    url: str


@dataclass
class DiscussionMetadata:
    """Metadata for a Kaggle discussion."""

    id: int
    title: str
    author: str
    total_votes: int
    total_comments: int
    tags: list[str]
    url: str


def _read_downloaded_metadata_refs(notebook_path: Path, field: str) -> list[str]:
    """Read one source-reference list from the pulled kernel metadata.

    Module-level on purpose: tests monkeypatch ``KaggleSearcher`` wholesale,
    so the reader methods must not resolve their implementation through the
    class name.
    """
    metadata_path = notebook_path.parent / "kernel-metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []

    sources = metadata.get(field) or []
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list):
        return []

    refs: list[str] = []
    for source in sources:
        if isinstance(source, str):
            ref = source
        elif isinstance(source, dict):
            ref = next(
                (str(source[key]) for key in ("ref", "slug", "name", "url") if source.get(key)),
                "",
            )
        else:
            ref = ""
        if ref:
            refs.append(ref)
    return refs


class KaggleSearcher:
    """
    Search and retrieve content from Kaggle competitions.

    This class provides methods to:
    - Search for top notebooks in a competition
    - Search for relevant discussions
    - Download notebook source code
    - Extract code snippets and strategies
    """

    def __init__(self):
        """Initialize the Kaggle searcher with API client."""
        # Lazy import: the kaggle package authenticates (and may exit) at import
        # time, which would break test collection / credential-less environments.
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            self.api = KaggleApi()
            self.api.authenticate()
        except (Exception, SystemExit) as e:
            raise RuntimeError("Kaggle API is unavailable or credentials are not configured") from e
        self.config = get_config()

    @staticmethod
    def _get_kernel_attr(kernel: Any, names: list[str], default: Any = None) -> Any:
        """
        Safely pull an attribute from a Kaggle Kernel object handling API field variants.

        Args:
            kernel: Kernel object returned by Kaggle API
            names: Candidate attribute names in priority order
            default: Fallback value

        Returns:
            Attribute value or default if missing
        """
        for name in names:
            if hasattr(kernel, name):
                try:
                    value = getattr(kernel, name)
                except Exception:
                    continue
                if value is not None:
                    return value
        return default

    @classmethod
    def _get_kernel_competitions(cls, kernel: Any) -> list[str]:
        """Return competition data-source refs exposed by Kaggle kernel metadata."""
        sources = cls._get_kernel_attr(
            kernel,
            ["competition_data_sources", "competitionDataSources"],
            [],
        )
        if isinstance(sources, str):
            return [sources] if sources else []
        if not isinstance(sources, (list, tuple)):
            return []

        refs: list[str] = []
        for source in sources:
            if isinstance(source, str):
                ref = source
            elif isinstance(source, dict):
                ref = next(
                    (str(source[key]) for key in ("ref", "slug", "name", "url") if source.get(key)),
                    "",
                )
            else:
                ref = str(cls._get_kernel_attr(source, ["ref", "slug", "name", "url"], ""))
            if ref:
                refs.append(ref)
        return refs

    def search_notebooks(
        self,
        competition: str,
        sort_by: str = "voteCount",
        page_size: int = 20,
        language: str = "python",
    ) -> list[NotebookMetadata]:
        """
        Search for notebooks in a competition.

        Args:
            competition: Competition name/slug
            sort_by: Sort order (voteCount, dateCreated, scoreAscending, scoreDescending)
            page_size: Number of results to return
            language: Programming language filter (python, r, all)

        Returns:
            List of notebook metadata
        """
        try:
            # Use Kaggle API to list kernels
            kernels = self.api.kernels_list(
                competition=competition,
                sort_by=sort_by,
                page_size=min(page_size, 100),  # API limit
                language=language if language != "all" else None,
            )

            notebooks = []
            for kernel in kernels:
                try:
                    ref = self._get_kernel_attr(kernel, ["ref", "slug"])
                    if not ref:
                        continue

                    total_votes = self._get_kernel_attr(
                        kernel,
                        ["total_votes", "totalVotes", "voteCount", "vote_count"],
                        0,
                    )

                    metadata = NotebookMetadata(
                        ref=ref,
                        title=self._get_kernel_attr(kernel, ["title"], ""),
                        author=self._get_kernel_attr(kernel, ["author", "owner"], ""),
                        total_votes=int(total_votes) if total_votes is not None else 0,
                        medal_type=self._get_kernel_attr(kernel, ["medal_type", "medalType"]),
                        language=self._get_kernel_attr(kernel, ["language"], language or "python"),
                        competition=competition,
                        url=f"https://www.kaggle.com/code/{ref}",
                    )
                    notebooks.append(metadata)
                except Exception as kernel_err:
                    print(f"  Skipping kernel due to parse error: {kernel_err}")

            return notebooks[:page_size]

        except Exception as e:
            print(f"  Error searching notebooks: {e}")
            return []

    def download_notebook(
        self,
        notebook_ref: str,
        output_dir: Path | str,
    ) -> Path | None:
        """
        Download notebook source code.

        Args:
            notebook_ref: Notebook reference (username/notebook-slug)
            output_dir: Directory to save the notebook

        Returns:
            Path to downloaded notebook file, or None if failed
        """
        output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Download kernel
            self.api.kernels_pull(
                notebook_ref,
                path=str(output_path),
                metadata=True,
            )

            # Find downloaded file
            notebook_files = list(output_path.glob("*.ipynb")) + list(output_path.glob("*.py"))
            if notebook_files:
                return notebook_files[0]

            return None

        except Exception as e:
            print(f"  Error downloading notebook {notebook_ref}: {e}")
            return None

    @staticmethod
    def read_downloaded_competition_sources(notebook_path: Path) -> list[str]:
        """Read authoritative competition refs written by ``kernels_pull``.

        Cross-competition retrieval must prove that a candidate comes from a
        different competition. Search-result metadata is not consistently
        populated across Kaggle API versions, so the downloaded kernel
        metadata is the final provenance source.
        """
        return _read_downloaded_metadata_refs(
            notebook_path, "competition_sources"
        )

    @staticmethod
    def read_downloaded_dataset_sources(notebook_path: Path) -> list[str]:
        """Read authoritative dataset refs written by ``kernels_pull``.

        Dataset-attached notebooks are a large, legal cross-competition
        population; their pulled dataset refs are the provenance that keeps
        them eligible without weakening target-blindness.
        """
        return _read_downloaded_metadata_refs(
            notebook_path, "dataset_sources"
        )

    def extract_code_from_notebook(self, notebook_path: Path) -> list[str]:
        """
        Extract code cells from a Jupyter notebook.

        Args:
            notebook_path: Path to .ipynb file

        Returns:
            List of code snippets
        """
        try:
            with open(notebook_path, encoding="utf-8") as f:
                notebook_data = json.load(f)

            code_snippets = []
            for cell in notebook_data.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = cell.get("source", [])
                    code = "".join(source) if isinstance(source, list) else source

                    # Skip empty cells and magic commands
                    if code.strip() and not code.strip().startswith("%"):
                        code_snippets.append(code)

            return code_snippets

        except Exception as e:
            print(f"  Error extracting code from {notebook_path}: {e}")
            return []

    def extract_code_from_script(self, script_path: Path) -> list[str]:
        """
        Extract code sections from a Python script.

        Args:
            script_path: Path to .py file

        Returns:
            List of code snippets (split by major sections)
        """
        try:
            with open(script_path, encoding="utf-8") as f:
                content = f.read()

            # Split by major comments (### or more #)
            sections = re.split(r"\n#{3,}.*?\n", content)

            # Filter out empty sections
            return [s.strip() for s in sections if s.strip()]

        except Exception as e:
            print(f"  Error extracting code from {script_path}: {e}")
            return []

    def analyze_notebook_strategies(self, code_snippets: list[str]) -> dict[str, Any]:
        """
        Analyze code to extract ML strategies and approaches.

        Args:
            code_snippets: List of code snippets from notebook

        Returns:
            Dictionary with extracted strategies
        """
        strategies = {
            "models_used": [],
            "feature_engineering": [],
            "ensemble_approach": None,
        }

        all_code = "\n".join(code_snippets)

        # Detect models
        model_patterns = {
            "XGBoost": r"xgboost|XGB|xgb\.train|XGBClassifier|XGBRegressor",
            "LightGBM": r"lightgbm|lgbm|lgb\.train|LGBMClassifier|LGBMRegressor",
            "CatBoost": r"catboost|CatBoost|CatBoostClassifier|CatBoostRegressor",
            "RandomForest": r"RandomForest|RandomForestClassifier|RandomForestRegressor",
            "Neural Network": r"keras|tensorflow|torch|nn\.Module|Sequential",
            "Linear Models": r"LinearRegression|LogisticRegression|Ridge|Lasso",
        }

        for model_name, pattern in model_patterns.items():
            if re.search(pattern, all_code, re.IGNORECASE):
                strategies["models_used"].append(model_name)

        # Detect feature engineering techniques
        feature_patterns = {
            "Target Encoding": r"TargetEncoder|target_encode",
            "One-Hot Encoding": r"OneHotEncoder|get_dummies",
            "Polynomial Features": r"PolynomialFeatures",
            "Feature Scaling": r"StandardScaler|MinMaxScaler|RobustScaler",
            "Feature Selection": r"SelectKBest|RFE|feature_importances",
            "PCA": r"PCA\(",
            "Time Features": r"dt\.year|dt\.month|dt\.day|dt\.hour",
        }

        for feat_name, pattern in feature_patterns.items():
            if re.search(pattern, all_code, re.IGNORECASE):
                strategies["feature_engineering"].append(feat_name)

        # Detect ensemble methods
        if re.search(r"VotingClassifier|VotingRegressor", all_code):
            strategies["ensemble_approach"] = "Voting"
        elif re.search(r"StackingClassifier|StackingRegressor", all_code):
            strategies["ensemble_approach"] = "Stacking"
        elif re.search(r"\.mean\(axis=|average.*predictions", all_code):
            strategies["ensemble_approach"] = "Averaging"
        elif re.search(r"weighted.*mean|weights.*predictions", all_code):
            strategies["ensemble_approach"] = "Weighted Averaging"

        return strategies

    def search_discussions(
        self,
        competition: str,
        max_results: int = 10,
    ) -> list[DiscussionMetadata]:
        """
        Search for discussions in a competition.

        Args:
            competition: Competition name/slug
            max_results: Maximum number of discussions to retrieve

        Returns:
            List of discussion metadata
        """
        discussions = []

        try:
            # Web scraping approach (Kaggle API doesn't provide discussion search)
            url = f"https://www.kaggle.com/competitions/{competition}/discussion"

            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Find discussion elements (this may need adjustment based on Kaggle's HTML structure)
            # Note: This is a simplified example and may need to be updated
            discussion_items = soup.find_all("div", class_="topic-list-item", limit=max_results)

            for item in discussion_items[:max_results]:
                try:
                    # Extract metadata (adjust selectors as needed)
                    title_elem = item.find("a", class_="topic-title")
                    votes_elem = item.find("span", class_="vote-count")

                    if title_elem:
                        discussion = DiscussionMetadata(
                            id=hash(title_elem.get("href", "")),
                            title=title_elem.text.strip(),
                            author="",  # Would need additional parsing
                            total_votes=int(votes_elem.text) if votes_elem else 0,
                            total_comments=0,  # Would need additional parsing
                            tags=[],
                            url=f"https://www.kaggle.com{title_elem.get('href', '')}",
                        )
                        discussions.append(discussion)

                except Exception as e:
                    print(f"  Error parsing discussion item: {e}")
                    continue

        except Exception as e:
            print(f"  Error searching discussions: {e}")

        return discussions

    def create_sota_solution(
        self,
        notebook_metadata: NotebookMetadata,
        code_snippets: list[str],
        strategies: dict[str, Any],
    ) -> SOTASolution:
        """
        Create a SOTASolution object from notebook data.

        Args:
            notebook_metadata: Notebook metadata
            code_snippets: Extracted code snippets
            strategies: Extracted strategies

        Returns:
            SOTASolution object
        """
        return SOTASolution(
            source=notebook_metadata.ref,
            title=notebook_metadata.title,
            score=0.0,  # Score not available from API
            votes=notebook_metadata.total_votes,
            code_snippets=code_snippets,
            strategies=[f"{k}: {v}" for k, v in strategies.items() if v],
            models_used=strategies.get("models_used", []),
            feature_engineering=strategies.get("feature_engineering", []),
            ensemble_approach=strategies.get("ensemble_approach"),
        )


# ==================== Target-Blind Retrieval Guard ====================
# Our evaluation protocol excludes solutions specific to the target task.
# Decision helpers live in utils/contamination.py (pure, unit-testable);
# here they are applied to retrieved notebooks with a full audit trail.


def _search_audit_record(
    record: dict[str, Any],
    *,
    iteration: int | None = None,
    search_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Attach retrieval-attempt context without inventing unavailable fields."""
    enriched = dict(record)
    if iteration is not None:
        enriched["iteration"] = iteration
    if search_attempt_id is not None:
        enriched["search_attempt_id"] = search_attempt_id
    return enriched


def filter_same_competition_candidates(
    candidates: list[NotebookMetadata],
    competition: str,
    competition_aliases: Iterable[str] | None = None,
    *,
    candidate_origins: dict[str, dict[str, Any]] | None = None,
    iteration: int | None = None,
    search_attempt_id: str | None = None,
) -> tuple[list[NotebookMetadata], list[dict]]:
    """
    Split candidates into (kept, audit records), filtering notebooks that look
    like they belong to the target competition.

    Every candidate produces an audit record so target-blind runs can be
    audited.
    """
    kept: list[NotebookMetadata] = []
    audit: list[dict] = []

    for nb in candidates:
        same = is_same_competition_candidate(
            nb.ref,
            nb.title,
            nb.competition,
            competition,
            competition_aliases,
        )
        origin = (candidate_origins or {}).get(nb.ref, {})
        audit.append(
            _search_audit_record(
                {
                    "ref": nb.ref,
                    "title": nb.title,
                    "votes": nb.total_votes,
                    "source_competitions": nb.competition,
                    "query": origin.get("query"),
                    "query_index": origin.get("query_index"),
                    "stage": "metadata",
                    "same_competition": same,
                    "filtered": same,
                    "filter_reason": ("target_competition_metadata" if same else None),
                },
                iteration=iteration,
                search_attempt_id=search_attempt_id,
            )
        )
        if not same:
            kept.append(nb)

    return kept, audit


def search_notebooks_cross_competition(
    competition: str,
    queries: list[str],
    max_notebooks: int = 10,
    min_votes: int = 5,
    competition_aliases: Iterable[str] | None = None,
    *,
    iteration: int | None = None,
    search_attempt_id: str | None = None,
) -> tuple[list[SOTASolution], list[dict]]:
    """
    Target-blind retrieval: search Kaggle notebooks by task/domain queries
    (not by the target identity) and filter anything belonging to the target
    competition itself.

    Args:
        competition: Target competition slug (used only for exclusion)
        queries: Cross-competition search queries (domain/task keywords)
        max_notebooks: Maximum solutions to return
        min_votes: Minimum votes threshold
        competition_aliases: Complete public target titles/aliases to exclude
        iteration: Workflow iteration attached to audit records when available
        search_attempt_id: Stable retrieval-attempt label for audit lineage

    Returns:
        Tuple of (solutions, audit records). Audit records document every
        provider candidate seen and every filter decision.
    """
    audit: list[dict] = []
    safe_queries: list[tuple[int, str]] = []
    for query_index, query in enumerate(queries, 1):
        identifies_target = query_references_competition(
            query,
            competition,
            competition_aliases,
        )
        audit.append(
            _search_audit_record(
                {
                    "query": query,
                    "query_index": query_index,
                    "stage": "query",
                    "same_competition": identifies_target,
                    "filtered": identifies_target,
                    "filter_reason": ("target_competition_query" if identifies_target else None),
                },
                iteration=iteration,
                search_attempt_id=search_attempt_id,
            )
        )
        if not identifies_target:
            safe_queries.append((query_index, query))

    if not safe_queries:
        print("  Contamination guard: no target-blind query remained")
        return [], audit

    try:
        searcher = KaggleSearcher()
    except RuntimeError as e:
        print(f"  Cross-competition search unavailable: {e}")
        audit.append(
            _search_audit_record(
                {
                    "ref": None,
                    "stage": "initialization",
                    "same_competition": False,
                    "filtered": False,
                    "error": str(e),
                },
                iteration=iteration,
                search_attempt_id=search_attempt_id,
            )
        )
        return [], audit

    seen_refs: set[str] = set()
    seen_origins: dict[str, dict[str, Any]] = {}
    candidate_origins: dict[str, dict[str, Any]] = {}
    candidates: list[NotebookMetadata] = []

    for query_index, query in safe_queries:
        try:
            kernels = searcher.api.kernels_list(
                search=query,
                sort_by="voteCount",
                page_size=20,
            )
        except Exception as e:
            print(f"  Error searching notebooks for query '{query}': {e}")
            audit.append(
                _search_audit_record(
                    {
                        "query": query,
                        "query_index": query_index,
                        "stage": "query_execution",
                        "same_competition": False,
                        "filtered": True,
                        "filter_reason": "query_failed",
                        "error": str(e),
                    },
                    iteration=iteration,
                    search_attempt_id=search_attempt_id,
                )
            )
            continue

        for candidate_index, kernel in enumerate(kernels or [], 1):
            ref: str | None = None
            title = ""
            votes: int | None = None
            try:
                raw_ref = KaggleSearcher._get_kernel_attr(kernel, ["ref", "slug"])
                ref = str(raw_ref or "").strip() or None
                title = str(KaggleSearcher._get_kernel_attr(kernel, ["title"], "") or "")
                if not ref:
                    raise ValueError("provider candidate is missing a source reference")

                total_votes = KaggleSearcher._get_kernel_attr(
                    kernel,
                    ["total_votes", "totalVotes", "voteCount", "vote_count"],
                    0,
                )
                votes = int(total_votes) if total_votes is not None else 0
                origin = {
                    "query": query,
                    "query_index": query_index,
                    "candidate_index": candidate_index,
                }

                if ref in seen_refs:
                    first_origin = seen_origins.get(ref, {})
                    audit.append(
                        _search_audit_record(
                            {
                                "stage": "provider_candidate",
                                **origin,
                                "ref": ref,
                                "title": title,
                                "votes": votes,
                                "filtered": True,
                                "filter_reason": "duplicate_source_ref",
                                "provider_decision": "duplicate",
                                "duplicate_of_query": first_origin.get("query"),
                                "duplicate_of_query_index": first_origin.get("query_index"),
                            },
                            iteration=iteration,
                            search_attempt_id=search_attempt_id,
                        )
                    )
                    continue
                seen_refs.add(ref)
                seen_origins[ref] = origin

                if votes < min_votes:
                    audit.append(
                        _search_audit_record(
                            {
                                "stage": "provider_candidate",
                                **origin,
                                "ref": ref,
                                "title": title,
                                "votes": votes,
                                "filtered": True,
                                "filter_reason": "below_min_votes",
                                "provider_decision": "below_min_votes",
                                "min_votes": min_votes,
                            },
                            iteration=iteration,
                            search_attempt_id=search_attempt_id,
                        )
                    )
                    continue

                candidates.append(
                    NotebookMetadata(
                        ref=ref,
                        title=title,
                        author=KaggleSearcher._get_kernel_attr(
                            kernel,
                            ["author", "owner"],
                            "",
                        ),
                        total_votes=votes,
                        medal_type=KaggleSearcher._get_kernel_attr(
                            kernel,
                            ["medal_type", "medalType"],
                        ),
                        language=KaggleSearcher._get_kernel_attr(
                            kernel,
                            ["language"],
                            "python",
                        ),
                        competition=" ".join(KaggleSearcher._get_kernel_competitions(kernel)),
                        url=f"https://www.kaggle.com/code/{ref}",
                    )
                )
                candidate_origins[ref] = origin
                audit.append(
                    _search_audit_record(
                        {
                            "stage": "provider_candidate",
                            **origin,
                            "ref": ref,
                            "title": title,
                            "votes": votes,
                            "filtered": False,
                            "filter_reason": None,
                            "provider_decision": "queued_for_metadata_guard",
                        },
                        iteration=iteration,
                        search_attempt_id=search_attempt_id,
                    )
                )
            except Exception as kernel_err:
                print(f"  Skipping kernel due to parse error: {kernel_err}")
                audit.append(
                    _search_audit_record(
                        {
                            "stage": "provider_candidate",
                            "query": query,
                            "query_index": query_index,
                            "candidate_index": candidate_index,
                            "ref": ref,
                            "title": title,
                            "votes": votes,
                            "filtered": True,
                            "filter_reason": "candidate_parse_error",
                            "provider_decision": "parse_error",
                            "error": (f"{type(kernel_err).__name__}: {kernel_err}")[:300],
                        },
                        iteration=iteration,
                        search_attempt_id=search_attempt_id,
                    )
                )

    candidates.sort(key=lambda nb: nb.total_votes, reverse=True)

    kept, metadata_audit = filter_same_competition_candidates(
        candidates,
        competition,
        competition_aliases,
        candidate_origins=candidate_origins,
        iteration=iteration,
        search_attempt_id=search_attempt_id,
    )
    audit.extend(metadata_audit)
    print(
        f"  Contamination guard: {len(candidates)} candidates, "
        f"{len(candidates) - len(kept)} filtered at metadata stage"
    )

    solutions: list[SOTASolution] = []
    for nb in kept:
        if len(solutions) >= max_notebooks:
            origin = candidate_origins.get(nb.ref, {})
            audit.append(
                _search_audit_record(
                    {
                        "ref": nb.ref,
                        "title": nb.title,
                        "query": origin.get("query"),
                        "query_index": origin.get("query_index"),
                        "stage": "selection",
                        "same_competition": False,
                        "filtered": True,
                        "filter_reason": "not_selected_top_k_budget",
                        "provider_decision": "not_selected_top_k_budget",
                    },
                    iteration=iteration,
                    search_attempt_id=search_attempt_id,
                )
            )
            continue

        origin = candidate_origins.get(nb.ref, {})

        def source_record(
            record: dict[str, Any],
            *,
            ref: str = nb.ref,
            title: str = nb.title,
            source_query: str | None = origin.get("query"),
            source_query_index: int | None = origin.get("query_index"),
        ) -> dict[str, Any]:
            """Attach the provider query and retrieval-attempt lineage."""
            return _search_audit_record(
                {
                    "ref": ref,
                    "title": title,
                    "query": source_query,
                    "query_index": source_query_index,
                    **record,
                },
                iteration=iteration,
                search_attempt_id=search_attempt_id,
            )

        # Raw third-party notebooks are inspection artifacts, not runtime
        # dependencies.  Keeping them in the repository-wide cache would make a
        # rejected target notebook readable by later generated code.  A
        # per-candidate TemporaryDirectory guarantees cleanup on every return,
        # rejection, parse error, and successful extraction.
        with tempfile.TemporaryDirectory(
            prefix="kaggle_agents_cross_comp_"
        ) as temporary_download:
            nb_path = searcher.download_notebook(
                nb.ref,
                Path(temporary_download),
            )
            if not nb_path:
                audit.append(
                    source_record(
                        {
                            "stage": "download",
                            "same_competition": False,
                            "filtered": True,
                            "filter_reason": "download_failed",
                            "error": "download_failed",
                        }
                    )
                )
                continue

            downloaded_competitions = searcher.read_downloaded_competition_sources(nb_path)
            # Fail closed: only ``kernel-metadata.json`` written by kernels_pull is
            # authoritative provenance. Search-result metadata can be missing,
            # stale, or inconsistent and must never make a candidate eligible.
            source_competitions = downloaded_competitions
            provenance_kind = "competition_sources"
            if not source_competitions:
                dataset_sources = searcher.read_downloaded_dataset_sources(nb_path)
                if not dataset_sources:
                    audit.append(
                        source_record(
                            {
                                "stage": "provenance",
                                "same_competition": False,
                                "filtered": True,
                                "filter_reason": "unverified_source_competition",
                                "source_competitions": [],
                            }
                        )
                    )
                    continue
                # Dataset-attached notebooks stay eligible only when no
                # attached dataset matches the target identity — target-data
                # mirrors are typically republished under the competition's
                # own name, which the identity aliases catch — and the
                # full-source contamination scan below still applies.
                if any(
                    is_same_competition_candidate(
                        "",
                        "",
                        dataset_ref,
                        competition,
                        competition_aliases,
                    )
                    for dataset_ref in dataset_sources
                ):
                    audit.append(
                        source_record(
                            {
                                "stage": "provenance",
                                "same_competition": True,
                                "filtered": True,
                                "filter_reason": "target_competition_dataset_source",
                                "dataset_sources": dataset_sources,
                            }
                        )
                    )
                    continue
                provenance_kind = "dataset_sources"
            if any(
                is_same_competition_candidate(
                    "",
                    "",
                    source,
                    competition,
                    competition_aliases,
                )
                for source in source_competitions
            ):
                audit.append(
                    source_record(
                        {
                            "stage": "provenance",
                            "same_competition": True,
                            "filtered": True,
                            "filter_reason": "target_competition_download_metadata",
                            "source_competitions": source_competitions,
                        }
                    )
                )
                continue

            # Scan the complete source document first, including notebook markdown
            # and data-source metadata. Code-only scanning misses target references
            # written outside executable cells.
            try:
                source_bytes = nb_path.read_bytes()
                source_document = source_bytes.decode("utf-8", errors="replace")
                source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            except OSError as e:
                audit.append(
                    source_record(
                        {
                            "stage": "source_read",
                            "same_competition": False,
                            "filtered": True,
                            "filter_reason": "source_read_failed",
                            "error": str(e),
                        }
                    )
                )
                continue

            if nb_path.suffix == ".ipynb":
                code_snippets = searcher.extract_code_from_notebook(nb_path)
            else:
                code_snippets = searcher.extract_code_from_script(nb_path)

            if not code_snippets:
                audit.append(
                    source_record(
                        {
                            "stage": "source_parse",
                            "same_competition": False,
                            "filtered": True,
                            "filter_reason": "source_parse_failed_or_empty",
                            "source_sha256": source_sha256,
                        }
                    )
                )
                continue

            # Second-stage filter: notebook code that reads the target competition's
            # input data is competition-specific -> discard.
            if code_references_competition(
                source_document,
                competition,
                competition_aliases,
            ):
                audit.append(
                    source_record(
                        {
                            "stage": "code_scan",
                            "same_competition": True,
                            "filtered": True,
                            "filter_reason": "target_competition_source_reference",
                            "source_sha256": source_sha256,
                        }
                    )
                )
                print(
                    f"  Contamination guard: filtered {nb.ref} "
                    f"(code references {competition})"
                )
                continue

            strategies = searcher.analyze_notebook_strategies(code_snippets)
            solution = searcher.create_sota_solution(
                nb,
                code_snippets,
                strategies,
            )
            solution.source_sha256 = source_sha256
            solutions.append(solution)
            audit.append(
                source_record(
                    {
                        "stage": "code_scan",
                        "same_competition": False,
                        "filtered": False,
                        "filter_reason": None,
                        "source_sha256": source_sha256,
                        "source_competitions": source_competitions,
                        "provenance_kind": provenance_kind,
                    }
                )
            )

        # Rate limiting
        time.sleep(1)

    print(f"  Contamination guard: {len(solutions)} cross-competition solutions retained")
    return solutions, audit


# ==================== Convenience Functions ====================


def search_competition_notebooks(
    competition: str,
    max_notebooks: int = 10,
    min_votes: int = 5,
) -> list[SOTASolution]:
    """
    Search and analyze top notebooks for a competition.

    Args:
        competition: Competition name
        max_notebooks: Maximum number of notebooks to analyze
        min_votes: Minimum votes threshold

    Returns:
        List of SOTASolution objects
    """
    try:
        searcher = KaggleSearcher()
    except RuntimeError as e:
        print(f"Search unavailable: {e}")
        return []

    # Search notebooks
    print(f"Searching notebooks for {competition}...")
    notebooks = searcher.search_notebooks(competition, page_size=max_notebooks * 2)

    # Filter by votes
    notebooks = [nb for nb in notebooks if nb.total_votes >= min_votes][:max_notebooks]

    print(f"= Found {len(notebooks)} high-quality notebooks")

    # Download and analyze
    solutions = []
    config = get_config()
    download_dir = config.paths.cache_dir / "notebooks" / competition

    for nb in notebooks:
        print(f"  = Analyzing: {nb.title} ({nb.total_votes} votes)")

        # Download notebook (per-ref subdir: download_notebook globs the dir and
        # returns the first file, so a shared dir would return the wrong notebook)
        nb_path = searcher.download_notebook(nb.ref, download_dir / nb.ref.replace("/", "__"))
        if not nb_path:
            continue

        # Extract code
        if nb_path.suffix == ".ipynb":
            code_snippets = searcher.extract_code_from_notebook(nb_path)
        else:
            code_snippets = searcher.extract_code_from_script(nb_path)

        # Analyze strategies
        strategies = searcher.analyze_notebook_strategies(code_snippets)

        # Create SOTA solution
        solution = searcher.create_sota_solution(nb, code_snippets, strategies)
        solutions.append(solution)

        # Rate limiting
        time.sleep(1)

    print(f" Successfully analyzed {len(solutions)} notebooks")
    return solutions
