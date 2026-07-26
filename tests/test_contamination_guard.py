"""Tests for target-blind external retrieval."""

from types import SimpleNamespace

import pytest

from kaggle_agents.utils.contamination import (
    code_references_competition,
    derive_competition_identity_aliases,
    is_same_competition_candidate,
    query_references_competition,
    references_competition_identity,
)


def test_cross_competition_search_degrades_gracefully_without_kaggle_auth(monkeypatch):
    from kaggle_agents.tools import kaggle_search

    class UnauthenticatedSearcher:
        def __init__(self):
            raise RuntimeError("Kaggle API credentials are not configured")

    monkeypatch.setattr(kaggle_search, "KaggleSearcher", UnauthenticatedSearcher)

    solutions, audit = kaggle_search.search_notebooks_cross_competition(
        "leaf-classification", ["generic classification"], max_notebooks=1
    )

    assert solutions == []
    initialization = next(
        record for record in audit if record["stage"] == "initialization"
    )
    assert "credentials" in initialization["error"]


def test_target_identifying_query_is_never_sent_to_provider(monkeypatch):
    from kaggle_agents.tools import kaggle_search

    calls: list[str] = []

    class FakeApi:
        def kernels_list(self, **kwargs):
            calls.append(kwargs["search"])
            return []

    class FakeSearcher:
        def __init__(self):
            self.api = FakeApi()

    monkeypatch.setattr(kaggle_search, "KaggleSearcher", FakeSearcher)

    solutions, audit = kaggle_search.search_notebooks_cross_competition(
        "aerial-cactus-identification",
        [
            "aerial cactus identification winning solution",
            "generic image classification transfer learning",
        ],
    )

    assert solutions == []
    assert calls == ["generic image classification transfer learning"]
    blocked = next(
        record
        for record in audit
        if record.get("filter_reason") == "target_competition_query"
    )
    assert blocked["query"] == "aerial cactus identification winning solution"
    assert blocked["filtered"] is True


def test_all_target_queries_fail_closed_before_provider_initialization(monkeypatch):
    from kaggle_agents.tools import kaggle_search

    class MustNotInitialize:
        def __init__(self):
            raise AssertionError("provider must not be initialized")

    monkeypatch.setattr(kaggle_search, "KaggleSearcher", MustNotInitialize)

    solutions, audit = kaggle_search.search_notebooks_cross_competition(
        "leaf-classification",
        ["leaf classification top notebook"],
    )

    assert solutions == []
    assert len(audit) == 1
    assert audit[0]["filter_reason"] == "target_competition_query"


def test_cross_competition_search_rejects_unparseable_source(
    temp_data_dir, monkeypatch
):
    from kaggle_agents.tools import kaggle_search

    invalid_notebook = temp_data_dir / "invalid.ipynb"
    invalid_notebook.write_text("not-json", encoding="utf-8")
    (temp_data_dir / "kernel-metadata.json").write_text(
        '{"competition_sources": ["other-classification"]}',
        encoding="utf-8",
    )

    class FakeApi:
        def kernels_list(self, **_kwargs):
            return [
                SimpleNamespace(
                    ref="user/generic-model",
                    title="Generic model",
                    author="user",
                    total_votes=10,
                    language="python",
                    competition_data_sources=["other-classification"],
                )
            ]

    class FakeSearcher:
        _get_kernel_attr = staticmethod(kaggle_search.KaggleSearcher._get_kernel_attr)
        _get_kernel_competitions = classmethod(
            kaggle_search.KaggleSearcher._get_kernel_competitions.__func__
        )
        read_downloaded_competition_sources = staticmethod(
            kaggle_search.KaggleSearcher.read_downloaded_competition_sources
        )
        read_downloaded_dataset_sources = staticmethod(
            kaggle_search.KaggleSearcher.read_downloaded_dataset_sources
        )

        def __init__(self):
            self.api = FakeApi()

        def download_notebook(self, *_args, **_kwargs):
            return invalid_notebook

        def extract_code_from_notebook(self, _path):
            return []

        def extract_code_from_script(self, _path):
            return []

    monkeypatch.setattr(kaggle_search, "KaggleSearcher", FakeSearcher)

    solutions, audit = kaggle_search.search_notebooks_cross_competition(
        "leaf-classification", ["generic classification"], max_notebooks=1
    )

    assert solutions == []
    rejected = next(record for record in audit if record["stage"] == "source_parse")
    assert rejected["filtered"] is True
    assert rejected["filter_reason"] == "source_parse_failed_or_empty"
    assert len(rejected["source_sha256"]) == 64


def test_cross_competition_search_rejects_unverified_notebook_provenance(
    temp_data_dir, monkeypatch
):
    from kaggle_agents.tools import kaggle_search

    notebook = temp_data_dir / "generic.py"
    notebook.write_text("print('generic model')", encoding="utf-8")

    class FakeApi:
        def kernels_list(self, **_kwargs):
            return [
                SimpleNamespace(
                    ref="user/generic-model",
                    title="Generic model",
                    author="user",
                    total_votes=10,
                    language="python",
                    competition_data_sources=[],
                )
            ]

    class FakeSearcher:
        _get_kernel_attr = staticmethod(kaggle_search.KaggleSearcher._get_kernel_attr)
        _get_kernel_competitions = classmethod(
            kaggle_search.KaggleSearcher._get_kernel_competitions.__func__
        )
        read_downloaded_competition_sources = staticmethod(
            kaggle_search.KaggleSearcher.read_downloaded_competition_sources
        )
        read_downloaded_dataset_sources = staticmethod(
            kaggle_search.KaggleSearcher.read_downloaded_dataset_sources
        )

        def __init__(self):
            self.api = FakeApi()

        def download_notebook(self, *_args, **_kwargs):
            return notebook

    monkeypatch.setattr(kaggle_search, "KaggleSearcher", FakeSearcher)

    solutions, audit = kaggle_search.search_notebooks_cross_competition(
        "leaf-classification", ["generic classification"], max_notebooks=1
    )

    assert solutions == []
    rejected = next(record for record in audit if record["stage"] == "provenance")
    assert rejected["filter_reason"] == "unverified_source_competition"


def test_cross_competition_search_rechecks_downloaded_competition_metadata(
    temp_data_dir, monkeypatch
):
    from kaggle_agents.tools import kaggle_search

    notebook = temp_data_dir / "generic.py"
    notebook.write_text("print('generic model')", encoding="utf-8")

    class FakeApi:
        def kernels_list(self, **_kwargs):
            return [
                SimpleNamespace(
                    ref="user/generic-model",
                    title="Generic model",
                    author="user",
                    total_votes=10,
                    language="python",
                    competition_data_sources=["apparently-other-task"],
                )
            ]

    class FakeSearcher:
        _get_kernel_attr = staticmethod(kaggle_search.KaggleSearcher._get_kernel_attr)
        _get_kernel_competitions = classmethod(
            kaggle_search.KaggleSearcher._get_kernel_competitions.__func__
        )

        def __init__(self):
            self.api = FakeApi()

        def download_notebook(self, *_args, **_kwargs):
            return notebook

        def read_downloaded_competition_sources(self, _path):
            return ["leaf-classification"]

    monkeypatch.setattr(kaggle_search, "KaggleSearcher", FakeSearcher)

    solutions, audit = kaggle_search.search_notebooks_cross_competition(
        "leaf-classification", ["generic classification"], max_notebooks=1
    )

    assert solutions == []
    rejected = next(record for record in audit if record["stage"] == "provenance")
    assert rejected["filter_reason"] == "target_competition_download_metadata"


def test_rejected_target_download_is_ephemeral_and_removed(
    tmp_path,
    monkeypatch,
):
    """A target notebook inspected for provenance must not remain readable."""
    from kaggle_agents.tools import kaggle_search

    download_roots = []

    class FakeApi:
        def kernels_list(self, **_kwargs):
            return [
                SimpleNamespace(
                    ref="user/metadata-ambiguous-model",
                    title="Generic model",
                    author="user",
                    total_votes=10,
                    language="python",
                    competition_data_sources=[],
                )
            ]

    class FakeSearcher:
        _get_kernel_attr = staticmethod(
            kaggle_search.KaggleSearcher._get_kernel_attr
        )
        _get_kernel_competitions = classmethod(
            kaggle_search.KaggleSearcher._get_kernel_competitions.__func__
        )
        read_downloaded_competition_sources = staticmethod(
            kaggle_search.KaggleSearcher.read_downloaded_competition_sources
        )
        read_downloaded_dataset_sources = staticmethod(
            kaggle_search.KaggleSearcher.read_downloaded_dataset_sources
        )

        def __init__(self):
            self.api = FakeApi()

        def download_notebook(self, _ref, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            download_roots.append(output_dir)
            notebook = output_dir / "candidate.py"
            notebook.write_text("model = GenericClassifier()", encoding="utf-8")
            (output_dir / "kernel-metadata.json").write_text(
                '{"competition_sources": ["opaque-target"]}',
                encoding="utf-8",
            )
            return notebook

    monkeypatch.setattr(kaggle_search, "KaggleSearcher", FakeSearcher)

    solutions, audit = kaggle_search.search_notebooks_cross_competition(
        "opaque-target",
        ["generic classification"],
        max_notebooks=1,
    )

    assert solutions == []
    assert download_roots
    assert all(not path.exists() for path in download_roots)
    assert any(
        record.get("filter_reason") == "target_competition_download_metadata"
        for record in audit
    )


@pytest.mark.parametrize("metadata_payload", [None, {"competition_sources": []}])
def test_search_result_cannot_substitute_for_downloaded_provenance(
    tmp_path,
    monkeypatch,
    metadata_payload,
):
    """Claiming another source in search results is insufficient evidence."""
    from kaggle_agents.tools import kaggle_search

    download_dir = tmp_path / "download"
    download_dir.mkdir()
    notebook = download_dir / "generic.py"
    notebook.write_text("print('generic model')", encoding="utf-8")
    if metadata_payload is not None:
        (download_dir / "kernel-metadata.json").write_text(
            __import__("json").dumps(metadata_payload),
            encoding="utf-8",
        )

    class FakeApi:
        def kernels_list(self, **_kwargs):
            return [
                SimpleNamespace(
                    ref="user/generic-model",
                    title="Generic model",
                    author="user",
                    total_votes=10,
                    language="python",
                    competition_data_sources=["definitely-another-competition"],
                )
            ]

    class FakeSearcher:
        _get_kernel_attr = staticmethod(kaggle_search.KaggleSearcher._get_kernel_attr)
        _get_kernel_competitions = classmethod(
            kaggle_search.KaggleSearcher._get_kernel_competitions.__func__
        )
        read_downloaded_competition_sources = staticmethod(
            kaggle_search.KaggleSearcher.read_downloaded_competition_sources
        )
        read_downloaded_dataset_sources = staticmethod(
            kaggle_search.KaggleSearcher.read_downloaded_dataset_sources
        )

        def __init__(self):
            self.api = FakeApi()

        def download_notebook(self, *_args, **_kwargs):
            return notebook

    monkeypatch.setattr(kaggle_search, "KaggleSearcher", FakeSearcher)

    solutions, audit = kaggle_search.search_notebooks_cross_competition(
        "opaque-target-slug",
        ["generic classification"],
        max_notebooks=1,
    )

    assert solutions == []
    rejected = next(record for record in audit if record["stage"] == "provenance")
    assert rejected["filter_reason"] == "unverified_source_competition"
    assert rejected["source_competitions"] == []


def test_cross_competition_search_keeps_verified_other_competition(
    tmp_path,
    monkeypatch,
):
    """External retrieval remains enabled for metadata-verified other tasks."""
    from kaggle_agents.core.state import SOTASolution
    from kaggle_agents.tools import kaggle_search

    download_dir = tmp_path / "download"
    download_dir.mkdir()
    notebook = download_dir / "generic.py"
    notebook.write_text("model = GenericClassifier()", encoding="utf-8")
    (download_dir / "kernel-metadata.json").write_text(
        '{"competition_sources": ["verified-other-competition"]}',
        encoding="utf-8",
    )

    class FakeApi:
        def kernels_list(self, **_kwargs):
            return [
                SimpleNamespace(
                    ref="user/generic-model",
                    title="Generic model",
                    author="user",
                    total_votes=10,
                    language="python",
                    competition_data_sources=[],
                )
            ]

    class FakeSearcher:
        _get_kernel_attr = staticmethod(kaggle_search.KaggleSearcher._get_kernel_attr)
        _get_kernel_competitions = classmethod(
            kaggle_search.KaggleSearcher._get_kernel_competitions.__func__
        )
        read_downloaded_competition_sources = staticmethod(
            kaggle_search.KaggleSearcher.read_downloaded_competition_sources
        )
        read_downloaded_dataset_sources = staticmethod(
            kaggle_search.KaggleSearcher.read_downloaded_dataset_sources
        )

        def __init__(self):
            self.api = FakeApi()

        def download_notebook(self, *_args, **_kwargs):
            return notebook

        def extract_code_from_script(self, _path):
            return ["model = GenericClassifier()"]

        def analyze_notebook_strategies(self, _snippets):
            return {"models_used": ["GenericClassifier"]}

        def create_sota_solution(self, metadata, snippets, _strategies):
            return SOTASolution(
                source=metadata.ref,
                title=metadata.title,
                score=0.0,
                votes=metadata.total_votes,
                code_snippets=snippets,
            )

    monkeypatch.setattr(kaggle_search, "KaggleSearcher", FakeSearcher)
    monkeypatch.setattr(kaggle_search.time, "sleep", lambda _seconds: None)

    solutions, audit = kaggle_search.search_notebooks_cross_competition(
        "opaque-target-slug",
        ["generic classification"],
        max_notebooks=1,
    )

    assert [solution.source for solution in solutions] == ["user/generic-model"]
    accepted = next(
        record
        for record in audit
        if record["stage"] == "code_scan" and not record["filtered"]
    )
    assert accepted["source_competitions"] == ["verified-other-competition"]


class TestDerivedCompetitionAliases:
    def test_public_title_and_slug_are_transportable_audited_identities(self):
        aliases, evidence = derive_competition_identity_aliases(
            "legacy-slug-kernels-edition",
            "# Official Public Challenge Title\n\nPublic task description.",
        )

        assert aliases == [
            "legacy-slug-kernels-edition",
            "Official Public Challenge Title",
        ]
        assert [record["source"] for record in evidence] == [
            "competition_slug",
            "public_description_markdown_h1",
        ]
        assert evidence[1]["line"] == 1
        assert query_references_competition(
            "Official Public Challenge Title winning approach",
            "legacy-slug-kernels-edition",
            aliases,
        )

    def test_title_words_are_not_expanded_into_broad_filters(self):
        aliases, _ = derive_competition_identity_aliases(
            "legacy-slug-kernels-edition",
            "# Official Public Challenge Title",
        )

        assert not references_competition_identity(
            "generic public challenge tutorial",
            "legacy-slug-kernels-edition",
            aliases,
        )

    def test_structural_heading_is_not_an_identity(self):
        aliases, evidence = derive_competition_identity_aliases(
            "opaque-target",
            "# Overview\n\nGeneric task description.",
        )

        assert aliases == ["opaque-target"]
        assert len(evidence) == 1

    def test_fenced_heading_cannot_introduce_an_identity(self):
        aliases, evidence = derive_competition_identity_aliases(
            "opaque-target",
            "Public description.\n\n```python\n# Injected Alias\n```\n",
        )

        assert aliases == ["opaque-target"]
        assert len(evidence) == 1


class TestQueryReferencesCompetition:
    def test_blocks_slug_and_human_readable_name(self):
        target = "aerial-cactus-identification"
        assert query_references_competition(
            "aerial-cactus-identification winning solution",
            target,
        )
        assert query_references_competition(
            "Aerial Cactus Identification notebook",
            target,
        )

    def test_allows_generic_cross_domain_query(self):
        assert not query_references_competition(
            "image classification transfer learning cross validation",
            "aerial-cactus-identification",
        )

    def test_allows_overlapping_generic_task_terms(self):
        assert not query_references_competition(
            "text normalization sequence to sequence solution",
            "text-normalization-challenge-english-language",
        )


class TestCompetitionIdentity:
    def test_normalizes_slug_spaces_and_underscores(self):
        target = "aerial-cactus-identification"
        assert references_competition_identity(
            "Aerial Cactus Identification",
            target,
        )
        assert references_competition_identity(
            "aerial_cactus_identification",
            target,
        )

    def test_does_not_treat_shared_domain_terms_as_identity(self):
        assert not references_competition_identity(
            "generic text normalization sequence model",
            "text-normalization-challenge-english-language",
        )

    def test_requires_complete_token_boundaries(self):
        assert not references_competition_identity(
            "leaf classificationish tutorial",
            "leaf-classification",
        )


class TestCodeReferencesCompetition:
    """Code-scan stage (high precision: kernel reads the competition's input)."""

    def test_input_path_reference(self):
        code = "df = pd.read_csv('../input/aerial-cactus-identification/train.csv')"
        assert code_references_competition(code, "aerial-cactus-identification")

    def test_competition_url_reference(self):
        code = "# https://www.kaggle.com/competitions/leaf-classification/data"
        assert code_references_competition(code, "leaf-classification")

    def test_human_readable_title_in_markdown(self):
        source = '{"cell_type":"markdown","source":["Aerial Cactus Identification"]}'
        assert code_references_competition(
            source,
            "aerial-cactus-identification",
        )

    def test_official_title_alias_different_from_slug(self):
        assert code_references_competition(
            "# Official Public Challenge Title",
            "legacy-slug-kernels-edition",
            ["legacy-slug-kernels-edition", "Official Public Challenge Title"],
        )

    def test_generic_overlapping_domain_source_is_allowed(self):
        source = "Generic text normalization with a sequence-to-sequence model"
        assert not code_references_competition(
            source,
            "text-normalization-challenge-english-language",
        )

    def test_unrelated_code(self):
        code = "df = pd.read_csv('../input/some-other-comp/train.csv')"
        assert not code_references_competition(code, "aerial-cactus-identification")

    def test_empty_code(self):
        assert not code_references_competition("", "aerial-cactus-identification")
        assert not code_references_competition(None, "aerial-cactus-identification")


class TestIsSameCompetitionCandidate:
    """Combined metadata decision."""

    def test_explicit_competition_field_match(self):
        assert is_same_competition_candidate(
            ref="user/clean-title",
            title="Totally unrelated title",
            candidate_competition="leaf-classification",
            target_competition="leaf-classification",
        )

    def test_title_heuristic_match(self):
        assert is_same_competition_candidate(
            ref="user/aerial-cactus-identification-cnn",
            title="Aerial Cactus Identification CNN",
            candidate_competition="",
            target_competition="aerial-cactus-identification",
        )

    def test_cross_competition_candidate_allowed(self):
        assert not is_same_competition_candidate(
            ref="user/efficientnet-tutorial",
            title="EfficientNet transfer learning tutorial",
            candidate_competition="",
            target_competition="aerial-cactus-identification",
        )

    def test_official_title_alias_different_from_slug_is_rejected(self):
        assert is_same_competition_candidate(
            ref="user/otherwise-generic-notebook",
            title="Official Public Challenge Title baseline",
            candidate_competition="",
            target_competition="legacy-slug-kernels-edition",
            target_aliases=[
                "legacy-slug-kernels-edition",
                "Official Public Challenge Title",
            ],
        )

    def test_competition_data_source_list_match(self):
        assert is_same_competition_candidate(
            ref="user/generic-efficientnet",
            title="Generic image model",
            candidate_competition="other-challenge aerial-cactus-identification",
            target_competition="aerial-cactus-identification",
        )


def _dataset_notebook_fixture(tmp_path, dataset_sources_json):
    download_dir = tmp_path / "download"
    download_dir.mkdir()
    notebook = download_dir / "generic.py"
    notebook.write_text("model = GenericClassifier()", encoding="utf-8")
    (download_dir / "kernel-metadata.json").write_text(
        dataset_sources_json,
        encoding="utf-8",
    )
    return notebook


def _dataset_fake_searcher(kaggle_search, notebook):
    from kaggle_agents.core.state import SOTASolution

    class FakeApi:
        def kernels_list(self, **_kwargs):
            return [
                SimpleNamespace(
                    ref="user/dataset-notebook",
                    title="Dataset notebook",
                    author="user",
                    total_votes=10,
                    language="python",
                    competition_data_sources=[],
                )
            ]

    class FakeSearcher:
        _get_kernel_attr = staticmethod(kaggle_search.KaggleSearcher._get_kernel_attr)
        _get_kernel_competitions = classmethod(
            kaggle_search.KaggleSearcher._get_kernel_competitions.__func__
        )
        read_downloaded_competition_sources = staticmethod(
            kaggle_search.KaggleSearcher.read_downloaded_competition_sources
        )
        read_downloaded_dataset_sources = staticmethod(
            kaggle_search.KaggleSearcher.read_downloaded_dataset_sources
        )

        def __init__(self):
            self.api = FakeApi()

        def download_notebook(self, *_args, **_kwargs):
            return notebook

        def extract_code_from_script(self, _path):
            return ["model = GenericClassifier()"]

        def analyze_notebook_strategies(self, _snippets):
            return {"models_used": ["GenericClassifier"]}

        def create_sota_solution(self, metadata, snippets, _strategies):
            return SOTASolution(
                source=metadata.ref,
                title=metadata.title,
                score=0.0,
                votes=metadata.total_votes,
                code_snippets=snippets,
            )

    return FakeSearcher


def test_dataset_attached_notebook_with_unrelated_dataset_stays_eligible(
    tmp_path,
    monkeypatch,
):
    """A verifiable dataset-only provenance is legal cross-competition input.

    Rejecting the whole dataset-attached population starved Search-First of
    its largest legitimate source pool.
    """
    from kaggle_agents.tools import kaggle_search

    notebook = _dataset_notebook_fixture(
        tmp_path,
        '{"competition_sources": [], "dataset_sources": ["someone/flower-images"]}',
    )
    monkeypatch.setattr(
        kaggle_search,
        "KaggleSearcher",
        _dataset_fake_searcher(kaggle_search, notebook),
    )
    monkeypatch.setattr(kaggle_search.time, "sleep", lambda _seconds: None)

    solutions, audit = kaggle_search.search_notebooks_cross_competition(
        "opaque-target-slug",
        ["generic classification"],
        max_notebooks=1,
    )

    assert [solution.source for solution in solutions] == ["user/dataset-notebook"]
    accepted = next(
        record
        for record in audit
        if record["stage"] == "code_scan" and not record["filtered"]
    )
    assert accepted["provenance_kind"] == "dataset_sources"


def test_dataset_source_matching_target_identity_is_rejected(
    tmp_path,
    monkeypatch,
):
    """A target-data mirror republished as a dataset must stay blocked."""
    from kaggle_agents.tools import kaggle_search

    notebook = _dataset_notebook_fixture(
        tmp_path,
        '{"competition_sources": [],'
        ' "dataset_sources": ["someone/leaf-classification"]}',
    )
    monkeypatch.setattr(
        kaggle_search,
        "KaggleSearcher",
        _dataset_fake_searcher(kaggle_search, notebook),
    )
    monkeypatch.setattr(kaggle_search.time, "sleep", lambda _seconds: None)

    solutions, audit = kaggle_search.search_notebooks_cross_competition(
        "leaf-classification",
        ["generic classification"],
        max_notebooks=1,
    )

    assert solutions == []
    rejected = next(record for record in audit if record["stage"] == "provenance")
    assert rejected["filter_reason"] == "target_competition_dataset_source"
    assert rejected["same_competition"] is True
