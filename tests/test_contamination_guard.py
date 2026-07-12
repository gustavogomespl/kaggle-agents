"""Tests for the MLE-bench contamination guard (utils/contamination.py)."""

from types import SimpleNamespace

from kaggle_agents.utils.contamination import (
    code_references_competition,
    competition_slug_tokens,
    is_same_competition_candidate,
    looks_like_same_competition,
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
    assert audit[0]["stage"] == "initialization"
    assert "credentials" in audit[0]["error"]


def test_cross_competition_search_rejects_unparseable_source(
    temp_data_dir, monkeypatch
):
    from kaggle_agents.tools import kaggle_search

    invalid_notebook = temp_data_dir / "invalid.ipynb"
    invalid_notebook.write_text("not-json", encoding="utf-8")

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


class TestCompetitionSlugTokens:
    """Distinctive-token extraction from competition slugs."""

    def test_extracts_distinctive_tokens(self):
        tokens = competition_slug_tokens("aerial-cactus-identification")
        assert tokens == {"aerial", "cactus"}

    def test_filters_stopwords_and_short_tokens(self):
        # "series", "playground", "tabular", "2021" are stopwords; "dec" is too short
        tokens = competition_slug_tokens("tabular-playground-series-dec-2021")
        assert tokens == set()

    def test_single_distinctive_token(self):
        assert competition_slug_tokens("leaf-classification") == {"leaf"}


class TestLooksLikeSameCompetition:
    """Metadata-stage heuristic (ref/title matching)."""

    def test_full_slug_match(self):
        assert looks_like_same_competition(
            "user/aerial-cactus-identification-starter", "aerial-cactus-identification"
        )

    def test_slug_with_spaces_match(self):
        assert looks_like_same_competition(
            "Aerial Cactus Identification with CNN", "aerial-cactus-identification"
        )

    def test_two_token_match(self):
        assert looks_like_same_competition(
            "Cactus aerial imagery CNN baseline", "aerial-cactus-identification"
        )

    def test_single_shared_word_not_enough(self):
        # Only "melanoma" matches (1 of {siim, isic, melanoma}) -> cross-competition OK
        assert not looks_like_same_competition(
            "Melanoma classification with EfficientNet",
            "siim-isic-melanoma-classification",
        )

    def test_single_token_slug_requires_one_hit(self):
        assert looks_like_same_competition(
            "Leaf classification with sklearn", "leaf-classification"
        )

    def test_unrelated_text(self):
        assert not looks_like_same_competition(
            "Titanic survival prediction tutorial", "aerial-cactus-identification"
        )

    def test_empty_inputs(self):
        assert not looks_like_same_competition("", "aerial-cactus-identification")
        assert not looks_like_same_competition(None, "aerial-cactus-identification")
        assert not looks_like_same_competition("anything", "")

    def test_generic_slug_only_matches_full_slug(self):
        # All tokens are stopwords -> only the literal slug should match
        slug = "tabular-playground-series-dec-2021"
        assert looks_like_same_competition(f"solution for {slug}", slug)
        assert not looks_like_same_competition("tabular playground fun with lightgbm", slug)


class TestCodeReferencesCompetition:
    """Code-scan stage (high precision: kernel reads the competition's input)."""

    def test_input_path_reference(self):
        code = "df = pd.read_csv('../input/aerial-cactus-identification/train.csv')"
        assert code_references_competition(code, "aerial-cactus-identification")

    def test_competition_url_reference(self):
        code = "# https://www.kaggle.com/competitions/leaf-classification/data"
        assert code_references_competition(code, "leaf-classification")

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
            ref="user/cactus-aerial-cnn",
            title="Aerial cactus CNN",
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

    def test_competition_data_source_list_match(self):
        assert is_same_competition_candidate(
            ref="user/generic-efficientnet",
            title="Generic image model",
            candidate_competition="other-challenge aerial-cactus-identification",
            target_competition="aerial-cactus-identification",
        )
