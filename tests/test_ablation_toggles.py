"""Tests for ablation toggles and contamination-guard configuration."""

from kaggle_agents.core.config import AblationTogglesConfig, SearchConfig


_TOGGLE_ENV_VARS = [
    "KAGGLE_AGENTS_ABLATE_SEARCH",
    "KAGGLE_AGENTS_ABLATE_ROBUSTNESS",
    "KAGGLE_AGENTS_ABLATE_META_EVALUATOR",
    "KAGGLE_AGENTS_ABLATE_ENSEMBLE",
]


class TestAblationTogglesConfig:
    def test_defaults_all_enabled(self, monkeypatch):
        for var in _TOGGLE_ENV_VARS:
            monkeypatch.delenv(var, raising=False)

        toggles = AblationTogglesConfig()
        assert toggles.disable_search is False
        assert toggles.disable_robustness is False
        assert toggles.disable_meta_evaluator is False
        assert toggles.disable_ensemble is False
        assert toggles.disabled_components() == []

    def test_env_enables_ablation(self, monkeypatch):
        for var in _TOGGLE_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("KAGGLE_AGENTS_ABLATE_SEARCH", "true")
        monkeypatch.setenv("KAGGLE_AGENTS_ABLATE_ENSEMBLE", "TRUE")

        toggles = AblationTogglesConfig()
        assert toggles.disable_search is True
        assert toggles.disable_ensemble is True
        assert toggles.disable_robustness is False
        assert set(toggles.disabled_components()) == {"search", "ensemble"}

    def test_non_true_values_ignored(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_AGENTS_ABLATE_SEARCH", "1")  # only "true" enables
        toggles = AblationTogglesConfig()
        assert toggles.disable_search is False


class TestContaminationGuardConfig:
    def test_guard_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_AGENTS_ALLOW_SAME_COMP_SOURCES", raising=False)
        search = SearchConfig()
        assert search.allow_same_competition_sources is False

    def test_explicit_override(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_AGENTS_ALLOW_SAME_COMP_SOURCES", "true")
        search = SearchConfig()
        assert search.allow_same_competition_sources is True
