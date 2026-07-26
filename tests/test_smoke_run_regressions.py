"""Regressions from the first real smoke run on aerial-cactus.

That run trained two models successfully (AUC 0.99998), then reported
`valid_submission: No` after 110 minutes. Three independent defects combined:
a cached component was rejected against the baseline it had itself set, the
rollback wrote None into a state field declared float, and the prompt builder
crashed on that None. Roughly half the GPU time went to re-training programs
that had already succeeded but skipped their artifact saves.
"""

from __future__ import annotations

import pytest

from kaggle_agents.agents.developer.agent import unsaved_expected_artifacts
from kaggle_agents.core.state.results import DevelopmentResult
from kaggle_agents.prompts.templates.builders.model import (
    build_performance_gap_instructions,
)


COMPONENT = "baseline_cnn_transfer_learning"
EXPECTED = [
    f"models/oof_{COMPONENT}.npy",
    f"models/test_{COMPONENT}.npy",
    f"models/train_ids_{COMPONENT}.npy",
    f"models/test_ids_{COMPONENT}.npy",
]


class TestPromptBuilderSurvivesRollback:
    """A rollback nulls the run's best score. The prompt builder must not turn
    that into a crash that ends a run holding an accepted submission."""

    def test_none_current_score_does_not_raise(self):
        assert build_performance_gap_instructions(None, None, "auc") == []
        assert build_performance_gap_instructions(None, 0.9, "auc") == []

    def test_none_target_score_is_still_ignored(self):
        assert build_performance_gap_instructions(0.8, None, "auc") == []

    def test_non_positive_score_is_still_ignored(self):
        assert build_performance_gap_instructions(0.0, 0.9, "auc") == []

    def test_a_real_gap_still_produces_instructions(self):
        assert build_performance_gap_instructions(0.80, 0.95, "auc")


class TestRollbackWritesAFloat:
    """`current_performance_score` is declared float, and its readers use
    `state.get(key, 0.0)` -- which does not protect against a key present with
    value None."""

    def _rollback_updates(self, module_path: str) -> str:
        from pathlib import Path

        return Path(module_path).read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "module_path",
        [
            "kaggle_agents/agents/developer/agent.py",
            "kaggle_agents/workflow/nodes/robustness_gate.py",
        ],
    )
    def test_no_module_writes_none_to_the_float_field(self, module_path):
        source = self._rollback_updates(module_path)

        assert '"current_performance_score": None' not in source


class TestCachedComponentIsNotRejudged:
    def test_cached_results_are_flagged(self):
        result = DevelopmentResult(code="x", success=True)

        assert result.reused_from_cache is False
        result.reused_from_cache = True
        assert result.reused_from_cache is True

    def test_the_gate_is_skipped_for_reused_results(self):
        """A component reused from cache scores exactly what it scored before,
        so re-judging it against its own baseline always rejects it -- and the
        rollback then quarantines the winning model's artifacts."""
        from pathlib import Path

        source = Path("kaggle_agents/agents/developer/agent.py").read_text(
            encoding="utf-8"
        )
        gate_call = source.index("_validate_component_improvement(")
        guard = source.rindex("reused_from_cache", 0, gate_call)

        # The guard sits immediately above the gate, in the same condition.
        assert source.count("reused_from_cache", guard, gate_call) == 1


class TestArtifactContractCheckedBeforeTraining:
    def _code(self, saves: list[str]) -> str:
        body = "\n".join(f'np.save(MODELS_DIR / {s}, arr)' for s in saves)
        return f"import numpy as np\n{body}\n"

    def test_the_smoke_run_program_is_caught(self):
        """It saved only the OOF array, then failed after full training."""
        code = self._code(['f"oof_{COMPONENT_NAME}.npy"'])

        missing = unsaved_expected_artifacts(code, EXPECTED, COMPONENT)

        assert sorted(missing) == sorted(
            [
                f"models/test_{COMPONENT}.npy",
                f"models/train_ids_{COMPONENT}.npy",
                f"models/test_ids_{COMPONENT}.npy",
            ]
        )

    def test_composed_filenames_count_as_saved(self):
        code = self._code(
            [
                'f"oof_{COMPONENT_NAME}.npy"',
                'f"test_{COMPONENT_NAME}.npy"',
                'f"train_ids_{COMPONENT_NAME}.npy"',
                'f"test_ids_{COMPONENT_NAME}.npy"',
            ]
        )

        assert unsaved_expected_artifacts(code, EXPECTED, COMPONENT) == []

    def test_literal_filenames_count_as_saved(self):
        code = self._code(
            [
                f'"oof_{COMPONENT}.npy"',
                f'"test_{COMPONENT}.npy"',
                f'"train_ids_{COMPONENT}.npy"',
                f'"test_ids_{COMPONENT}.npy"',
            ]
        )

        assert unsaved_expected_artifacts(code, EXPECTED, COMPONENT) == []

    def test_test_ids_does_not_satisfy_test(self):
        """`test_` is a prefix of `test_ids_`; the check must not confuse them."""
        code = self._code(
            [
                'f"oof_{COMPONENT_NAME}.npy"',
                'f"train_ids_{COMPONENT_NAME}.npy"',
                'f"test_ids_{COMPONENT_NAME}.npy"',
            ]
        )

        assert unsaved_expected_artifacts(code, EXPECTED, COMPONENT) == [
            f"models/test_{COMPONENT}.npy"
        ]

    def test_unparseable_code_is_never_blocked(self):
        assert unsaved_expected_artifacts("def (", EXPECTED, COMPONENT) == []

    def test_code_without_save_calls_is_never_blocked(self):
        """Absence of evidence is not evidence of absence: only report when
        save calls exist and demonstrably omit an artifact."""
        assert unsaved_expected_artifacts("x = 1\n", EXPECTED, COMPONENT) == []

    def test_no_expectations_means_no_findings(self):
        code = self._code(['f"oof_{COMPONENT_NAME}.npy"'])

        assert unsaved_expected_artifacts(code, None, COMPONENT) == []
        assert unsaved_expected_artifacts(code, [], COMPONENT) == []

    def test_unknown_component_name_is_never_blocked(self):
        code = self._code(['f"oof_{COMPONENT_NAME}.npy"'])

        assert unsaved_expected_artifacts(code, EXPECTED, "") == []

    def test_an_unusual_save_expression_is_not_blocked(self):
        """A path built through a variable cannot be read statically."""
        code = "import numpy as np\nfor p in paths:\n    np.save(p, arr)\n"

        assert unsaved_expected_artifacts(code, EXPECTED, COMPONENT) == []
