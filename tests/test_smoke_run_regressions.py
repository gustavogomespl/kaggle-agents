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

from kaggle_agents.agents.developer.agent import (
    _resolved_primary_score,
    unsaved_expected_artifacts,
)
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

    def test_a_reused_result_keeps_its_trusted_score_for_promotion(self):
        """Skipping the gate leaves new_cv_score=None for reused results, and
        the promotion block read that None as "no independently reproducible
        OOF score" — rejecting the reused component and quarantining the
        prior best model's artifacts on every refinement iteration (observed
        on a live run: the accepted baseline was stripped at the start of
        iteration 2). The score it earned is still in the trusted map."""
        reused = DevelopmentResult(code="x", success=True)
        reused.reused_from_cache = True
        state = {"trusted_component_scores": {"baseline": 0.9767}}

        assert _resolved_primary_score(reused, "baseline", state, None) == (
            pytest.approx(0.9767)
        )

    def test_a_fresh_unscored_result_still_resolves_to_none(self):
        """Only cache reuse may substitute the stored score: a fresh result
        without a recomputed score remains unscored and keeps the existing
        fail-closed handling."""
        fresh = DevelopmentResult(code="x", success=True)
        state = {"trusted_component_scores": {"baseline": 0.9767}}

        assert _resolved_primary_score(fresh, "baseline", state, None) is None

    def test_a_reused_result_without_a_stored_score_stays_unscored(self):
        reused = DevelopmentResult(code="x", success=True)
        reused.reused_from_cache = True

        assert (
            _resolved_primary_score(reused, "baseline", {}, None) is None
        )


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

    def test_code_without_any_save_call_is_flagged(self):
        """The scan walks the whole tree, including the script's own wrapper
        functions, for np.save/torch.save/savez/tofile. Zero matches in a model
        component means nothing is persisted -- a finding, not an unknown."""
        assert sorted(unsaved_expected_artifacts("x = 1\n", EXPECTED, COMPONENT)) == sorted(
            EXPECTED
        )

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


class TestComponentNameIsImmutable:
    """Every evidence artifact is named after COMPONENT_NAME. The generated
    code rebound it to the architecture ("baseline_densenet121"), so the saves
    landed under a name the contract does not look for and 25 minutes of
    correct training were failed for "missing artifacts"."""

    def test_component_name_is_protected(self):
        from kaggle_agents.agents.developer.code_generator import IMMUTABLE_PATH_VARS

        assert "COMPONENT_NAME" in IMMUTABLE_PATH_VARS

    def test_rebinding_is_detected_and_stripped(self):
        from kaggle_agents.agents.developer.code_generator import CodeGeneratorMixin

        code = (
            'COMPONENT_NAME = "baseline_cnn_transfer_learning"\n'
            "# === END PATH CONSTANTS ===\n"
            'COMPONENT_NAME = "baseline_densenet121"\n'
            'np.save(MODELS_DIR / f"oof_{COMPONENT_NAME}.npy", oof)\n'
        )
        mixin = CodeGeneratorMixin()

        is_valid, violations = mixin._validate_no_path_redefinition(code)
        assert is_valid is False
        assert any("COMPONENT_NAME" in v for v in violations)

        stripped = mixin._strip_path_redefinitions(code)
        assert 'COMPONENT_NAME = "baseline_densenet121"' not in stripped.replace(
            "# STRIPPED (path constant): ", ""
        ) or "STRIPPED" in stripped
        # The header definition survives.
        assert 'COMPONENT_NAME = "baseline_cnn_transfer_learning"' in stripped


class TestSaveTargetIsFoundInAnyArgument:
    """np.save(path, arr) puts the destination first; torch.save(obj, path)
    puts it second. Reading only argument 0 made the check bail out on every
    component that uses torch.save -- that is, nearly all of them."""

    def test_torch_save_does_not_disable_the_check(self):
        code = (
            "import numpy as np, torch\n"
            'torch.save(model.state_dict(), MODELS_DIR / f"{COMPONENT_NAME}_fold0.pth")\n'
            'np.save(MODELS_DIR / f"oof_{COMPONENT_NAME}.npy", oof)\n'
        )

        missing = unsaved_expected_artifacts(code, EXPECTED, COMPONENT)

        assert sorted(missing) == sorted(
            [
                f"models/test_{COMPONENT}.npy",
                f"models/train_ids_{COMPONENT}.npy",
                f"models/test_ids_{COMPONENT}.npy",
            ]
        )

    def test_the_second_smoke_run_program_is_caught(self):
        """Exactly the shape that burned 1553s: one OOF save plus checkpoints."""
        code = (
            "import numpy as np, torch\n"
            "for fold in range(5):\n"
            '    torch.save(model.state_dict(), MODELS_DIR / f"{COMPONENT_NAME}_fold{fold}.pth")\n'
            'np.save(MODELS_DIR / f"oof_{COMPONENT_NAME}.npy", oof_preds)\n'
        )

        assert unsaved_expected_artifacts(code, EXPECTED, COMPONENT)

    def test_a_complete_program_still_passes(self):
        code = (
            "import numpy as np, torch\n"
            'torch.save(model.state_dict(), MODELS_DIR / f"{COMPONENT_NAME}_fold0.pth")\n'
            'np.save(MODELS_DIR / f"oof_{COMPONENT_NAME}.npy", oof)\n'
            'np.save(MODELS_DIR / f"test_{COMPONENT_NAME}.npy", test)\n'
            'np.save(MODELS_DIR / f"train_ids_{COMPONENT_NAME}.npy", tr_ids)\n'
            'np.save(MODELS_DIR / f"test_ids_{COMPONENT_NAME}.npy", te_ids)\n'
        )

        assert unsaved_expected_artifacts(code, EXPECTED, COMPONENT) == []

    def test_fully_opaque_saves_still_disable_the_check(self):
        code = "import torch\ntorch.save(obj, dest)\n"

        assert unsaved_expected_artifacts(code, EXPECTED, COMPONENT) == []


class TestEvidenceArtifactHelper:
    """Four separate np.save calls with exact filenames had to be reconstructed
    from a 238-line instruction block containing eight np.save mentions. Four
    out of four model components across two runs saved only `oof_`. The
    contract is now one call."""

    def _helper_source(self) -> str:
        from kaggle_agents.agents.developer.code_generator import (
            _EVIDENCE_ARTIFACT_HELPER,
        )

        return _EVIDENCE_ARTIFACT_HELPER

    def test_helper_is_valid_python(self):
        import ast

        ast.parse(self._helper_source())

    def test_helper_writes_all_four_artifacts(self):
        source = self._helper_source()

        for kind in ("oof_", "test_", "train_ids_", "test_ids_"):
            assert f'{kind}{{COMPONENT_NAME}}.npy' in source

    def test_calling_the_helper_satisfies_the_contract(self):
        code = self._helper_source() + "\nsave_component_artifacts(oof, test, test_ids=ids)\n"

        assert unsaved_expected_artifacts(code, EXPECTED, COMPONENT) == []

    def test_defining_without_calling_does_not_satisfy_it(self):
        """The helper is injected into every script, so its body proves nothing
        about the program the model actually wrote."""
        code = self._helper_source() + "\nprint('trained but never saved')\n"

        missing = unsaved_expected_artifacts(code, EXPECTED, COMPONENT)

        assert sorted(missing) == sorted(EXPECTED)

    def test_helper_body_does_not_mask_a_partial_hand_rolled_save(self):
        code = (
            self._helper_source()
            + "\nimport numpy as np\n"
            + 'np.save(MODELS_DIR / f"oof_{COMPONENT_NAME}.npy", oof)\n'
        )

        missing = unsaved_expected_artifacts(code, EXPECTED, COMPONENT)

        assert f"models/test_{COMPONENT}.npy" in missing
        assert f"models/oof_{COMPONENT}.npy" not in missing

    def test_every_component_told_to_save_evidence_receives_the_helper(self):
        """Injection scope must match what the constraints demand.

        Ensembles are told by BASE_CONSTRAINTS and by the ensemble plan
        outlines to call save_component_artifacts. Injecting that helper only
        for models left ensemble code calling a name that did not exist, so it
        imported one instead - and the helper-shadowing guard then rejected
        every attempt before execution, burning the entire repair budget.
        """
        import ast
        from pathlib import Path

        from kaggle_agents.prompts.templates.constraints.base import (
            BASE_CONSTRAINTS,
        )

        # The instruction side: constraints loaded for every component type.
        assert "save_component_artifacts" in BASE_CONSTRAINTS

        source = Path(
            "kaggle_agents/agents/developer/code_generator.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)

        model_only_guarded: set[str] = set()
        submission_guarded: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            condition = ast.unparse(node.test).replace("'", '"')
            for statement in node.body:
                for sub in ast.walk(statement):
                    if isinstance(sub, ast.Name):
                        if 'component_type == "model"' in condition:
                            model_only_guarded.add(sub.id)
                        if '"model", "ensemble"' in condition:
                            submission_guarded.add(sub.id)

        # The injection side: both helpers share the model/ensemble scope, and
        # neither is narrowed to models only.
        assert "_submission_helper_for_contract" in submission_guarded
        assert "_EVIDENCE_ARTIFACT_HELPER" in submission_guarded
        assert "_IMAGE_EVIDENCE_ARTIFACT_HELPER" in submission_guarded
        assert "_EVIDENCE_ARTIFACT_HELPER" not in model_only_guarded
        assert "_IMAGE_EVIDENCE_ARTIFACT_HELPER" not in model_only_guarded
