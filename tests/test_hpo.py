"""Tests for HPO multi-fidelity utilities (Hyperband/ASHA)."""

import sys
from pathlib import Path

import pytest


# Import directly from module file to avoid circular imports through __init__.py
# This is a test-specific workaround
_module_path = Path(__file__).parent.parent / "kaggle_agents" / "optimization"
sys.path.insert(0, str(_module_path.parent))

# Direct import from hpo.py without going through __init__.py
import importlib.util


spec = importlib.util.spec_from_file_location("hpo", _module_path / "hpo.py")
hpo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hpo_module)

HPO_MULTI_FIDELITY_INSTRUCTIONS = hpo_module.HPO_MULTI_FIDELITY_INSTRUCTIONS
validate_pruning_contract = hpo_module.validate_pruning_contract


class TestValidatePruningContract:
    """Tests for Optuna pruning contract validation."""

    def test_passes_when_no_optuna(self):
        """Should pass when code doesn't use Optuna."""
        code = """
import pandas as pd
import numpy as np

model.fit(X_train, y_train)
predictions = model.predict(X_test)
"""
        is_valid, error = validate_pruning_contract(code)
        assert is_valid
        assert error == ""

    def test_passes_when_optuna_without_pruner(self):
        """Should pass when Optuna is used but no pruner is active."""
        code = """
import optuna

study = optuna.create_study(direction='minimize')

def objective(trial):
    params = {'lr': trial.suggest_float('lr', 0.01, 0.1)}
    return 0.5

study.optimize(objective, n_trials=10)
"""
        is_valid, error = validate_pruning_contract(code)
        assert is_valid
        assert error == ""

    def test_fails_when_hyperband_without_report(self):
        """Should fail when HyperbandPruner is used but trial.report() is missing."""
        code = """
import optuna
from optuna.pruners import HyperbandPruner

study = optuna.create_study(
    direction='minimize',
    pruner=HyperbandPruner(),
)

def objective(trial):
    params = {'lr': trial.suggest_float('lr', 0.01, 0.1)}
    return 0.5

study.optimize(objective, n_trials=10)
"""
        is_valid, error = validate_pruning_contract(code)
        assert not is_valid
        assert "trial.report()" in error
        assert "Hyperband" in error

    def test_fails_when_median_without_should_prune(self):
        """Should fail when MedianPruner is used with report but without should_prune."""
        code = """
import optuna
from optuna.pruners import MedianPruner

study = optuna.create_study(
    direction='minimize',
    pruner=MedianPruner(),
)

def objective(trial):
    for step in range(100):
        score = train_step()
        trial.report(score, step)
    return score

study.optimize(objective, n_trials=10)
"""
        is_valid, error = validate_pruning_contract(code)
        assert not is_valid
        assert "should_prune" in error
        assert "Median" in error

    def test_passes_with_full_contract(self):
        """Should pass when pruner is used with both report and should_prune."""
        code = """
import optuna
from optuna.pruners import HyperbandPruner

study = optuna.create_study(
    direction='minimize',
    pruner=HyperbandPruner(),
)

def objective(trial):
    for step in range(100):
        score = train_step()
        trial.report(score, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return score

study.optimize(objective, n_trials=10)
"""
        is_valid, error = validate_pruning_contract(code)
        assert is_valid
        assert error == ""

    def test_passes_with_trialPruned_only(self):
        """Should pass when TrialPruned is used (alternative pattern)."""
        code = """
import optuna
from optuna.pruners import SuccessiveHalvingPruner

study = optuna.create_study(
    direction='minimize',
    pruner=SuccessiveHalvingPruner(),
)

def objective(trial):
    for step in range(100):
        score = train_step()
        trial.report(score, step)
        if condition:
            raise optuna.TrialPruned()
    return score
"""
        is_valid, error = validate_pruning_contract(code)
        assert is_valid
        assert error == ""

    def test_detects_all_pruner_types(self):
        """Should detect various pruner types."""
        pruner_types = [
            ("HyperbandPruner", "Hyperband"),
            ("MedianPruner", "Median"),
            ("SuccessiveHalvingPruner", "SuccessiveHalving"),
            ("ThresholdPruner", "Threshold"),
            ("PercentilePruner", "Percentile"),
        ]

        for pruner_class, expected_name in pruner_types:
            code = f"""
import optuna
from optuna.pruners import {pruner_class}

study = optuna.create_study(pruner={pruner_class}())

def objective(trial):
    return 0.5  # Missing report and prune check
"""
            is_valid, error = validate_pruning_contract(code)
            assert not is_valid, f"{pruner_class} should fail validation"
            assert expected_name in error, f"Error should mention {expected_name}"

    def test_detects_from_optuna_import(self):
        """Should detect Optuna usage via 'from optuna' import."""
        code = """
from optuna import create_study
from optuna.pruners import HyperbandPruner

study = create_study(pruner=HyperbandPruner())

def objective(trial):
    return 0.5
"""
        is_valid, error = validate_pruning_contract(code)
        assert not is_valid

    def test_detects_optuna_Study_class(self):
        """Should detect Optuna usage via optuna.Study."""
        code = """
import optuna
from optuna.pruners import MedianPruner

study = optuna.Study(study_name='test', storage=None, pruner=MedianPruner())

def objective(trial):
    return 0.5
"""
        is_valid, error = validate_pruning_contract(code)
        assert not is_valid


class TestHPOMultiFidelityInstructions:
    """Tests for HPO instruction constant."""

    def test_instructions_mention_hyperband(self):
        """Instructions should mention Hyperband."""
        assert "Hyperband" in HPO_MULTI_FIDELITY_INSTRUCTIONS

    def test_instructions_mention_trial_report(self):
        """Instructions should mention trial.report."""
        assert "trial.report" in HPO_MULTI_FIDELITY_INSTRUCTIONS

    def test_instructions_mention_should_prune(self):
        """Instructions should mention should_prune."""
        assert "should_prune" in HPO_MULTI_FIDELITY_INSTRUCTIONS

    def test_instructions_mention_trialPruned(self):
        """Instructions should mention TrialPruned."""
        assert "TrialPruned" in HPO_MULTI_FIDELITY_INSTRUCTIONS

    def test_instructions_include_lgbm_example(self):
        """Instructions should include LightGBM example."""
        assert "lgb" in HPO_MULTI_FIDELITY_INSTRUCTIONS.lower() or "lightgbm" in HPO_MULTI_FIDELITY_INSTRUCTIONS.lower()

    def test_instructions_include_xgb_example(self):
        """Instructions should include XGBoost example."""
        assert "xgb" in HPO_MULTI_FIDELITY_INSTRUCTIONS.lower()


class TestOptunaBuildersIntegration:
    """Tests for optuna prompts builder integration."""

    @pytest.fixture
    def optuna_builder(self):
        """Load optuna builder module directly to avoid circular imports."""
        optuna_path = Path(__file__).parent.parent / "kaggle_agents" / "prompts" / "templates" / "builders" / "optuna.py"
        spec = importlib.util.spec_from_file_location("optuna_builder", optuna_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_build_optuna_instructions_without_pruner(self, optuna_builder):
        """Should return basic instructions without pruner."""
        build_optuna_tuning_instructions = optuna_builder.build_optuna_tuning_instructions

        instructions = build_optuna_tuning_instructions(
            n_trials=5,
            timeout=540,
            use_pruner=False,
        )

        # Should have basic Optuna instructions
        instructions_text = "\n".join(instructions)
        assert "OPTUNA" in instructions_text
        assert "n_trials" in instructions_text

        # Should NOT have pruner-specific instructions
        assert "HyperbandPruner" not in instructions_text

    def test_build_optuna_instructions_with_pruner(self, optuna_builder):
        """Should include pruner instructions when use_pruner=True."""
        build_optuna_tuning_instructions = optuna_builder.build_optuna_tuning_instructions

        instructions = build_optuna_tuning_instructions(
            n_trials=5,
            timeout=540,
            use_pruner=True,
        )

        instructions_text = "\n".join(instructions)

        # Should have basic Optuna instructions
        assert "OPTUNA" in instructions_text

        # Should ALSO have pruner-specific instructions
        assert "HyperbandPruner" in instructions_text
        assert "trial.report" in instructions_text
        assert "should_prune" in instructions_text

    def test_instructions_length_increases_with_pruner(self, optuna_builder):
        """Instructions should be longer when pruner is enabled."""
        build_optuna_tuning_instructions = optuna_builder.build_optuna_tuning_instructions

        without_pruner = build_optuna_tuning_instructions(use_pruner=False)
        with_pruner = build_optuna_tuning_instructions(use_pruner=True)

        assert len(with_pruner) > len(without_pruner)
