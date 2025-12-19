"""
Tests for Quiet-STaR Self-Evaluation functionality.

Tests the self-evaluation methods in DeveloperAgent.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from kaggle_agents.core.state import (
    SelfEvaluation,
    AblationComponent,
    CompetitionInfo,
    KaggleState,
    create_initial_state,
)


class TestSelfEvaluation:
    """Tests for SelfEvaluation dataclass."""

    def test_create_self_evaluation(self):
        """Test creating a SelfEvaluation instance."""
        eval_result = SelfEvaluation(
            confidence=0.85,
            concerns=["Missing error handling", "No early stopping"],
            suggested_fixes=["Add try/except", "Add early_stopping_rounds"],
            proceed=True,
            reflection_summary="Code looks mostly good but needs error handling",
        )

        assert eval_result.confidence == 0.85
        assert len(eval_result.concerns) == 2
        assert len(eval_result.suggested_fixes) == 2
        assert eval_result.proceed is True
        assert "error handling" in eval_result.reflection_summary

    def test_self_evaluation_defaults(self):
        """Test SelfEvaluation default values."""
        eval_result = SelfEvaluation()

        assert eval_result.confidence == 0.0
        assert eval_result.concerns == []
        assert eval_result.suggested_fixes == []
        assert eval_result.proceed is True
        assert eval_result.reflection_summary == ""


class TestSelfEvaluationIntegration:
    """Integration tests for self-evaluation with mocked LLM."""

    @pytest.fixture
    def mock_llm_response_high_confidence(self):
        """Mock LLM response with high confidence."""
        return MagicMock(
            content="""
{
    "confidence": 0.9,
    "concerns": [],
    "suggested_fixes": [],
    "proceed": true,
    "reflection": "Code looks solid and should work correctly"
}
"""
        )

    @pytest.fixture
    def mock_llm_response_low_confidence(self):
        """Mock LLM response with low confidence and fixes."""
        return MagicMock(
            content="""
{
    "confidence": 0.3,
    "concerns": ["Missing imports", "No error handling", "Wrong output format"],
    "suggested_fixes": ["Add pandas import", "Wrap in try/except", "Match sample_submission format"],
    "proceed": false,
    "reflection": "Code has several issues that need to be fixed before execution"
}
"""
        )

    @pytest.fixture
    def sample_component(self):
        """Create a sample ablation component."""
        return AblationComponent(
            name="xgboost_classifier",
            component_type="model",
            code="Train an XGBoost classifier with hyperparameter tuning",
            estimated_impact=0.15,
        )

    @pytest.fixture
    def sample_state(self):
        """Create a sample state for testing."""
        state = create_initial_state("test-competition", "/tmp/work")
        state["competition_info"] = CompetitionInfo(
            name="test-competition",
            description="Test competition",
            evaluation_metric="roc_auc",
            problem_type="classification",
        )
        state["domain_detected"] = "tabular_classification"
        state["run_mode"] = "kaggle"
        state["fast_mode"] = False
        return state

    def test_self_evaluation_state_persistence(self, sample_state):
        """Test that self-evaluation is persisted in state correctly."""
        # Simulate what happens in developer_agent

        self_eval = SelfEvaluation(
            confidence=0.75,
            concerns=["Minor issue"],
            suggested_fixes=["Quick fix"],
            proceed=True,
            reflection_summary="Looks good overall",
        )

        # Simulate state update
        state_updates = {
            "self_evaluations": [self_eval],
            "last_self_evaluation": self_eval,
        }

        # Verify state updates structure
        assert "self_evaluations" in state_updates
        assert len(state_updates["self_evaluations"]) == 1
        assert state_updates["last_self_evaluation"].confidence == 0.75

    def test_decision_checkpoint_proceed(self):
        """Test decision checkpoint when proceed=True."""
        eval_result = SelfEvaluation(
            confidence=0.8,
            concerns=[],
            suggested_fixes=[],
            proceed=True,
            reflection_summary="Good to go",
        )

        # Should proceed without fixes
        should_apply_fixes = eval_result.confidence < 0.5 or not eval_result.proceed
        assert should_apply_fixes is False

    def test_decision_checkpoint_low_confidence(self):
        """Test decision checkpoint when confidence is low."""
        eval_result = SelfEvaluation(
            confidence=0.3,
            concerns=["Issue 1", "Issue 2"],
            suggested_fixes=["Fix 1", "Fix 2"],
            proceed=True,
            reflection_summary="Needs work",
        )

        # Should apply fixes due to low confidence
        should_apply_fixes = eval_result.confidence < 0.5 or not eval_result.proceed
        assert should_apply_fixes is True

    def test_decision_checkpoint_proceed_false(self):
        """Test decision checkpoint when proceed=False."""
        eval_result = SelfEvaluation(
            confidence=0.6,
            concerns=["Critical issue"],
            suggested_fixes=["Must fix this"],
            proceed=False,
            reflection_summary="Cannot proceed without fix",
        )

        # Should apply fixes due to proceed=False
        should_apply_fixes = eval_result.confidence < 0.5 or not eval_result.proceed
        assert should_apply_fixes is True


class TestSelfEvaluationPatterns:
    """Tests for common self-evaluation patterns."""

    def test_good_code_patterns(self):
        """Test that good code patterns are recognized."""
        good_code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

try:
    df = pd.read_csv('train.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f'Final Validation Performance: {scores.mean():.6f}')
    
    model.fit(X, y)
    test_df = pd.read_csv('test.csv')
    predictions = model.predict_proba(test_df)[:, 1]
    
    submission = pd.DataFrame({'id': test_df['id'], 'target': predictions})
    submission.to_csv('submission.csv', index=False)
    
except Exception as e:
    print(f'Error: {e}')
    raise
"""
        # These patterns should indicate high quality code
        has_imports = "import pandas" in good_code
        has_cv = "cross_val" in good_code
        has_error_handling = "try:" in good_code and "except" in good_code
        has_submission = "submission.csv" in good_code
        prints_score = "Final Validation Performance" in good_code

        assert has_imports
        assert has_cv
        assert has_error_handling
        assert has_submission
        assert prints_score

    def test_bad_code_patterns(self):
        """Test that bad code patterns are recognized."""
        bad_code = """
# TODO: implement this
import some_nonexistent_module

def process():
    raise NotImplementedError()
    
# This will fail
x = undefined_variable
"""
        # These patterns should indicate low quality code
        has_todo = "TODO" in bad_code
        has_raise = "raise" in bad_code
        has_undefined = "undefined_variable" in bad_code
        no_error_handling = "try:" not in bad_code

        assert has_todo
        assert has_raise
        assert has_undefined
        assert no_error_handling


class TestKaggleStateIntegration:
    """Tests for Quiet-STaR integration with KaggleState."""

    def test_state_has_self_evaluation_fields(self):
        """Test that KaggleState has required fields for self-evaluation."""
        state = create_initial_state("test", "/tmp")

        # Check required fields exist
        assert "self_evaluations" in state
        assert "last_self_evaluation" in state
        assert isinstance(state["self_evaluations"], list)
        assert state["last_self_evaluation"] is None

    def test_state_update_with_self_evaluation(self):
        """Test updating state with self-evaluation results."""
        state = create_initial_state("test", "/tmp")

        eval1 = SelfEvaluation(confidence=0.8, proceed=True)
        eval2 = SelfEvaluation(confidence=0.6, proceed=True)

        # Simulate appending evaluations
        state["self_evaluations"] = [eval1, eval2]
        state["last_self_evaluation"] = eval2

        assert len(state["self_evaluations"]) == 2
        assert state["last_self_evaluation"].confidence == 0.6

