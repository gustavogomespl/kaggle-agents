"""Tests for the log parser module."""

import pytest

from kaggle_agents.utils.log_parser import (
    TrainingFeedback,
    format_feedback_for_llm,
    parse_training_logs,
)


class TestLogParser:
    """Tests for parse_training_logs function."""

    def test_parse_fold_logs(self):
        """Test parsing [LOG:FOLD] entries."""
        stdout = """
[LOG:FOLD] fold=1 score=0.8732 time=45.23
[LOG:FOLD] fold=2 score=0.8756 time=43.12
[LOG:FOLD] fold=3 score=0.8721 time=44.56
[LOG:FOLD] fold=4 score=0.8745 time=42.89
[LOG:FOLD] fold=5 score=0.8738 time=45.01
"""
        feedback = parse_training_logs(stdout)

        assert len(feedback.fold_scores) == 5
        assert feedback.fold_scores[0] == pytest.approx(0.8732)
        assert feedback.fold_scores[1] == pytest.approx(0.8756)
        assert feedback.cv_mean == pytest.approx(0.87384, rel=1e-3)

    def test_parse_optuna_logs(self):
        """Test parsing [LOG:OPTUNA] entries."""
        stdout = """
[LOG:OPTUNA] trial=1 score=0.8650 time=30.5 params={'learning_rate': 0.05, 'max_depth': 6}
[LOG:OPTUNA] trial=2 score=0.8720 time=28.3 params={'learning_rate': 0.03, 'max_depth': 7}
[LOG:OPTUNA] trial=3 score=0.8680 time=32.1 params={'learning_rate': 0.08, 'max_depth': 5}
"""
        feedback = parse_training_logs(stdout)

        assert len(feedback.optuna_trials) == 3
        assert feedback.best_optuna_trial is not None
        assert feedback.best_optuna_trial["trial"] == 2
        assert feedback.best_optuna_trial["score"] == pytest.approx(0.8720)

    def test_parse_timing_logs(self):
        """Test parsing [LOG:TIMING] entries."""
        stdout = """
[LOG:TIMING] step=DATA_LOADING time=2.50 cumulative=2.50
[LOG:TIMING] step=PREPROCESSING time=12.30 cumulative=14.80
[LOG:TIMING] step=MODEL_TRAINING time=180.50 cumulative=195.30
"""
        feedback = parse_training_logs(stdout)

        assert len(feedback.timing_breakdown) == 3
        assert feedback.timing_breakdown["DATA_LOADING"] == pytest.approx(2.50)
        assert feedback.timing_breakdown["MODEL_TRAINING"] == pytest.approx(180.50)
        assert feedback.slowest_step == "MODEL_TRAINING"
        assert feedback.total_time == pytest.approx(195.30)

    def test_parse_features_logs(self):
        """Test parsing [LOG:FEATURES] entries."""
        stdout = """
[LOG:FEATURES] top=['feature1', 'feature2', 'feature3'] importances=[0.15, 0.12, 0.08]
"""
        feedback = parse_training_logs(stdout)

        assert len(feedback.top_features) == 3
        assert feedback.top_features[0] == "feature1"
        assert feedback.feature_importances[0] == pytest.approx(0.15)

    def test_parse_memory_logs(self):
        """Test parsing [LOG:MEMORY] entries."""
        stdout = """
[LOG:MEMORY] current_mb=1250.5 peak_mb=1800.2
"""
        feedback = parse_training_logs(stdout)

        assert feedback.memory_current_mb == pytest.approx(1250.5)
        assert feedback.memory_peak_mb == pytest.approx(1800.2)

    def test_parse_hyperparams_logs(self):
        """Test parsing [LOG:HYPERPARAMS] entries."""
        stdout = """
[LOG:HYPERPARAMS] params={'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 7}
"""
        feedback = parse_training_logs(stdout)

        assert feedback.hyperparams is not None
        assert "n_estimators" in feedback.hyperparams or "learning_rate" in feedback.hyperparams

    def test_parse_cv_summary(self):
        """Test parsing [LOG:CV_SUMMARY] entries."""
        stdout = """
[LOG:CV_SUMMARY] mean=0.8732 std=0.0085 scores=[0.871, 0.875, 0.872, 0.874, 0.873]
"""
        feedback = parse_training_logs(stdout)

        assert feedback.cv_mean == pytest.approx(0.8732)
        assert feedback.cv_std == pytest.approx(0.0085)

    def test_parse_warning_logs(self):
        """Test parsing [LOG:WARNING] entries."""
        stdout = """
[LOG:WARNING] message=High variance detected across folds (std > 0.02)
[LOG:WARNING] message=GPU memory exhausted, falling back to CPU
"""
        feedback = parse_training_logs(stdout)

        assert len(feedback.warnings) == 2
        assert "High variance" in feedback.warnings[0]

    def test_parse_error_logs(self):
        """Test parsing [LOG:ERROR] entries."""
        stdout = """
[LOG:ERROR] message=Failed to load model checkpoint
"""
        feedback = parse_training_logs(stdout)

        assert len(feedback.errors) == 1
        assert "Failed to load" in feedback.errors[0]

    def test_parse_complete_output(self):
        """Test parsing a complete training output with all log types."""
        stdout = """
Loading data...
[LOG:TIMING] step=DATA_LOADING time=2.50 cumulative=2.50
[LOG:MEMORY] current_mb=500.0 peak_mb=500.0

Starting Optuna tuning...
[LOG:OPTUNA] trial=1 score=0.8650 time=30.5 params={'lr': 0.05}
[LOG:OPTUNA] trial=2 score=0.8720 time=28.3 params={'lr': 0.03}

Training with 5-fold CV...
[LOG:FOLD] fold=1 score=0.8732 time=45.23
[LOG:FOLD] fold=2 score=0.8756 time=43.12
[LOG:FOLD] fold=3 score=0.8721 time=44.56
[LOG:FOLD] fold=4 score=0.8745 time=42.89
[LOG:FOLD] fold=5 score=0.8738 time=45.01

[LOG:CV_SUMMARY] mean=0.8738 std=0.0012 scores=[0.8732, 0.8756, 0.8721, 0.8745, 0.8738]
[LOG:HYPERPARAMS] params={'n_estimators': 1000, 'lr': 0.03}
[LOG:FEATURES] top=['feat1', 'feat2'] importances=[0.15, 0.12]
[LOG:TIMING] step=MODEL_TRAINING time=220.81 cumulative=223.31
[LOG:MEMORY] current_mb=1200.0 peak_mb=1800.0

Final Validation Performance: 0.873800
"""
        feedback = parse_training_logs(stdout)

        assert feedback.has_data()
        assert len(feedback.fold_scores) == 5
        assert len(feedback.optuna_trials) == 2
        assert feedback.cv_mean == pytest.approx(0.8738)
        assert feedback.slowest_step == "MODEL_TRAINING"

    def test_has_data_empty(self):
        """Test has_data returns False for empty feedback."""
        feedback = TrainingFeedback()
        assert not feedback.has_data()

    def test_has_data_with_content(self):
        """Test has_data returns True when data is present."""
        feedback = TrainingFeedback(fold_scores=[0.85, 0.86, 0.87])
        assert feedback.has_data()


class TestImprovementSuggestions:
    """Tests for improvement suggestions generation."""

    def test_high_variance_suggestion(self):
        """Test suggestion for high CV variance."""
        feedback = TrainingFeedback(
            fold_scores=[0.85, 0.92, 0.80, 0.88, 0.83],
            cv_mean=0.856,
            cv_std=0.045,
        )

        suggestions = feedback.get_improvement_suggestions()

        assert any("variance" in s.lower() for s in suggestions)

    def test_low_score_suggestion(self):
        """Test suggestion for low CV score."""
        feedback = TrainingFeedback(
            fold_scores=[0.45, 0.48, 0.46, 0.47, 0.44],
            cv_mean=0.46,
            cv_std=0.015,
        )

        suggestions = feedback.get_improvement_suggestions()

        assert any("low" in s.lower() and "score" in s.lower() for s in suggestions)

    def test_zero_importance_features_suggestion(self):
        """Test suggestion for zero-importance features."""
        feedback = TrainingFeedback(
            zero_importance_features=["useless_feat1", "useless_feat2"],
        )

        suggestions = feedback.get_improvement_suggestions()

        assert any("zero-importance" in s.lower() for s in suggestions)


class TestFormatFeedbackForLLM:
    """Tests for format_feedback_for_llm function."""

    def test_format_with_full_data(self):
        """Test formatting with complete training data."""
        feedback = TrainingFeedback(
            fold_scores=[0.8732, 0.8756, 0.8721, 0.8745, 0.8738],
            cv_mean=0.8738,
            cv_std=0.0012,
            optuna_trials=[
                {"trial": 1, "score": 0.865, "params": {"lr": 0.05}},
                {"trial": 2, "score": 0.872, "params": {"lr": 0.03}},
            ],
            best_optuna_trial={"trial": 2, "score": 0.872, "params": {"lr": 0.03}},
            timing_breakdown={"DATA_LOADING": 2.5, "MODEL_TRAINING": 180.0},
            total_time=182.5,
            slowest_step="MODEL_TRAINING",
            memory_peak_mb=1800.0,
            memory_current_mb=1200.0,
        )

        formatted = format_feedback_for_llm(feedback)

        assert "## Training Results Analysis" in formatted
        assert "CV Performance" in formatted
        assert "0.8738" in formatted
        assert "Optuna" in formatted
        assert "MODEL_TRAINING" in formatted

    def test_format_with_warnings(self):
        """Test formatting includes warnings."""
        feedback = TrainingFeedback(
            warnings=["High variance detected"],
            errors=["GPU memory exhausted"],
        )

        formatted = format_feedback_for_llm(feedback)

        assert "Issues Detected" in formatted
        assert "High variance" in formatted
        assert "GPU memory" in formatted

    def test_format_empty_feedback(self):
        """Test formatting with empty feedback still produces valid output."""
        feedback = TrainingFeedback()

        formatted = format_feedback_for_llm(feedback)

        assert "## Training Results Analysis" in formatted
