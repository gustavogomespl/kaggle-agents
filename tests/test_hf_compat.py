"""Tests for HuggingFace compatibility utilities."""

from unittest.mock import patch

import pytest


class TestGetTransformersVersion:
    """Tests for transformers version detection."""

    def test_parses_simple_version(self):
        """Should parse simple version strings."""
        from kaggle_agents.utils.hf_compat import get_transformers_version

        with patch("importlib.metadata.version", return_value="4.38.0"):
            version = get_transformers_version()
            assert version == (4, 38, 0)

    def test_parses_version_with_suffix(self):
        """Should handle versions with alpha/beta/rc suffixes."""
        from kaggle_agents.utils.hf_compat import get_transformers_version

        with patch("importlib.metadata.version", return_value="4.38.0a0"):
            version = get_transformers_version()
            assert version == (4, 38, 0)

        with patch("importlib.metadata.version", return_value="4.38.0b1"):
            version = get_transformers_version()
            assert version == (4, 38, 0)

        with patch("importlib.metadata.version", return_value="4.38.0rc1"):
            version = get_transformers_version()
            assert version == (4, 38, 0)

    def test_parses_version_with_cuda_suffix(self):
        """Should handle versions with CUDA suffixes."""
        from kaggle_agents.utils.hf_compat import get_transformers_version

        with patch("importlib.metadata.version", return_value="4.38.0+cu118"):
            version = get_transformers_version()
            assert version == (4, 38, 0)

    def test_fallback_on_error(self):
        """Should return fallback version on error."""
        from kaggle_agents.utils.hf_compat import get_transformers_version

        with patch("importlib.metadata.version", side_effect=Exception("Not found")):
            version = get_transformers_version()
            assert version == (4, 30, 0)


class TestGetTrainingArgsKwargs:
    """Tests for version-appropriate training args."""

    def test_new_version_uses_eval_strategy(self):
        """New transformers versions should use eval_strategy."""
        from kaggle_agents.utils.hf_compat import get_training_args_kwargs

        with patch("kaggle_agents.utils.hf_compat.get_transformers_version", return_value=(4, 40, 0)):
            kwargs = get_training_args_kwargs(eval_strategy="steps", eval_steps=500)

            assert "eval_strategy" in kwargs
            assert "evaluation_strategy" not in kwargs
            assert kwargs["eval_strategy"] == "steps"
            assert kwargs["eval_steps"] == 500

    def test_old_version_uses_evaluation_strategy(self):
        """Old transformers versions should use evaluation_strategy."""
        from kaggle_agents.utils.hf_compat import get_training_args_kwargs

        with patch("kaggle_agents.utils.hf_compat.get_transformers_version", return_value=(4, 30, 0)):
            kwargs = get_training_args_kwargs(eval_strategy="steps", eval_steps=500)

            assert "evaluation_strategy" in kwargs
            assert "eval_strategy" not in kwargs
            assert kwargs["evaluation_strategy"] == "steps"
            assert kwargs["eval_steps"] == 500

    def test_boundary_version_438(self):
        """Version 4.38.0 should use new eval_strategy."""
        from kaggle_agents.utils.hf_compat import get_training_args_kwargs

        with patch("kaggle_agents.utils.hf_compat.get_transformers_version", return_value=(4, 38, 0)):
            kwargs = get_training_args_kwargs(eval_strategy="epoch")

            assert "eval_strategy" in kwargs
            assert kwargs["eval_strategy"] == "epoch"

    def test_version_437_uses_old(self):
        """Version 4.37.x should use old evaluation_strategy."""
        from kaggle_agents.utils.hf_compat import get_training_args_kwargs

        with patch("kaggle_agents.utils.hf_compat.get_transformers_version", return_value=(4, 37, 2)):
            kwargs = get_training_args_kwargs(eval_strategy="no")

            assert "evaluation_strategy" in kwargs
            assert kwargs["evaluation_strategy"] == "no"

    def test_passes_through_additional_kwargs(self):
        """Should pass through additional keyword arguments."""
        from kaggle_agents.utils.hf_compat import get_training_args_kwargs

        with patch("kaggle_agents.utils.hf_compat.get_transformers_version", return_value=(4, 40, 0)):
            kwargs = get_training_args_kwargs(
                eval_strategy="steps",
                eval_steps=500,
                learning_rate=1e-4,
                max_steps=2000,
            )

            assert kwargs["learning_rate"] == 1e-4
            assert kwargs["max_steps"] == 2000


class TestHfCompatCodeSnippet:
    """Tests for the code snippet for injection."""

    def test_snippet_is_valid_python(self):
        """The code snippet should be valid Python."""
        from kaggle_agents.utils.hf_compat import HF_COMPAT_CODE_SNIPPET

        # Should not raise SyntaxError
        compile(HF_COMPAT_CODE_SNIPPET, "<string>", "exec")

    def test_snippet_defines_expected_functions(self):
        """The code snippet should define expected functions."""
        from kaggle_agents.utils.hf_compat import HF_COMPAT_CODE_SNIPPET

        # Execute the snippet
        namespace = {}
        exec(HF_COMPAT_CODE_SNIPPET, namespace)

        assert "_get_hf_version" in namespace
        assert "_hf_eval_strategy_kwarg" in namespace
        assert callable(namespace["_get_hf_version"])
        assert callable(namespace["_hf_eval_strategy_kwarg"])
