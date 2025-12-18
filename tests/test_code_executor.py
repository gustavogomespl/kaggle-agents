"""Tests for the code executor error parsing."""

from kaggle_agents.tools.code_executor import CodeExecutor


def test_parse_errors_ignores_tqdm_progress_on_stderr():
    # Avoid instantiating CodeExecutor to keep the test lightweight and free of
    # external configuration side-effects.
    executor = CodeExecutor.__new__(CodeExecutor)

    stderr = (
        "Fold0 Train Epoch1:   0%|          | 0/275 [00:00<?, ?it/s]\n"
        "Fold0 Train Epoch1:   0%|          | 0/275 [00:02<?, ?it/s, loss=1.79]\n"
        "Fold0 Train Epoch1:   0%|          | 1/275 [00:02<10:37,  2.33s/it, loss=1.79]\n"
    )

    assert executor._parse_errors(stderr=stderr, stdout="") == []


def test_parse_errors_still_detects_traceback_with_tqdm_noise():
    executor = CodeExecutor.__new__(CodeExecutor)

    stderr = (
        "Validation:   1%|▏         | 2/138 [00:02<02:28,  1.09s/it]\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "    raise ValueError('boom')\n"
        "ValueError: boom\n"
    )

    errors = executor._parse_errors(stderr=stderr, stdout="")
    assert errors, "Expected at least one parsed error"
    assert any("Value" in e or "boom" in e for e in errors)
