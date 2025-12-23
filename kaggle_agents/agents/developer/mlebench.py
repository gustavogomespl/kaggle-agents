"""
MLE-Bench integration for medal optimization.

Provides grading capabilities using the mlebench CLI tool for
evaluating submissions and determining medal achievement.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ...core.config import is_metric_minimization
from ...core.state import KaggleState


class MLEBenchMixin:
    """Mixin providing MLE-bench grading integration."""

    def _grade_with_mlebench(
        self,
        *,
        competition_name: str,
        submission_path: Path,
    ) -> dict[str, Any]:
        """Grade a submission locally using the `mlebench grade-sample` CLI."""
        if shutil.which("mlebench") is None:
            return {
                "valid_submission": False,
                "error": "mlebench CLI not found in PATH",
            }

        try:
            result = subprocess.run(
                ["mlebench", "grade-sample", str(submission_path), competition_name],
                capture_output=True,
                text=True,
                timeout=60,
            )

            output = (result.stdout or "") + (result.stderr or "")

            # Extract the first JSON object from output (mlebench prints extra lines)
            json_start = output.find("{")
            json_end = output.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    parsed = json.loads(output[json_start:json_end])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

            return {
                "valid_submission": False,
                "error": f"Could not parse mlebench output (exit={result.returncode}): {output[:500]}",
            }

        except subprocess.TimeoutExpired:
            return {
                "valid_submission": False,
                "error": "MLE-bench grading timeout (60s)",
            }
        except Exception as e:
            return {
                "valid_submission": False,
                "error": str(e),
            }

    def _should_stop_on_mlebench_grade(
        self,
        *,
        grading: dict[str, Any],
        state: KaggleState,
        metric_name: str,
    ) -> bool:
        """Decide whether to stop implementing more components based on MLE-bench grading."""
        if not grading.get("valid_submission"):
            return False

        # Medal achieved -> always stop.
        if any(grading.get(m) for m in ["gold_medal", "silver_medal", "bronze_medal"]):
            return True

        score = grading.get("score")
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = None

        target_score = state.get("target_score")
        if isinstance(target_score, str):
            try:
                target_score = float(target_score)
            except ValueError:
                target_score = None

        if isinstance(score, (int, float)) and isinstance(target_score, (int, float)):
            if is_metric_minimization(metric_name):
                if float(score) <= float(target_score):
                    return True
            else:
                if float(score) >= float(target_score):
                    return True

        return False
