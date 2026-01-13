"""
Failure analysis and error classification for Meta-Evaluator.

Contains methods for analyzing component failures and execution logs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from ...utils.llm_utils import get_text_content
from .prompts import SEMANTIC_LOG_ANALYSIS_PROMPT


if TYPE_CHECKING:
    from ...core.state import KaggleState


class AnalysisMixin:
    """Mixin providing failure analysis and error classification methods."""

    def _analyze_failures(self, state: KaggleState) -> dict[str, Any]:
        """
        Analyze component failures and extract patterns (PREFACE pattern).

        Args:
            state: Current workflow state

        Returns:
            Failure analysis with error patterns and success patterns
        """
        print("\n   🔍 Analyzing component failures...")

        dev_results = state.get("development_results", [])
        ablation_plan = state.get("ablation_plan", [])

        if not dev_results:
            return {
                "failed_components": [],
                "success_components": [],
                "error_patterns": [],
                "success_patterns": [],
                "by_type": {},
            }

        analysis = {
            "failed_components": [],
            "success_components": [],
            "error_patterns": set(),
            "success_patterns": set(),
            "by_type": {},
        }

        # Analyze each component result
        for i, result in enumerate(dev_results):
            component = ablation_plan[i] if i < len(ablation_plan) else None
            component_type = component.component_type if component else "unknown"
            component_name = component.name if component else f"component_{i}"

            if not result.success:
                # Extract error information
                error_msg = result.errors[0] if result.errors else result.stderr[:200]
                error_type = self._classify_error(error_msg)

                analysis["failed_components"].append(
                    {
                        "name": component_name,
                        "type": component_type,
                        "error": error_msg,
                        "error_type": error_type,
                        "execution_time": result.execution_time,
                    }
                )

                # Track error pattern
                analysis["error_patterns"].add(error_type)

                # Track by component type
                if component_type not in analysis["by_type"]:
                    analysis["by_type"][component_type] = {
                        "failures": 0,
                        "successes": 0,
                        "common_errors": [],
                    }
                analysis["by_type"][component_type]["failures"] += 1
                if error_type not in analysis["by_type"][component_type]["common_errors"]:
                    analysis["by_type"][component_type]["common_errors"].append(error_type)

            else:
                # Track success
                analysis["success_components"].append(
                    {
                        "name": component_name,
                        "type": component_type,
                        "execution_time": result.execution_time,
                    }
                )

                # Track success pattern
                success_pattern = f"{component_type}_success"
                analysis["success_patterns"].add(success_pattern)

                # Track by component type
                if component_type not in analysis["by_type"]:
                    analysis["by_type"][component_type] = {
                        "failures": 0,
                        "successes": 0,
                        "common_errors": [],
                    }
                analysis["by_type"][component_type]["successes"] += 1

        # Convert sets to lists for serialization
        analysis["error_patterns"] = list(analysis["error_patterns"])
        analysis["success_patterns"] = list(analysis["success_patterns"])

        # Print summary
        total = len(dev_results)
        success_count = len(analysis["success_components"])
        failed_count = len(analysis["failed_components"])
        success_rate = (success_count / total * 100) if total > 0 else 0

        print(f"   ✅ Success: {success_count}/{total} ({success_rate:.1f}%)")
        print(f"   ❌ Failed: {failed_count}/{total}")
        if analysis["error_patterns"]:
            print(f"   📋 Error patterns: {', '.join(analysis['error_patterns'])}")

        return analysis

    def _classify_error(self, error_msg: str) -> str:
        """
        Classify error type from error message.

        Now uses ROOT CAUSE analysis to differentiate data alignment errors
        from resource errors (e.g., timeout, memory) that may have data
        alignment as their actual root cause.

        Args:
            error_msg: Error message

        Returns:
            Error classification
        """
        if not error_msg:
            return "unknown_error"

        error_lower = error_msg.lower()

        # ===== PRIORITY 1: Data alignment errors (often misclassified) =====
        # Check for data alignment issues FIRST before other classifications
        data_alignment_patterns = [
            "shape mismatch",
            "dimension mismatch",
            "broadcast",
            "shapes do not match",
            "could not broadcast",
            "operands could not be broadcast",
            "inconsistent number of samples",
            "number of features",
            "oof.*mismatch",
            "prediction.*alignment",
        ]
        for pattern in data_alignment_patterns:
            if pattern in error_lower:
                return "data_alignment"

        # ===== LightGBM/XGBoost/CatBoost specific errors =====
        if "best gain: -inf" in error_lower:
            return "lightgbm_split_failure"
        if "no more leaves" in error_lower:
            return "lightgbm_leaf_constraint"
        if "no valid split" in error_lower:
            return "xgboost_split_failure"
        if "min_child" in error_lower or "min_data_in_leaf" in error_lower:
            return "hyperparameter_constraint"
        if "can't calculate leaf values" in error_lower:
            return "catboost_leaf_failure"
        if "not enough samples for bootstrap" in error_lower:
            return "catboost_bootstrap_failure"

        # Neural network specific errors
        if "cuda out of memory" in error_lower:
            return "gpu_oom"
        if "exploding gradient" in error_lower or "gradient explosion" in error_lower:
            return "exploding_gradients"
        if "vanishing gradient" in error_lower:
            return "vanishing_gradients"
        if "nan" in error_lower and ("loss" in error_lower or "gradient" in error_lower):
            return "nn_nan_loss"

        # Common error patterns
        if "importerror" in error_lower or "modulenotfounderror" in error_lower:
            return "import_error"
        if "filenotfounderror" in error_lower or "no such file" in error_lower:
            return "file_not_found"
        if "keyerror" in error_lower:
            return "key_error"
        if "valueerror" in error_lower:
            # Check for data alignment in ValueError
            if "shape" in error_lower or "dimension" in error_lower:
                return "data_alignment"  # Changed from dimension_mismatch
            if "nan" in error_lower or "infinity" in error_lower:
                return "data_contains_nans"
            return "value_error"
        if "typeerror" in error_lower:
            return "type_error"
        if "memoryerror" in error_lower or "out of memory" in error_lower:
            # Check if memory error might be caused by data alignment
            if any(p in error_lower for p in ["shape", "dimension", "broadcast"]):
                return "data_alignment"
            return "memory_error"
        if "timeout" in error_lower or "timed out" in error_lower:
            # Check if timeout might be caused by data alignment (stacking wrong shapes)
            if any(p in error_lower for p in ["shape", "dimension", "broadcast", "stacking"]):
                return "data_alignment"
            return "timeout_error"
        if "syntaxerror" in error_lower:
            return "syntax_error"
        if "attributeerror" in error_lower:
            return "attribute_error"
        if "indexerror" in error_lower:
            # Index errors often indicate data alignment issues
            if "out of bounds" in error_lower:
                return "data_alignment"
            return "index_error"
        if "validation failed" in error_lower:
            return "validation_error"
        if "final validation performance" in error_lower:
            return "missing_output_format"
        if "convergence" in error_lower and "warning" in error_lower:
            return "convergence_warning"
        return "runtime_error"

    def _classify_error_root_cause(self, error_message: str, component_type: str = "model") -> dict:
        """
        Classify an error by its ROOT CAUSE, not just the symptom.

        This is a more detailed version that returns additional context
        beyond just the error type string.

        Args:
            error_message: The error message to classify
            component_type: Type of component (model, ensemble, etc.)

        Returns:
            Dict with root_cause, category, priority, is_data_error, and suggested_fix
        """
        try:
            from ...nodes.curriculum_learning import classify_error_root_cause
            return classify_error_root_cause(error_message, component_type)
        except ImportError:
            # Fallback if import fails
            error_type = self._classify_error(error_message)
            is_data_error = error_type in ["data_alignment", "dimension_mismatch"]
            return {
                "root_cause": error_type,
                "category": "data" if is_data_error else "unknown",
                "priority": 1 if is_data_error else 3,
                "is_data_error": is_data_error,
                "suggested_fix": "Check data alignment with canonical train_ids." if is_data_error else "Debug and fix.",
            }

    def _analyze_execution_logs(self, state: KaggleState) -> dict[str, Any]:
        """
        Analyze execution logs using LLM for semantic error detection.

        Uses the LLM to parse stdout/stderr and identify model training
        issues, providing dynamic and context-aware feedback.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with detected issues and remediation guidance
        """
        dev_results = state.get("development_results", [])

        if not dev_results:
            return {
                "detected_issues": [],
                "planner_directives": [],
                "developer_directives": [],
                "has_semantic_errors": False,
                "severity": "info",
                "summary": "No execution results to analyze.",
            }

        # Collect logs from all results (limit to avoid token overflow)
        all_logs = []
        code_summaries = []

        for i, result in enumerate(dev_results[-3:]):  # Last 3 components
            stdout = (result.stdout or "")[-2000:]  # Last 2000 chars
            stderr = (result.stderr or "")[-1000:]  # Last 1000 chars
            logs = f"=== Component {i + 1} ===\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            all_logs.append(logs)

            # Code summary (first 30 lines + last 10 lines)
            code_lines = (result.code or "").split("\n")
            if len(code_lines) > 50:
                code_summary = "\n".join(code_lines[:30]) + "\n...\n" + "\n".join(code_lines[-10:])
            else:
                code_summary = result.code or ""
            code_summaries.append(code_summary[:1500])

        combined_logs = "\n\n".join(all_logs)[-6000:]  # Limit total logs
        combined_code = "\n---\n".join(code_summaries)[-3000:]

        # Use LLM to analyze logs
        prompt = SEMANTIC_LOG_ANALYSIS_PROMPT.format(
            logs=combined_logs,
            code_summary=combined_code,
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = get_text_content(response.content).strip()

            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            analysis = json.loads(content)

            # Ensure all expected keys exist
            analysis.setdefault("detected_issues", [])
            analysis.setdefault("planner_directives", [])
            analysis.setdefault("developer_directives", [])
            analysis.setdefault("severity", "info")
            analysis.setdefault("summary", "")
            analysis["has_semantic_errors"] = len(analysis["detected_issues"]) > 0

            # Print summary
            if analysis["has_semantic_errors"]:
                print(f"   ⚠️  Semantic Analysis: {analysis['summary']}")
                for issue in analysis["detected_issues"][:3]:
                    print(
                        f"      - {issue.get('pattern', 'Unknown')}: {issue.get('root_cause', '')}"
                    )

            return analysis

        except Exception as e:
            print(f"   ⚠️  LLM log analysis failed: {e}")
            return {
                "detected_issues": [],
                "planner_directives": [],
                "developer_directives": [],
                "has_semantic_errors": False,
                "severity": "info",
                "summary": f"Log analysis failed: {e}",
            }
