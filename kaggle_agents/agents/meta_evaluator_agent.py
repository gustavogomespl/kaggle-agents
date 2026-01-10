"""
Meta-Evaluator Agent with Reinforcement Learning.

This agent analyzes code generation results and optimize prompts for other agents using RL techniques.

Based on:
- CodeRL+: Execution Semantics Alignment
- PREFACE: Error-guided prompt repair
- RLPrompt: Discrete prompt optimization
- ML-Agent: RL for ML engineering
"""

import json
import math
import os
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.config import calculate_score_improvement, get_config, get_llm_for_role
from ..core.state import IterationMemory, KaggleState, get_memory_summary_for_planning
from ..optimization import create_training_collector
from ..prompts.templates.developer_prompts import format_component_details
from ..prompts.templates.planner_prompts import get_domain_guidance
from ..utils.llm_utils import get_text_content


# ==================== Semantic Log Analysis Prompt ====================

SEMANTIC_LOG_ANALYSIS_PROMPT = """You are an expert ML engineer analyzing execution logs from a Kaggle competition pipeline.

## Execution Logs (stdout + stderr)
```
{logs}
```

## Code Summary
```python
{code_summary}
```

## Your Task
Analyze these logs to identify:
1. **Training Issues**: Any warnings or errors from ML libraries (LightGBM, XGBoost, CatBoost, sklearn, PyTorch, TensorFlow)
2. **Hyperparameter Problems**: Signs of misconfigured parameters (e.g., "best gain: -inf", "no valid split", memory issues)
3. **Data Issues**: NaN, missing values, shape mismatches, type errors
4. **Resource Issues**: Memory, timeout, GPU problems

## Response Format
Return a JSON object:
{{
    "detected_issues": [
        {{
            "pattern": "exact text pattern found in logs",
            "root_cause": "what is causing this issue",
            "diagnosis": "detailed explanation of why this happens",
            "solutions": ["solution 1", "solution 2", "solution 3"]
        }}
    ],
    "planner_directives": [
        "High-level directive for the Planner agent to avoid this issue in next iteration"
    ],
    "developer_directives": [
        "Specific code-level fixes the Developer agent should apply"
    ],
    "severity": "critical" | "warning" | "info",
    "summary": "1-2 sentence summary of the main issues found"
}}

If no issues found, return:
{{
    "detected_issues": [],
    "planner_directives": [],
    "developer_directives": [],
    "severity": "info",
    "summary": "No significant issues detected in execution logs."
}}

IMPORTANT: Be specific. Quote the exact error messages. Provide actionable solutions."""


# ==================== Meta-Evaluator Agent ====================


class MetaEvaluatorAgent:
    """
    Meta-agent that evaluates other agents and optimizes their prompts using RL.

    Features:
    - Analyzes code generation failures and successes
    - Extracts error patterns (PREFACE pattern)
    - Calculates reward signals (CodeRL+ pattern)
    - Generates refinement guidance for prompt optimization
    - Collects training data for DSPy optimization
    """

    def __init__(self):
        """Initialize meta-evaluator with configured model."""
        self.config = get_config()

        # Use configured LLM (supports OpenAI and Anthropic)
        self.llm = get_llm_for_role(role="evaluator")

        provider = self.config.llm.provider.upper()
        model = self.config.llm.model
        print(f"   🧠 Meta-Evaluator initialized with {provider} ({model})")

        # Training data collector for RL
        self.training_collector = create_training_collector()

    def __call__(self, state: KaggleState) -> dict[str, Any]:
        """
        Execute meta-evaluation after performance evaluation.

        Args:
            state: Current workflow state

        Returns:
            State updates with failure analysis and refinement guidance
        """
        print("\n" + "=" * 60)
        print("= META-EVALUATOR: Analyzing Performance & Optimizing Prompts")
        print("=" * 60)

        current_iteration = state.get("current_iteration", 0)
        print(f"\n📊 Iteration: {current_iteration}")

        # Analyze component performance
        failure_analysis = self._analyze_failures(state)

        # Calculate reward signals (CodeRL+ pattern)
        reward_signals = self._calculate_reward_signals(state, failure_analysis)

        # Generate refinement guidance (PREFACE pattern)
        refinement_guidance = self._generate_refinement_guidance(
            state, failure_analysis, reward_signals
        )

        # Create iteration memory for learning
        iteration_memory = self._create_iteration_memory(state, failure_analysis, reward_signals)

        # Collect training data for DSPy optimization
        self._collect_training_data(state, failure_analysis, reward_signals)

        # Eureka: Perform evolutionary crossover for next generation planning
        crossover_guidance = self._evolutionary_crossover(state)

        # Inner Loop Refinement: Check for performance gaps that need debug loops
        debug_loop_trigger = self._check_performance_gap_for_debug(state)

        # Detect stagnation for SOTA search trigger
        stagnation_detection = self._detect_stagnation(state)

        # Update state
        debug_updates = {}
        if debug_loop_trigger.get("trigger_debug"):
            debug_updates = {
                "trigger_debug_loop": True,
                "debug_target_model": debug_loop_trigger.get("worst_model"),
                "debug_hints": debug_loop_trigger.get("debug_hints", []),
                "performance_gap": debug_loop_trigger.get("gap"),
            }
            print(f"\n   ⚠️  TRIGGERING DEBUG LOOP for {debug_loop_trigger.get('worst_model')}")

        result = {
            "failure_analysis": failure_analysis,
            "reward_signals": reward_signals,
            "refinement_guidance": refinement_guidance,
            "crossover_guidance": crossover_guidance,  # Eureka: for planner
            "stagnation_detection": stagnation_detection,  # For SOTA search trigger
            "iteration_memory": [iteration_memory],  # Append to list
            "last_updated": datetime.now(),
        }
        result.update(debug_updates)  # Add debug loop trigger if applicable
        return result

    def _check_performance_gap_for_debug(self, state: KaggleState) -> dict[str, Any]:
        """
        Inner Loop Refinement: Detect when one model performs drastically worse.

        Triggers a dedicated debug iteration when:
        - Two or more models exist
        - Performance gap > 1.0 (for logloss-like metrics)

        This prevents moving forward with broken models in the ensemble.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with trigger_debug, worst_model, gap, debug_hints
        """
        dev_results = state.get("development_results", [])
        ablation_plan = state.get("ablation_plan", [])

        if not dev_results:
            return {"trigger_debug": False}

        model_scores = {}

        for i, result in enumerate(dev_results):
            # Get component info
            component = ablation_plan[i] if i < len(ablation_plan) else None
            component_type = component.component_type if component else "unknown"
            component_name = component.name if component else f"component_{i}"

            if component_type != "model":
                continue

            # Try to extract score from stdout
            # DevelopmentResult is a dataclass - use getattr for safety
            stdout = getattr(result, "stdout", "") or ""

            # Look for common score patterns
            patterns = [
                r"(?:CV|Validation|Val|OOF).*?(?:Score|Loss|logloss|LogLoss|RMSE|MAE|AUC).*?:\s*([\d.]+)",
                r"(?:Score|Loss|logloss|LogLoss).*?:\s*([\d.]+)",
                r"Final.*?(?:Score|Loss|Validation Performance).*?:\s*([\d.]+)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, stdout, re.IGNORECASE)
                if matches:
                    try:
                        model_scores[component_name] = float(matches[-1])
                        break
                    except ValueError:
                        continue

        # Need at least 2 models to compare
        if len(model_scores) < 2:
            return {"trigger_debug": False, "model_scores": model_scores}

        scores = list(model_scores.values())
        max_gap = max(scores) - min(scores)

        # For logloss (lower is better), gap > 1.0 is HUGE
        if max_gap > 1.0:
            worst_model = max(model_scores, key=model_scores.get)
            best_model = min(model_scores, key=model_scores.get)

            debug_hints = [
                "Check if LabelEncoder class order is consistent with other models",
                "Verify class_weight='balanced' is appropriate for this metric",
                "Compare data preprocessing between models",
                "Check if same train/val splits are used (random_state)",
                "Verify the objective function matches the competition metric",
                "Check for data type mismatches (categorical vs numeric)",
            ]

            print("\n   📊 PERFORMANCE GAP DETECTED:")
            print(f"      Worst: {worst_model} = {model_scores[worst_model]:.4f}")
            print(f"      Best: {best_model} = {model_scores[best_model]:.4f}")
            print(f"      Gap: {max_gap:.2f}")

            return {
                "trigger_debug": True,
                "worst_model": worst_model,
                "best_model": best_model,
                "gap": max_gap,
                "model_scores": model_scores,
                "debug_hints": debug_hints,
                "action": "PAUSE_AND_DEBUG",
            }

        if max_gap > 0.5:
            # Moderate gap - warning only
            print(f"\n   ⚠️  Moderate performance gap ({max_gap:.2f}) between models")
            return {
                "trigger_debug": False,
                "gap": max_gap,
                "model_scores": model_scores,
                "warning": f"Moderate gap of {max_gap:.2f} detected",
            }

        return {"trigger_debug": False, "model_scores": model_scores}

    def _detect_stagnation(self, state: KaggleState) -> dict[str, Any]:
        """
        Detect if progress has stagnated over recent iterations.

        Triggers SOTA search when:
        1. Stagnation: avg improvement < threshold over last N iterations
        2. Score gap: current score is far from target after minimum iterations

        Args:
            state: Current workflow state

        Returns:
            Dict with stagnation info and SOTA search trigger
        """
        iteration_memory = state.get("iteration_memory", [])
        current_iteration = state.get("current_iteration", 0)
        config = self.config.iteration

        # Get stagnation config
        stagnation_window = getattr(config, "stagnation_window", 3)
        stagnation_threshold = getattr(config, "stagnation_threshold", 0.01)
        score_gap_threshold = getattr(config, "score_gap_threshold", 0.3)

        result = {
            "stagnated": False,
            "trigger_sota_search": False,
            "reason": None,
            "avg_improvement": 0.0,
            "score_gap": 0.0,
            "iterations_checked": 0,
        }

        # Check stagnation: avg improvement over last N iterations
        # Only run if we have enough iterations for meaningful stagnation detection
        if len(iteration_memory) >= stagnation_window:
            recent_improvements = []
            for memory in iteration_memory[-stagnation_window:]:
                # IterationMemory is a dataclass, use attribute access (not dict.get())
                improvement = getattr(memory, "score_improvement", 0)
                if isinstance(improvement, (int, float)):
                    recent_improvements.append(abs(float(improvement)))

            if recent_improvements:
                avg_improvement = sum(recent_improvements) / len(recent_improvements)
                result["avg_improvement"] = avg_improvement
                result["iterations_checked"] = len(recent_improvements)

                # Stagnation: improvement below threshold
                if avg_improvement < stagnation_threshold:
                    result["stagnated"] = True
                    result["trigger_sota_search"] = True
                    result["reason"] = f"stagnation: avg_improvement={avg_improvement:.4f} < {stagnation_threshold}"
                    print(f"\n   📉 STAGNATION DETECTED: avg improvement {avg_improvement:.4f} over last {len(recent_improvements)} iterations")

        # Check score gap: far from target after minimum iterations
        # NOTE: This runs INDEPENDENTLY of stagnation check, even in early iterations
        if current_iteration >= 2:  # After 2 iterations
            current_score = state.get("current_performance_score", 0.0)
            target_score = state.get("target_score")

            if target_score and isinstance(target_score, (int, float)) and float(target_score) > 0:
                try:
                    score_gap = abs(float(target_score) - float(current_score)) / float(target_score)
                    result["score_gap"] = score_gap

                    if score_gap > score_gap_threshold:
                        result["trigger_sota_search"] = True
                        if result["reason"]:
                            result["reason"] += f" AND score_gap={score_gap:.1%} > {score_gap_threshold:.0%}"
                        else:
                            result["reason"] = f"score_gap: {score_gap:.1%} > {score_gap_threshold:.0%}"
                        print(f"\n   📊 SCORE GAP DETECTED: {score_gap:.1%} from target after {current_iteration} iterations")
                except (TypeError, ValueError):
                    pass

        if result["trigger_sota_search"]:
            print(f"   🔍 TRIGGERING SOTA SEARCH: {result['reason']}")

        return result

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
            from ..nodes.curriculum_learning import classify_error_root_cause
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

    def _detect_undertrained_models(
        self,
        state: KaggleState,
    ) -> dict[str, Any] | None:
        """
        Detect if model performance indicates insufficient training.

        Compares CV score against random baseline for the problem type,
        respecting the metric direction (minimize vs maximize).

        Args:
            state: Current workflow state

        Returns:
            Diagnostic dict if undertrained, None otherwise
        """
        from ..core.config import is_metric_minimization

        dev_results = state.get("development_results", [])
        if not dev_results:
            return None

        # Get the best CV score from successful results
        # Note: DevelopmentResult doesn't have cv_score attribute - extract from stdout
        cv_scores = []
        for result in dev_results:
            if result.success and result.stdout:
                # Extract CV score from stdout (pattern: "Final Validation Performance: X.XXXX")
                match = re.search(r"Final Validation Performance[:\s]+([0-9.]+)", result.stdout)
                if match:
                    try:
                        cv_scores.append(float(match.group(1)))
                    except ValueError:
                        pass

        if not cv_scores:
            return None

        # Determine metric and its direction
        competition_info = state.get("competition_info")
        metric_name = ""
        problem_type = ""
        n_classes = 2

        if competition_info:
            metric_name = str(getattr(competition_info, "evaluation_metric", "")).lower()
            problem_type = str(getattr(competition_info, "problem_type", "")).lower()

        # Determine if we're minimizing or maximizing
        is_minimize = is_metric_minimization(metric_name) if metric_name else True

        # Get best score based on metric direction
        if is_minimize:
            best_cv_score = min(cv_scores)
        else:
            best_cv_score = max(cv_scores)

        # Try to infer n_classes from sample submission
        sample_submission_path = state.get("sample_submission_path")
        if sample_submission_path:
            try:
                import pandas as pd
                sample_sub = pd.read_csv(sample_submission_path)
                n_cols = sample_sub.shape[1]
                if n_cols > 2:
                    n_classes = n_cols - 1  # Subtract ID column
            except Exception:
                pass

        # Calculate random baselines based on metric type
        if is_minimize:
            # Minimization metrics (log_loss, RMSE, etc.)
            random_baselines = {
                "multiclass": -math.log(1 / max(n_classes, 2)),  # log_loss for random
                "binary": 0.693,  # -log(0.5) for binary log_loss
            }
            baseline_key = "multiclass" if n_classes > 2 else "binary"
            baseline = random_baselines.get(baseline_key, 4.0)

            # For minimization: score > threshold * baseline means undertrained
            threshold = float(os.environ.get("KAGGLE_AGENTS_UNDERTRAINED_THRESHOLD", "0.85"))
            is_undertrained = best_cv_score > baseline * threshold
            comparison_msg = f"Score {best_cv_score:.4f} is too high (within {int((1-threshold)*100)}% of random baseline {baseline:.4f})"
        else:
            # Maximization metrics (accuracy, F1, AUC, etc.)
            random_baselines = {
                "accuracy_multiclass": 1.0 / max(n_classes, 2),  # random accuracy
                "accuracy_binary": 0.5,
                "auc": 0.5,  # random AUC
                "f1": 0.0,  # worst F1
            }

            # Determine appropriate baseline
            if "auc" in metric_name or "roc" in metric_name:
                baseline = 0.5
            elif "f1" in metric_name or "precision" in metric_name or "recall" in metric_name:
                baseline = 0.0
            else:
                # Default to accuracy baseline
                baseline = 1.0 / max(n_classes, 2)

            # For maximization: score < threshold * optimal means undertrained
            # Use a different threshold logic: if score is close to random baseline
            threshold = float(os.environ.get("KAGGLE_AGENTS_UNDERTRAINED_THRESHOLD", "0.85"))
            # For maximize metrics, undertrained means score is within 15% above baseline
            # e.g., for binary accuracy: baseline=0.5, threshold=0.85 → 0.5 + 0.15*(1-0.5) = 0.575
            undertrained_ceiling = baseline + (1 - threshold) * (1.0 - baseline)
            is_undertrained = best_cv_score < undertrained_ceiling
            comparison_msg = f"Score {best_cv_score:.4f} is too low (below {undertrained_ceiling:.4f}, near random baseline {baseline:.4f})"

        if is_undertrained:
            direction = "minimize" if is_minimize else "maximize"
            print(f"   ⚠️ UNDERTRAINED MODEL DETECTED ({direction}): {comparison_msg}")
            return {
                "type": "UNDERTRAINED_MODEL",
                "severity": "critical",
                "cv_score": best_cv_score,
                "random_baseline": baseline,
                "n_classes": n_classes,
                "metric_name": metric_name,
                "is_minimize": is_minimize,
                "message": comparison_msg,
                "suggestions": [
                    "Increase training epochs (model may not have converged)",
                    "Verify preprocessing matches model requirements (e.g., preprocess_input for pretrained models)",
                    "Check if learning rate is appropriate (may be too high or too low)",
                    "Ensure data augmentation isn't too aggressive",
                    "Verify class order alignment between predictions and ground truth labels",
                ],
                "planner_directive": "CRITICAL: Current model is near-random. Prioritize training convergence over new features.",
                "developer_directive": "CRITICAL: Model is undertrained. Check preprocessing, increase epochs, verify label encoding.",
            }

        return None

    def _calculate_reward_signals(
        self,
        state: KaggleState,
        failure_analysis: dict[str, Any],
    ) -> dict[str, float]:
        """
        Calculate reward signals for RL optimization (CodeRL+ pattern).

        Implements multi-faceted reward:
        - Functional correctness (execution success)
        - Performance (Kaggle score)
        - Code quality (execution semantics)

        Args:
            state: Current workflow state
            failure_analysis: Failure analysis results

        Returns:
            Reward signals dictionary
        """
        print("\n   💰 Calculating reward signals...")

        dev_results = state.get("development_results", [])
        submissions = state.get("submissions", [])
        # Prefer MLE-bench grading when available (enables medal-oriented rewards).
        mlebench_grade = state.get("mlebench_grade")
        current_score = state.get("current_performance_score", 0.0)
        if isinstance(mlebench_grade, dict) and mlebench_grade.get("valid_submission"):
            score = mlebench_grade.get("score")
            if isinstance(score, (int, float)):
                current_score = float(score)
        best_score = state.get("best_score", 0.0)
        run_mode = str(state.get("run_mode", "")).lower()
        objective = str(state.get("objective", "")).lower()

        # Reward 1: Functional Correctness (binary)
        total_components = len(dev_results)
        successful_components = len(failure_analysis["success_components"])
        r_functional = successful_components / total_components if total_components > 0 else 0.0

        # Reward 2: Performance (continuous, normalized 0-1)
        # Try to get dynamic target from state (e.g. from leaderboard), else default
        target_score = state.get("target_score")
        if target_score is None:
            target_score = 1.0
        elif isinstance(target_score, str):
            try:
                target_score = float(target_score)
            except (ValueError, TypeError):
                target_score = 1.0
        elif not isinstance(target_score, (int, float)):
            target_score = 1.0

        # Medal-aware shaping (MLE-bench objective)
        r_medal = 0.0
        if isinstance(mlebench_grade, dict) and mlebench_grade.get("valid_submission"):
            if mlebench_grade.get("gold_medal"):
                r_medal = 1.0
            elif mlebench_grade.get("silver_medal"):
                r_medal = 0.8
            elif mlebench_grade.get("bronze_medal"):
                r_medal = 0.6
            elif mlebench_grade.get("above_median"):
                r_medal = 0.4

        # Ensure current_score is numeric
        if isinstance(current_score, str):
            try:
                current_score = float(current_score)
            except (ValueError, TypeError):
                current_score = 0.0

        score_component = (
            min(float(current_score) / float(target_score), 1.0) if float(target_score) > 0 else 0.0
        )
        if run_mode == "mlebench" or "medal" in objective:
            # Blend medal attainment with raw score; medal dominates to keep the objective explicit.
            r_performance = min(0.7 * r_medal + 0.3 * score_component, 1.0)
        else:
            r_performance = score_component

        # Reward 3: Improvement (delta from previous best)
        # Get evaluation metric to handle both minimize and maximize metrics correctly
        competition_info = state.get("competition_info")
        metric_name = competition_info.evaluation_metric if competition_info else ""

        # Calculate improvement considering metric direction (positive = better)
        score_improvement = calculate_score_improvement(current_score, best_score, metric_name)
        r_improvement = max(0.0, min(score_improvement * 10, 1.0))  # Scale to 0-1

        # Reward 4: Execution Semantics (no errors, fast execution)
        avg_execution_time = (
            sum(r.execution_time for r in dev_results) / total_components
            if total_components > 0
            else 0.0
        )
        r_semantics = 1.0 - min(avg_execution_time / 300.0, 1.0)  # Normalize by 5min timeout

        # Reward 5: Diversity
        # Encourages trying different types of components (e.g. not just 5 XGBoosts)
        unique_types = len(
            {c.get("type", "unknown") for c in failure_analysis["success_components"]}
        )
        r_diversity = min(unique_types / 3.0, 1.0)  # Target: at least 3 different types working

        # Reward 6: Robustness/Overfitting Penalty
        # Penalize if Public LB score is much lower than Validation score
        validation_score = state.get("overall_validation_score", 0.0)
        public_score = 0.0
        if submissions:
            public_score = submissions[-1].public_score or 0.0

        # If we have both scores, check gap. If gap > 0.1, heavy penalty.
        gap = abs(validation_score - public_score)
        r_robustness = (
            1.0 - min(gap * 5, 1.0) if (validation_score > 0 and public_score > 0) else 1.0
        )

        # Combined reward (weighted)
        # Performance-focused weights: prioritize score improvement for aggressive optimization.
        # MLE-bench mode: speed and medal attainment matter more.
        if run_mode == "mlebench" or "medal" in objective:
            weights = {
                "functional": 0.15,      # Reduced: working code is baseline
                "performance": 0.50,     # Increased: medal achievement is key
                "improvement": 0.10,     # Increased: reward progress
                "semantics": 0.10,       # Reduced slightly
                "diversity": 0.05,       # Reduced: focus on what works
                "robustness": 0.10,      # Increased: prevent overfitting
            }
        else:
            # Standard Kaggle mode: heavily prioritize performance/score
            weights = {
                "functional": 0.15,      # Reduced from 0.25
                "performance": 0.55,     # Increased from 0.40 - main driver
                "improvement": 0.15,     # Increased from 0.10 - reward progress
                "semantics": 0.05,       # Maintained
                "diversity": 0.05,       # Reduced from 0.10
                "robustness": 0.05,      # Reduced from 0.10
            }

        r_combined = (
            weights["functional"] * r_functional
            + weights["performance"] * r_performance
            + weights["improvement"] * r_improvement
            + weights["semantics"] * r_semantics
            + weights["diversity"] * r_diversity
            + weights["robustness"] * r_robustness
        )

        rewards = {
            "r_functional": r_functional,
            "r_performance": r_performance,
            "r_improvement": r_improvement,
            "r_semantics": r_semantics,
            "r_diversity": r_diversity,
            "r_robustness": r_robustness,
            "r_medal": r_medal,
            "r_combined": r_combined,
        }

        print(
            f"   📊 Rewards: functional={r_functional:.2f}, performance={r_performance:.2f}, "
            f"diversity={r_diversity:.2f}, robustness={r_robustness:.2f}, combined={r_combined:.3f}"
        )

        return rewards

    def _generate_refinement_guidance(
        self,
        state: KaggleState,
        failure_analysis: dict[str, Any],
        reward_signals: dict[str, float],
    ) -> dict[str, str]:
        """
        Generate refinement guidance for prompt optimization (PREFACE pattern).

        Uses LLM to analyze failures and generate strategic guidance
        for improving prompts in next iteration. Integrates semantic
        log analysis for deeper insights.

        Args:
            state: Current workflow state
            failure_analysis: Failure analysis results
            reward_signals: Calculated rewards

        Returns:
            Refinement guidance dictionary
        """
        print("\n   🎯 Generating refinement guidance...")

        # Analyze execution logs for semantic errors (LLM-driven)
        log_analysis = self._analyze_execution_logs(state)

        # Detect undertrained models (critical for image/multiclass problems)
        undertrained_info = self._detect_undertrained_models(state)

        # Build context for LLM
        context = self._build_evaluation_context(state, failure_analysis, reward_signals)

        # Inject undertrained model detection into context (highest priority)
        if undertrained_info:
            context += "\n\n## ⚠️ CRITICAL: UNDERTRAINED MODEL DETECTED\n"
            context += f"**Severity**: {undertrained_info.get('severity', 'critical')}\n"
            context += f"**Message**: {undertrained_info.get('message', '')}\n"
            context += f"**CV Score**: {undertrained_info.get('cv_score', 0):.4f}\n"
            context += f"**Random Baseline**: {undertrained_info.get('random_baseline', 0):.4f}\n"
            context += f"**Classes**: {undertrained_info.get('n_classes', 2)}\n\n"
            context += "**Suggestions**:\n"
            for sugg in undertrained_info.get("suggestions", []):
                context += f"  - {sugg}\n"

        # Inject semantic analysis into context
        if log_analysis.get("has_semantic_errors"):
            context += "\n\n## Semantic Log Analysis (from LLM)\n"
            context += f"**Severity**: {log_analysis.get('severity', 'unknown')}\n"
            context += f"**Summary**: {log_analysis.get('summary', '')}\n\n"

            for issue in log_analysis.get("detected_issues", [])[:5]:
                context += f"### Issue: `{issue.get('pattern', 'Unknown')}`\n"
                context += f"- **Root Cause**: {issue.get('root_cause', '')}\n"
                context += f"- **Diagnosis**: {issue.get('diagnosis', '')}\n"
                context += "- **Solutions**:\n"
                for sol in issue.get("solutions", []):
                    context += f"  - {sol}\n"
                context += "\n"

            if log_analysis.get("planner_directives"):
                context += "**Planner Directives (from log analysis)**:\n"
                for directive in log_analysis["planner_directives"]:
                    context += f"- {directive}\n"

            if log_analysis.get("developer_directives"):
                context += "\n**Developer Directives (from log analysis)**:\n"
                for directive in log_analysis["developer_directives"]:
                    context += f"- {directive}\n"

        # Generate guidance using LLM
        prompt = self._build_refinement_prompt(context)

        messages = [
            SystemMessage(content=META_EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

        # Parse guidance from response
        try:
            guidance = json.loads(get_text_content(response.content))
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            guidance = {
                "planner_guidance": "Focus on high-impact components with proven track record.",
                "developer_guidance": "Ensure code follows all requirements and outputs correct format.",
                "priority_fixes": failure_analysis["error_patterns"],
            }

        # Inject semantic directives into guidance (high priority)
        if log_analysis.get("planner_directives"):
            semantic_guidance = " | ".join(log_analysis["planner_directives"])
            existing = guidance.get("planner_guidance", "")
            guidance["planner_guidance"] = f"PRIORITY: {semantic_guidance}. {existing}"

        if log_analysis.get("developer_directives"):
            existing_dev = guidance.get("developer_guidance", "")
            dev_directives = " | ".join(log_analysis["developer_directives"])
            guidance["developer_guidance"] = f"PRIORITY: {dev_directives}. {existing_dev}"

        # Inject undertrained model directives (HIGHEST priority - overrides everything)
        if undertrained_info:
            existing_planner = guidance.get("planner_guidance", "")
            guidance["planner_guidance"] = f"🔴 {undertrained_info.get('planner_directive', '')} {existing_planner}"

            existing_dev = guidance.get("developer_guidance", "")
            guidance["developer_guidance"] = f"🔴 {undertrained_info.get('developer_directive', '')} {existing_dev}"

            guidance["undertrained_analysis"] = undertrained_info

        # Store full analysis for downstream use
        guidance["semantic_analysis"] = log_analysis

        print("   ✓ Generated guidance for Planner and Developer")

        return guidance

    def _build_evaluation_context(
        self,
        state: KaggleState,
        failure_analysis: dict[str, Any],
        reward_signals: dict[str, float],
    ) -> str:
        """Build context string for LLM evaluation."""
        current_iteration = state.get("current_iteration", 0)
        run_mode = str(state.get("run_mode", "")).lower()
        objective = str(state.get("objective", "")).lower()
        mlebench_grade = state.get("mlebench_grade")

        current_score = state.get("current_performance_score", 0.0)
        if isinstance(mlebench_grade, dict) and mlebench_grade.get("valid_submission"):
            score = mlebench_grade.get("score")
            if isinstance(score, (int, float)):
                current_score = float(score)

        target_score = state.get("target_score")
        if target_score is None:
            target_score = 1.0
        elif isinstance(target_score, str):
            try:
                target_score = float(target_score)
            except (ValueError, TypeError):
                target_score = 1.0
        elif not isinstance(target_score, (int, float)):
            target_score = 1.0

        context = f"""# Iteration {current_iteration} Evaluation

## Objective
- run_mode: {run_mode or "kaggle"}
- objective: {objective or "top20"}

## Current Performance
- Score: {current_score:.4f}
- Target: {target_score:.4f}
- Gap: {target_score - current_score:.4f}

## Component Results
- Total: {len(state.get("development_results", []))}
- Successful: {len(failure_analysis["success_components"])}
- Failed: {len(failure_analysis["failed_components"])}

## Success Patterns
{chr(10).join("- " + p for p in failure_analysis["success_patterns"])}

## Error Patterns
{chr(10).join("- " + p for p in failure_analysis["error_patterns"])}
"""

        if isinstance(mlebench_grade, dict):
            medals = []
            if mlebench_grade.get("gold_medal"):
                medals.append("Gold")
            if mlebench_grade.get("silver_medal"):
                medals.append("Silver")
            if mlebench_grade.get("bronze_medal"):
                medals.append("Bronze")
            context += f"""
## MLE-bench Grading
- valid_submission: {bool(mlebench_grade.get("valid_submission", False))}
- score: {mlebench_grade.get("score")}
- above_median: {bool(mlebench_grade.get("above_median", False))}
- medals: {", ".join(medals) if medals else "None"}
"""

        # FULL CODE AND PERFORMANCE ANALYSIS
        context += "\n## Component Code and Performance Analysis\n"

        dev_results = state.get("development_results", [])

        # Limit to 5 most recent components to reduce token usage
        recent_results = dev_results[-5:] if len(dev_results) > 5 else dev_results
        if len(dev_results) > 5:
            context += f"*(Showing 5 most recent components out of {len(dev_results)} total)*\n\n"

        for i, res in enumerate(recent_results):
            # Extract score from stdout if possible
            score_match = re.search(r"Final Validation Performance: (0\.\d+)", res.stdout)
            score = float(score_match.group(1)) if score_match else "N/A"

            # Determine component name (heuristic)
            comp_name = f"Component_{i + 1}"
            if "class " in res.code:
                # Try to find class name
                class_match = re.search(r"class (\w+)", res.code)
                if class_match:
                    comp_name = class_match.group(1)

            status = "✅ Success" if res.success else "❌ Failed"

            context += f"\n### {comp_name}\n"
            context += f"**Status**: {status}\n"
            context += f"**Score**: {score}\n"
            context += f"**Execution Time**: {res.execution_time:.2f}s\n"

            if not res.success:
                context += f"**Error**: {res.stderr[-200:] if res.stderr else 'Unknown Error'}\n"

            # Send code summary instead of full code to reduce tokens
            code_lines = res.code.split("\n")
            context += "**Code Summary**:\n```python\n"
            context += "\n".join(code_lines[:20])  # First 20 lines
            if len(code_lines) > 30:
                context += "\n# ... (middle section omitted) ...\n"
                context += "\n".join(code_lines[-10:])  # Last 10 lines
            context += "\n```\n"
            context += f"**Total Lines**: {len(code_lines)}\n"
            stdout_tail = res.stdout[-3000:] if res.stdout else ""
            stderr_tail = res.stderr[-1500:] if res.stderr else ""
            if stdout_tail:
                context += "**STDOUT (tail)**:\n```text\n"
                context += stdout_tail
                context += "\n```\n"
            if stderr_tail:
                context += "**STDERR (tail)**:\n```text\n"
                context += stderr_tail
                context += "\n```\n"
            context += "-" * 40 + "\n"

        context += "\n## Reward Signals\n"
        for key, value in reward_signals.items():
            context += f"- {key}: {value:.3f}\n"

        return context

    def _build_refinement_prompt(self, context: str) -> str:
        """Build prompt for refinement guidance generation."""
        return f"""{context}

## Your Task
Analyze the above results and provide strategic guidance for improving prompts in the next iteration.

If the objective is `mlebench_medal`, explicitly prioritize:
- Achieving at least a Bronze medal (then Silver/Gold)
- Reducing wall-clock execution time (avoid expensive CV / over-sized models)
- Robustness and deterministic outputs (no flaky dependencies)

Return a JSON object with:
{{
  "planner_guidance": "Specific guidance for Planner agent on how to improve component selection",
  "developer_guidance": "Specific guidance for Developer agent on how to avoid errors",
  "priority_fixes": ["error_type_1", "error_type_2"],
  "success_amplification": ["what worked that should be emphasized"],
  "component_type_guidance": {{
    "model": "guidance for model components",
    "feature_engineering": "guidance for feature engineering",
    "ensemble": "guidance for ensemble components"
  }}
}}

Focus on actionable, specific improvements based on error patterns and performance gaps.
"""

    def _create_iteration_memory(
        self,
        state: KaggleState,
        failure_analysis: dict[str, Any],
        reward_signals: dict[str, float],
    ) -> IterationMemory:
        """
        Create iteration memory for learning history.

        Args:
            state: Current workflow state
            failure_analysis: Failure analysis
            reward_signals: Reward signals

        Returns:
            IterationMemory object
        """
        current_iteration = state.get("current_iteration", 0)
        current_score = state.get("current_performance_score", 0.0)
        previous_score = state.get("best_score", 0.0)

        return IterationMemory(
            iteration=current_iteration,
            phase="meta_evaluation",
            actions_taken=[
                "analyzed_failures",
                "calculated_rewards",
                "generated_refinement_guidance",
            ],
            results={
                "failure_analysis": failure_analysis,
                "reward_signals": reward_signals,
            },
            score_improvement=current_score - previous_score,
            what_worked=failure_analysis["success_patterns"],
            what_failed=failure_analysis["error_patterns"],
        )

    # ==================== Eureka: Evolutionary Crossover ====================

    def _evolutionary_crossover(self, state: KaggleState) -> dict[str, Any]:
        """
        Eureka-style evolutionary crossover.

        Combines elements from the best-performing iterations to guide
        the next generation of plans.

        Args:
            state: Current workflow state

        Returns:
            Crossover guidance dictionary
        """
        print("\n   🧬 Eureka: Performing evolutionary crossover...")

        iteration_memory = state.get("iteration_memory", [])
        candidate_plans = state.get("candidate_plans", [])

        if not iteration_memory:
            return {}

        # Find top-performing iterations by score improvement
        sorted_memories = sorted(
            iteration_memory,
            key=lambda m: m.score_improvement if hasattr(m, "score_improvement") else 0.0,
            reverse=True,
        )
        top_memories = sorted_memories[:2]  # Top 2 iterations

        # Extract success patterns from top iterations
        success_patterns = set()
        for memory in top_memories:
            what_worked = memory.what_worked if hasattr(memory, "what_worked") else []
            success_patterns.update(what_worked)

        # Extract failure patterns to avoid
        avoid_patterns = set()
        for memory in iteration_memory:
            what_failed = memory.what_failed if hasattr(memory, "what_failed") else []
            avoid_patterns.update(what_failed)

        # Analyze candidate plans if available
        successful_strategies = []
        if candidate_plans:
            # Get strategies from plans with highest fitness
            sorted_candidates = sorted(
                candidate_plans,
                key=lambda p: p.fitness_score if hasattr(p, "fitness_score") else 0.0,
                reverse=True,
            )
            for candidate in sorted_candidates[:2]:
                if hasattr(candidate, "strategy"):
                    successful_strategies.append(candidate.strategy)

        # Generate crossover guidance
        crossover_guidance = {
            "preserve_components": list(success_patterns),
            "avoid_components": list(
                avoid_patterns - success_patterns
            ),  # Don't avoid if also succeeded
            "successful_strategies": successful_strategies,
            "suggested_combinations": self._suggest_combinations(success_patterns),
            "evolutionary_pressure": self._calculate_evolutionary_pressure(iteration_memory),
        }

        print(
            f"   ✓ Crossover: Preserve {len(crossover_guidance['preserve_components'])} patterns, "
            f"Avoid {len(crossover_guidance['avoid_components'])} patterns"
        )

        return crossover_guidance

    def _suggest_combinations(self, success_patterns: set) -> list[str]:
        """
        Suggest promising component combinations based on success patterns.

        Args:
            success_patterns: Set of successful patterns

        Returns:
            List of suggested combinations
        """
        suggestions = []

        pattern_list = list(success_patterns)

        # If we have multiple success patterns, suggest combinations
        if len(pattern_list) >= 2:
            suggestions.append(f"Combine {pattern_list[0]} with {pattern_list[1]}")

        # Standard high-value combinations
        if (
            "model_success" in success_patterns
            and "feature_engineering_success" not in success_patterns
        ):
            suggestions.append("Add advanced feature engineering to successful model")

        if (
            "feature_engineering_success" in success_patterns
            and "ensemble_success" not in success_patterns
        ):
            suggestions.append("Apply ensemble techniques to leverage good features")

        # Suggest based on what's missing
        all_types = {
            "model_success",
            "feature_engineering_success",
            "ensemble_success",
            "preprocessing_success",
        }
        missing = all_types - success_patterns
        if missing:
            missing_type = list(missing)[0].replace("_success", "")
            suggestions.append(f"Explore {missing_type} components for diversity")

        return suggestions[:3]  # Limit to 3 suggestions

    def _calculate_evolutionary_pressure(self, iteration_memory: list) -> str:
        """
        Calculate evolutionary pressure based on iteration history.

        Determines if we should explore (low scores, early iterations)
        or exploit (high scores, late iterations).

        Args:
            iteration_memory: List of iteration memories

        Returns:
            "explore", "exploit", or "balanced"
        """
        if not iteration_memory:
            return "explore"

        # Calculate average score improvement
        improvements = [
            m.score_improvement if hasattr(m, "score_improvement") else 0.0
            for m in iteration_memory
        ]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0

        # Recent trend
        recent_improvements = improvements[-3:] if len(improvements) >= 3 else improvements
        recent_avg = (
            sum(recent_improvements) / len(recent_improvements) if recent_improvements else 0.0
        )

        # If recent improvements are positive and growing, exploit
        if recent_avg > 0.01 and len(iteration_memory) > 2:
            return "exploit"

        # If recent improvements are stagnating or negative, explore
        if recent_avg <= 0 or (len(iteration_memory) > 3 and avg_improvement < 0.005):
            return "explore"

        return "balanced"

    def _collect_training_data(
        self,
        state: KaggleState,
        failure_analysis: dict[str, Any],
        reward_signals: dict[str, float],
    ) -> None:
        """
        Collect training data for DSPy optimization.

        Stores examples with reward signals for later prompt optimization.

        Args:
            state: Current workflow state
            failure_analysis: Failure analysis
            reward_signals: Reward signals
        """
        print("\n   💾 Collecting training data for prompt optimization...")

        ablation_plan = state.get("ablation_plan", [])
        dev_results = state.get("development_results", [])

        if not ablation_plan or not dev_results:
            return

        competition_info = state.get("competition_info")
        domain = state.get("domain_detected", "tabular")
        working_dir = state.get("working_directory", "")

        # Build planner inputs matching AblationPlannerSignature
        comp_info_str = ""
        if competition_info:
            comp_info_str = (
                f"Name: {competition_info.name}\n"
                f"Description: {getattr(competition_info, 'description', '')}\n"
                f"Problem Type: {getattr(competition_info, 'problem_type', '')}\n"
                f"Metric: {getattr(competition_info, 'evaluation_metric', '')}\n"
                f"Domain: {domain}\n"
            )

        sota_solutions = state.get("sota_solutions", [])
        if sota_solutions:
            sota_lines = []
            for sol in sota_solutions[:5]:
                title = getattr(sol, "title", "Unknown")
                score = getattr(sol, "score", 0.0)
                votes = getattr(sol, "votes", 0)
                models = getattr(sol, "models_used", []) or []
                strategies = getattr(sol, "strategies", []) or []
                sota_lines.append(
                    f"- {title} (score={score}, votes={votes}) "
                    f"models={models[:3]} strategies={strategies[:3]}"
                )
            sota_summary = "\n".join(sota_lines)
        else:
            sota_summary = "No SOTA solutions available."

        domain_guidance = get_domain_guidance(str(domain))
        memory_summary = get_memory_summary_for_planning(state)

        # Collect planner example
        plan_quality_score = reward_signals["r_combined"]

        self.training_collector.add_example(
            agent_name="planner",
            inputs={
                "competition_info": comp_info_str,
                "domain": str(domain),
                "sota_summary": sota_summary,
                "domain_guidance": domain_guidance,
                "memory_summary": memory_summary,
            },
            outputs={
                "ablation_plan": json.dumps(
                    [
                        {
                            "name": c.name,
                            "component_type": c.component_type,
                            "description": c.code,
                            "estimated_impact": c.estimated_impact,
                            "rationale": "",
                            "code_outline": "",
                        }
                        for c in ablation_plan
                    ],
                    indent=2,
                ),
                "analysis": "",
            },
            score=plan_quality_score,
        )

        # Collect developer examples (one per component)
        for i, result in enumerate(dev_results):
            if i >= len(ablation_plan):
                continue

            component = ablation_plan[i]
            component_score = 1.0 if result.success else 0.0

            self.training_collector.add_example(
                agent_name="developer_generator",
                inputs={
                    "component_details": format_component_details(component),
                    "competition_context": (
                        f"Name: {competition_info.name if competition_info else 'unknown'}\n"
                        f"Domain: {domain}\n"
                        f"Problem Type: {competition_info.problem_type if competition_info else 'unknown'}\n"
                        f"Metric: {competition_info.evaluation_metric if competition_info else 'unknown'}\n"
                    ),
                    "data_paths": (
                        f"Train: {state.get('current_train_path') or state.get('train_data_path', '')}\n"
                        f"Test: {state.get('current_test_path') or state.get('test_data_path', '')}\n"
                        f"Models: {working_dir}/models\n"
                        f"Sample Submission: {state.get('sample_submission_path', '')}\n"
                    ),
                    "requirements": (
                        f"Implement {component.component_type}: {component.name}\n"
                        f"Time budget: {state.get('component_timeout', state.get('testing_timeout', 'unknown'))}\n"
                        "Return a complete, executable script and write a valid submission CSV."
                    ),
                },
                outputs={
                    "code": result.code,
                    "explanation": "",
                },
                score=component_score,
            )

        print("   ✓ Collected training examples for Planner and Developer")


# ==================== Prompts ====================

META_EVALUATOR_SYSTEM_PROMPT = """# You are a Meta-Evaluator AI

You are an expert meta-evaluator analyzing the performance of AI agents that solve Kaggle competitions.

Your role is to:
1. Analyze component failures and identify root causes
2. Extract actionable patterns from errors and successes
3. Generate strategic guidance for improving agent prompts
4. Provide specific, concrete recommendations

You have access to:
- Component execution results (success/failure)
- Error messages and types
- Performance scores
- Execution times

Your output must be:
- **Actionable**: Specific changes to make
- **Strategic**: Focus on high-impact improvements
- **Evidence-based**: Based on actual error patterns
- **Concrete**: Avoid generic advice

Return structured JSON with clear guidance for each agent type."""


# ==================== LangGraph Node Function ====================


def meta_evaluator_node(state: KaggleState) -> dict[str, Any]:
    """
    LangGraph node function for meta-evaluation.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = MetaEvaluatorAgent()
    return agent(state)
