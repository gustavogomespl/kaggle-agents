"""
Meta-Evaluator Agent with Reinforcement Learning.

This agent uses a superior model (GPT-5) to analyze code generation
results and optimize prompts for other agents using RL techniques.

Based on:
- CodeRL+: Execution Semantics Alignment
- PREFACE: Error-guided prompt repair
- RLPrompt: Discrete prompt optimization
- ML-Agent: RL for ML engineering
"""

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.config import calculate_score_improvement, get_config, get_llm_for_role
from ..core.state import IterationMemory, KaggleState
from ..optimization import create_training_collector
from ..prompts.templates.developer_prompts import format_component_details
from ..prompts.templates.planner_prompts import get_domain_guidance
from ..utils.llm_utils import get_text_content


# ==================== Meta-Evaluator Agent ====================

class MetaEvaluatorAgent:
    """
    Meta-agent that evaluates other agents and optimizes their prompts using RL.

    Features:
    - Uses superior model (GPT-5) for better analysis
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
        print("\n" + "="*60)
        print("= META-EVALUATOR: Analyzing Performance & Optimizing Prompts")
        print("="*60)

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
        iteration_memory = self._create_iteration_memory(
            state, failure_analysis, reward_signals
        )

        # Collect training data for DSPy optimization
        self._collect_training_data(state, failure_analysis, reward_signals)

        # Update state
        return {
            "failure_analysis": failure_analysis,
            "reward_signals": reward_signals,
            "refinement_guidance": refinement_guidance,
            "iteration_memory": [iteration_memory],  # Append to list
            "last_updated": datetime.now(),
        }

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

                analysis["failed_components"].append({
                    "name": component_name,
                    "type": component_type,
                    "error": error_msg,
                    "error_type": error_type,
                    "execution_time": result.execution_time,
                })

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
                analysis["success_components"].append({
                    "name": component_name,
                    "type": component_type,
                    "execution_time": result.execution_time,
                })

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

        Args:
            error_msg: Error message

        Returns:
            Error classification
        """
        if not error_msg:
            return "unknown_error"

        error_lower = error_msg.lower()

        # Common error patterns
        if "importerror" in error_lower or "modulenotfounderror" in error_lower:
            return "import_error"
        if "filenotfounderror" in error_lower or "no such file" in error_lower:
            return "file_not_found"
        if "keyerror" in error_lower:
            return "key_error"
        if "valueerror" in error_lower:
            if "shape" in error_lower or "dimension" in error_lower:
                return "dimension_mismatch"
            if "nan" in error_lower or "infinity" in error_lower:
                return "data_contains_nans"
            return "value_error"
        if "typeerror" in error_lower:
            return "type_error"
        if "memoryerror" in error_lower or "out of memory" in error_lower:
            return "memory_error"
        if "timeout" in error_lower or "timed out" in error_lower:
            return "timeout_error"
        if "syntaxerror" in error_lower:
            return "syntax_error"
        if "attributeerror" in error_lower:
            return "attribute_error"
        if "indexerror" in error_lower:
            return "index_error"
        if "validation failed" in error_lower:
            return "validation_error"
        if "final validation performance" in error_lower:
            return "missing_output_format"
        return "runtime_error"

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
        target_score = state.get("target_score", 1)  # Default to 1 if not set
        if isinstance(target_score, str):
            try:
                target_score = float(target_score)
            except ValueError:
                target_score = 1

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

        score_component = min(current_score / float(target_score), 1.0) if float(target_score) > 0 else 0.0
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
        avg_execution_time = sum(r.execution_time for r in dev_results) / total_components if total_components > 0 else 0.0
        r_semantics = 1.0 - min(avg_execution_time / 300.0, 1.0)  # Normalize by 5min timeout

        # Reward 5: Diversity
        # Encourages trying different types of components (e.g. not just 5 XGBoosts)
        unique_types = len({c.get("type", "unknown") for c in failure_analysis["success_components"]})
        r_diversity = min(unique_types / 3.0, 1.0)  # Target: at least 3 different types working

        # Reward 6: Robustness/Overfitting Penalty
        # Penalize if Public LB score is much lower than Validation score
        validation_score = state.get("overall_validation_score", 0.0)
        public_score = 0.0
        if submissions:
            public_score = submissions[-1].public_score or 0.0

        # If we have both scores, check gap. If gap > 0.1, heavy penalty.
        gap = abs(validation_score - public_score)
        r_robustness = 1.0 - min(gap * 5, 1.0) if (validation_score > 0 and public_score > 0) else 1.0

        # Combined reward (weighted)
        # In MLE-bench, speed and medal attainment matter more than exploring many components.
        if run_mode == "mlebench" or "medal" in objective:
            weights = {
                "functional": 0.25,
                "performance": 0.45,
                "improvement": 0.05,
                "semantics": 0.15,
                "diversity": 0.05,
                "robustness": 0.05,
            }
        else:
            weights = {
                "functional": 0.25,
                "performance": 0.40,
                "improvement": 0.1,
                "semantics": 0.05,
                "diversity": 0.1,
                "robustness": 0.1,
            }

        r_combined = (
            weights["functional"] * r_functional +
            weights["performance"] * r_performance +
            weights["improvement"] * r_improvement +
            weights["semantics"] * r_semantics +
            weights["diversity"] * r_diversity +
            weights["robustness"] * r_robustness
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

        print(f"   📊 Rewards: functional={r_functional:.2f}, performance={r_performance:.2f}, "
              f"diversity={r_diversity:.2f}, robustness={r_robustness:.2f}, combined={r_combined:.3f}")

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
        for improving prompts in next iteration.

        Args:
            state: Current workflow state
            failure_analysis: Failure analysis results
            reward_signals: Calculated rewards

        Returns:
            Refinement guidance dictionary
        """
        print("\n   🎯 Generating refinement guidance...")

        # Build context for LLM
        context = self._build_evaluation_context(state, failure_analysis, reward_signals)

        # Generate guidance using superior model (GPT-5)
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

        target_score = state.get("target_score", 1)
        if isinstance(target_score, str):
            try:
                target_score = float(target_score)
            except ValueError:
                target_score = 1

        context = f"""# Iteration {current_iteration} Evaluation

## Objective
- run_mode: {run_mode or 'kaggle'}
- objective: {objective or 'top20'}

## Current Performance
- Score: {current_score:.4f}
- Target: {target_score:.4f}
- Gap: {target_score - current_score:.4f}

## Component Results
- Total: {len(state.get('development_results', []))}
- Successful: {len(failure_analysis['success_components'])}
- Failed: {len(failure_analysis['failed_components'])}

## Success Patterns
{chr(10).join('- ' + p for p in failure_analysis['success_patterns'])}

## Error Patterns
{chr(10).join('- ' + p for p in failure_analysis['error_patterns'])}
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
- valid_submission: {bool(mlebench_grade.get('valid_submission', False))}
- score: {mlebench_grade.get('score')}
- above_median: {bool(mlebench_grade.get('above_median', False))}
- medals: {', '.join(medals) if medals else 'None'}
"""

        # FULL CODE AND PERFORMANCE ANALYSIS
        context += "\n## Component Code and Performance Analysis\n"

        dev_results = state.get("development_results", [])
        import re

        # Limit to 5 most recent components to reduce token usage
        recent_results = dev_results[-5:] if len(dev_results) > 5 else dev_results
        if len(dev_results) > 5:
            context += f"*(Showing 5 most recent components out of {len(dev_results)} total)*\n\n"

        for i, res in enumerate(recent_results):
            # Extract score from stdout if possible
            score_match = re.search(r"Final Validation Performance: (0\.\d+)", res.stdout)
            score = float(score_match.group(1)) if score_match else "N/A"

            # Determine component name (heuristic)
            comp_name = f"Component_{i+1}"
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
            code_lines = res.code.split('\n')
            context += "**Code Summary**:\n```python\n"
            context += '\n'.join(code_lines[:20])  # First 20 lines
            if len(code_lines) > 30:
                context += "\n# ... (middle section omitted) ...\n"
                context += '\n'.join(code_lines[-10:])  # Last 10 lines
            context += "\n```\n"
            context += f"**Total Lines**: {len(code_lines)}\n"
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

        # Collect planner example
        plan_quality_score = reward_signals["r_combined"]

        self.training_collector.add_example(
            agent_name="planner",
            inputs={
                "competition_info": comp_info_str,
                "domain": str(domain),
                "sota_summary": sota_summary,
                "domain_guidance": domain_guidance,
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
