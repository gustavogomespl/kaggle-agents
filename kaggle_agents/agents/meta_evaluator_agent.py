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
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.state import KaggleState, DevelopmentResult, IterationMemory
from ..core.config import get_config
from ..optimization import create_training_collector


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
        """Initialize meta-evaluator with superior model."""
        self.config = get_config()

        self.llm = ChatOpenAI(
            model="gpt-5.1",  
            temperature=1,  
            max_tokens=30000,
        )

        print("   🧠 Meta-Evaluator initialized with GPT-5")

        # Training data collector for RL
        self.training_collector = create_training_collector()

    def __call__(self, state: KaggleState) -> Dict[str, Any]:
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

    def _analyze_failures(self, state: KaggleState) -> Dict[str, Any]:
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
        elif "filenotfounderror" in error_lower or "no such file" in error_lower:
            return "file_not_found"
        elif "keyerror" in error_lower:
            return "key_error"
        elif "valueerror" in error_lower:
            return "value_error"
        elif "typeerror" in error_lower:
            return "type_error"
        elif "memoryerror" in error_lower or "out of memory" in error_lower:
            return "memory_error"
        elif "timeout" in error_lower or "timed out" in error_lower:
            return "timeout_error"
        elif "syntaxerror" in error_lower:
            return "syntax_error"
        elif "attributeerror" in error_lower:
            return "attribute_error"
        elif "indexerror" in error_lower:
            return "index_error"
        elif "validation failed" in error_lower:
            return "validation_error"
        elif "final validation performance" in error_lower:
            return "missing_output_format"
        else:
            return "runtime_error"

    def _calculate_reward_signals(
        self,
        state: KaggleState,
        failure_analysis: Dict[str, Any],
    ) -> Dict[str, float]:
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
        current_score = state.get("current_performance_score", 0.0)
        best_score = state.get("best_score", 0.0)

        # Reward 1: Functional Correctness (binary)
        total_components = len(dev_results)
        successful_components = len(failure_analysis["success_components"])
        r_functional = successful_components / total_components if total_components > 0 else 0.0

        # Reward 2: Performance (continuous, normalized 0-1)
        target_score = 0.9238  # Top 20%
        r_performance = min(current_score / target_score, 1.0) if target_score > 0 else 0.0

        # Reward 3: Improvement (delta from previous best)
        score_improvement = current_score - best_score
        r_improvement = max(0.0, min(score_improvement * 10, 1.0))  # Scale to 0-1

        # Reward 4: Execution Semantics (no errors, fast execution)
        avg_execution_time = sum(r.execution_time for r in dev_results) / total_components if total_components > 0 else 0.0
        r_semantics = 1.0 - min(avg_execution_time / 300.0, 1.0)  # Normalize by 5min timeout

        # Combined reward (weighted)
        weights = {
            "functional": 0.3,
            "performance": 0.5,
            "improvement": 0.1,
            "semantics": 0.1,
        }

        r_combined = (
            weights["functional"] * r_functional +
            weights["performance"] * r_performance +
            weights["improvement"] * r_improvement +
            weights["semantics"] * r_semantics
        )

        rewards = {
            "r_functional": r_functional,
            "r_performance": r_performance,
            "r_improvement": r_improvement,
            "r_semantics": r_semantics,
            "r_combined": r_combined,
        }

        print(f"   📊 Rewards: functional={r_functional:.3f}, performance={r_performance:.3f}, "
              f"improvement={r_improvement:.3f}, combined={r_combined:.3f}")

        return rewards

    def _generate_refinement_guidance(
        self,
        state: KaggleState,
        failure_analysis: Dict[str, Any],
        reward_signals: Dict[str, float],
    ) -> Dict[str, str]:
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
            guidance = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            guidance = {
                "planner_guidance": "Focus on high-impact components with proven track record.",
                "developer_guidance": "Ensure code follows all requirements and outputs correct format.",
                "priority_fixes": failure_analysis["error_patterns"],
            }

        print(f"   ✓ Generated guidance for Planner and Developer")

        return guidance

    def _build_evaluation_context(
        self,
        state: KaggleState,
        failure_analysis: Dict[str, Any],
        reward_signals: Dict[str, float],
    ) -> str:
        """Build context string for LLM evaluation."""
        current_iteration = state.get("current_iteration", 0)
        current_score = state.get("current_performance_score", 0.0)
        target_score = 0.9238

        context = f"""# Iteration {current_iteration} Evaluation

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

## Failed Components
"""
        for comp in failure_analysis["failed_components"]:
            context += f"\n### {comp['name']} ({comp['type']})\n"
            context += f"Error: {comp['error_type']}\n"
            context += f"Message: {comp['error'][:150]}...\n"

        context += f"\n## Reward Signals\n"
        for key, value in reward_signals.items():
            context += f"- {key}: {value:.3f}\n"

        return context

    def _build_refinement_prompt(self, context: str) -> str:
        """Build prompt for refinement guidance generation."""
        return f"""{context}

## Your Task
Analyze the above results and provide strategic guidance for improving prompts in the next iteration.

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
        failure_analysis: Dict[str, Any],
        reward_signals: Dict[str, float],
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
        failure_analysis: Dict[str, Any],
        reward_signals: Dict[str, float],
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
        competition_info = state.get("competition_info")

        if not ablation_plan or not dev_results:
            return

        # Collect planner example
        plan_quality_score = reward_signals["r_combined"]

        self.training_collector.add_example(
            agent_name="planner",
            inputs={
                "competition_name": competition_info.name if competition_info else "unknown",
                "domain": state.get("domain_detected", "tabular"),
                "problem_type": competition_info.problem_type if competition_info else "unknown",
            },
            outputs={
                "ablation_plan": [
                    {
                        "name": c.name,
                        "type": c.component_type,
                        "code_outline": c.code,
                        "estimated_impact": c.estimated_impact,
                    }
                    for c in ablation_plan
                ]
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
                    "component_type": component.component_type,
                    "component_name": component.name,
                    "code_outline": component.code,  # Use code outline as description
                },
                outputs={
                    "code": result.code,
                },
                score=component_score,
            )

        print(f"   ✓ Collected training examples for Planner and Developer")


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

def meta_evaluator_node(state: KaggleState) -> Dict[str, Any]:
    """
    LangGraph node function for meta-evaluation.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = MetaEvaluatorAgent()
    return agent(state)
