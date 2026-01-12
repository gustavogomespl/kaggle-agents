"""
Planner Agent with Ablation-Driven Optimization.

This agent implements the ablation planning strategy from Google ADK,
identifying high-impact components for systematic improvement.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import Any

import dspy
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.config import get_config, get_llm_for_role
from ..core.state import (
    AblationComponent,
    DevelopmentResult,
    KaggleState,
    SOTASolution,
    get_memory_summary_for_planning,
)
from ..optimization import create_optimizer
from ..prompts.templates.planner_prompts import (
    ANALYZE_GAPS_PROMPT,
    ANALYZE_SOTA_PROMPT,
    CREATE_ABLATION_PLAN_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REFINE_ABLATION_PLAN_PROMPT,
    get_domain_guidance,
)
from ..utils.llm_utils import get_text_content


# ==================== Extended Strategy Templates ====================
# These strategies are used after initial iterations to increase diversity

EXTENDED_STRATEGIES_TABULAR = {
    "feature_engineering_heavy": {
        "name": "feature_engineering_heavy",
        "prompt_modifier": """
Prioritize extensive feature engineering before modeling:
- Create derived features: interactions, ratios, aggregations
- Apply target encoding with proper CV to avoid leakage
- Generate temporal features if applicable (lags, rolling stats)
- Use clustering-based features (KMeans cluster assignments)
- Save engineered features for reuse by all models
""",
        "model_preference": ["lightgbm", "xgboost"],
        "component_emphasis": ["feature_engineering"],
    },
    "neural_exploration": {
        "name": "neural_exploration",
        "prompt_modifier": """
Explore neural network approaches for tabular data:
- TabNet for interpretable neural networks
- MLP with embeddings for categorical features
- Neural network + gradient boosting ensemble
- Use proper regularization (dropout, weight decay)
""",
        "model_preference": ["tabnet", "mlp", "neural_ensemble"],
        "component_emphasis": ["model"],
    },
    "hyperparameter_variant": {
        "name": "hyperparameter_variant",
        "prompt_modifier": """
Explore hyperparameter variations of successful models:
- If LightGBM worked, try different learning_rate (0.01, 0.05, 0.1)
- Vary max_depth (4, 6, 8, 10) and num_leaves
- Test different regularization (lambda_l1, lambda_l2)
- Try different n_estimators (500, 1000, 2000)
""",
        "model_preference": ["lightgbm", "xgboost", "catboost"],
        "component_emphasis": ["model"],
        "inherit_from_best": True,
    },
    "stacking_ensemble": {
        "name": "stacking_ensemble",
        "prompt_modifier": """
Focus on advanced stacking ensembles:
- Create diverse base models (GBM, RF, Linear)
- Use OOF predictions as meta-features
- Add second-level meta-learner (Ridge, LightGBM)
- Ensure proper CV alignment to avoid leakage
""",
        "model_preference": ["stacking", "blending"],
        "component_emphasis": ["ensemble"],
    },
}

EXTENDED_STRATEGIES_CV = {
    "feature_engineering_heavy": {
        "name": "feature_engineering_heavy",
        "prompt_modifier": """
Focus on advanced image augmentation and preprocessing:
- Heavy augmentation: Cutmix, Mixup, GridMask, RandomErasing
- Test Time Augmentation (TTA) with multiple crops
- External data integration if allowed
- Multi-scale feature extraction
""",
        "model_preference": ["efficientnet_b3", "resnet50"],
        "component_emphasis": ["preprocessing", "augmentation"],
    },
    "neural_exploration": {
        "name": "neural_exploration",
        "prompt_modifier": """
Explore SOTA vision architectures:
- Vision Transformers (ViT, DeiT, Swin)
- ConvNeXt for modern CNN approach
- Hybrid CNN-Transformer models
- Knowledge distillation from larger models
""",
        "model_preference": ["vit", "swin", "convnext", "deit"],
        "component_emphasis": ["model"],
    },
    "hyperparameter_variant": {
        "name": "hyperparameter_variant",
        "prompt_modifier": """
Explore training variations:
- Different learning rate schedules (cosine, warmup)
- Vary image sizes (224, 384, 512)
- Different optimizers (AdamW, SAM, LAMB)
- Label smoothing and loss variants
""",
        "model_preference": ["efficientnet", "resnet"],
        "component_emphasis": ["model"],
        "inherit_from_best": True,
    },
}

EXTENDED_STRATEGIES_NLP = {
    "feature_engineering_heavy": {
        "name": "feature_engineering_heavy",
        "prompt_modifier": """
Focus on text preprocessing and feature extraction:
- Advanced tokenization strategies
- Domain-specific vocabulary expansion
- Text augmentation (back-translation, synonym replacement)
- Sentence-level and document-level features
""",
        "model_preference": ["roberta_base", "deberta"],
        "component_emphasis": ["preprocessing"],
    },
    "neural_exploration": {
        "name": "neural_exploration",
        "prompt_modifier": """
Explore advanced NLP architectures:
- DeBERTa-v3 (large and xlarge variants)
- Longformer for long documents
- Multi-task learning approaches
- Ensemble of different model sizes
""",
        "model_preference": ["deberta_v3_large", "longformer", "roberta_large"],
        "component_emphasis": ["model"],
    },
    "hyperparameter_variant": {
        "name": "hyperparameter_variant",
        "prompt_modifier": """
Explore training variations:
- Different learning rates (1e-5, 2e-5, 3e-5)
- Layer-wise learning rate decay
- Different pooling strategies (CLS, mean, max)
- Gradient accumulation for larger batch sizes
""",
        "model_preference": ["roberta", "deberta"],
        "component_emphasis": ["model"],
        "inherit_from_best": True,
    },
}


# ==================== DSPy Signatures ====================


class AblationPlannerSignature(dspy.Signature):
    """Signature for ablation plan generation."""

    competition_info: str = dspy.InputField(desc="Competition metadata and description")
    domain: str = dspy.InputField(desc="Competition domain (tabular, CV, NLP, etc.)")
    sota_details: str = dspy.InputField(
        desc="Detailed SOTA solutions with code snippets, votes, and complexity"
    )
    sota_summary: str = dspy.InputField(
        desc="Summary of SOTA patterns (models, features, ensembles)"
    )
    domain_guidance: str = dspy.InputField(desc="Domain-specific guidance and priorities")
    memory_summary: str = dspy.InputField(
        desc="Memory summary of past results, errors, and best hyperparameters"
    )

    ablation_plan: str = dspy.OutputField(
        desc="JSON list of ablation components using Adopt & Improve strategy"
    )
    analysis: str = dspy.OutputField(desc="Analysis of which SOTA solution was adopted and why")


class SOTAAnalysisSignature(dspy.Signature):
    """Signature for SOTA solution analysis."""

    sota_solutions: str = dspy.InputField(desc="List of SOTA solutions with strategies")

    common_models: str = dspy.OutputField(desc="Most frequently used models")
    feature_patterns: str = dspy.OutputField(desc="Common feature engineering techniques")
    ensemble_strategies: str = dspy.OutputField(desc="Popular ensemble methods")
    unique_tricks: str = dspy.OutputField(desc="Novel or unique approaches")
    success_factors: str = dspy.OutputField(desc="Key factors separating top solutions")


# ==================== DSPy Modules ====================


class AblationPlannerModule(dspy.Module):
    """DSPy module for ablation planning."""

    def __init__(self):
        super().__init__()
        self.generate_plan = dspy.ChainOfThought(AblationPlannerSignature)

    def forward(
        self, competition_info, domain, sota_details, sota_summary, domain_guidance, memory_summary
    ):
        """Generate ablation plan using Adopt & Improve strategy."""
        return self.generate_plan(
            competition_info=competition_info,
            domain=domain,
            sota_details=sota_details,
            sota_summary=sota_summary,
            domain_guidance=domain_guidance,
            memory_summary=memory_summary,
        )


class SOTAAnalyzerModule(dspy.Module):
    """DSPy module for SOTA analysis."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(SOTAAnalysisSignature)

    def forward(self, sota_solutions):
        """Analyze SOTA solutions."""
        return self.analyze(sota_solutions=sota_solutions)


# ==================== Planner Agent ====================


class PlannerAgent:
    """
    Agent responsible for creating ablation plans.

    Strategy (Google ADK Ablation-Driven Optimization):
    1. Analyze SOTA solutions to identify patterns
    2. Identify high-impact components (features, models, preprocessing, ensemble)
    3. Estimate impact of each component
    4. Create prioritized plan focusing on high-ROI components
    5. Use DSPy for prompt optimization over time
    """

    def __init__(self, use_dspy: bool = True):
        """
        Initialize the planner agent.

        Args:
            use_dspy: Whether to use DSPy modules (vs direct LLM calls)
        """
        self.config = get_config()
        self.use_dspy = use_dspy and self.config.dspy.enabled

        # ALWAYS initialize self.llm (used by _analyze_gaps, _analyze_sota_solutions, etc.)
        self.llm = get_llm_for_role(
            role="planner",
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )

        if self.use_dspy:
            # Try to load optimized module, fallback to direct LLM path
            optimizer = create_optimizer()
            self.planner_module = optimizer.load_optimized_prompt("planner")

            if self.planner_module is None:
                print("   No optimized planner module found -> using direct LLM path")
                self.use_dspy = False
            else:
                self.sota_analyzer = SOTAAnalyzerModule()

    def __call__(self, state: KaggleState) -> dict[str, Any]:
        """
        Execute the planner agent.

        Args:
            state: Current workflow state

        Returns:
            State updates with ablation plan
        """
        # Check if this is a refinement iteration
        current_iteration = state.get("current_iteration", 0)
        is_refinement = current_iteration > 1

        # Check if Eureka mode should be used
        # Eureka is activated when:
        # 1. crossover_guidance exists (from meta-evaluator evolutionary crossover)
        # 2. OR evolutionary_generation > 0 (continuing evolutionary search)
        # 3. OR is_refinement and we want to explore multiple strategies
        crossover_guidance = state.get("crossover_guidance", {})
        evolutionary_generation = state.get("evolutionary_generation", 0)
        use_eureka = bool(crossover_guidance) or evolutionary_generation > 0 or is_refinement

        # Check for debug loop trigger from Meta-Evaluator (INNER LOOP REFINEMENT)
        debug_result = self._handle_debug_loop_trigger(state)
        if debug_result and debug_result.get("skip_normal_planning"):
            print("\n" + "=" * 60)
            print("= PLANNER AGENT: DEBUG LOOP (Inner Loop Refinement)")
            print("=" * 60)
            return {
                "ablation_plan": debug_result["ablation_plan"],
                "is_debug_iteration": True,
                "debug_target": debug_result.get("debug_target"),
                "debug_hints": debug_result.get("debug_hints", []),
                "last_updated": datetime.now(),
            }

        # Detect multi-modal competition (images + rich tabular features)
        multimodal_info = self._detect_multimodal_competition(state)
        if multimodal_info.get("is_multimodal"):
            print(f"\n  📋 Multi-modal Strategy: {multimodal_info.get('recommendation', '')[:100]}...")

        if use_eureka:
            print("\n" + "=" * 60)
            print("= PLANNER AGENT: Eureka Multi-Candidate Planning")
            print("=" * 60)
        elif is_refinement:
            print("\n" + "=" * 60)
            print("= PLANNER AGENT: Refining Ablation Plan (RL-based)")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("= PLANNER AGENT: Creating Ablation Plan")
            print("=" * 60)

        # 1. Analyze SOTA solutions
        print("\nAnalyzing SOTA patterns...")
        sota_analysis = self._analyze_sota_solutions(state)

        # 2. Generate ablation plan
        if use_eureka:
            # Eureka: Generate multiple candidate plans and select best
            print("\n🧬 Using Eureka multi-candidate evolutionary planning...")
            eureka_result = self.generate_with_eureka(state, sota_analysis, n_candidates=3)

            # Validate the selected plan
            validated_plan = self._validate_plan(eureka_result["ablation_plan"], state=state)

            # Print summary
            self._print_summary(validated_plan)

            # Return Eureka-specific state updates
            return {
                "ablation_plan": validated_plan,
                "candidate_plans": eureka_result["candidate_plans"],
                "current_plan_index": eureka_result["current_plan_index"],
                "evolutionary_generation": eureka_result["evolutionary_generation"],
                "optimization_strategy": eureka_result["optimization_strategy"],
                "last_updated": datetime.now(),
            }

        if is_refinement:
            print("\n🔄 Refining plan based on previous results...")
            ablation_plan = self._refine_ablation_plan(state, sota_analysis)
        else:
            print("\n📝 Generating ablation plan...")
            ablation_plan = self._generate_ablation_plan(state, sota_analysis)

        # 3. Validate and enhance plan
        validated_plan = self._validate_plan(ablation_plan, state=state)

        # 4. Print summary
        self._print_summary(validated_plan)

        # Return state updates
        return {
            "ablation_plan": validated_plan,
            "optimization_strategy": "ablation_driven",
            "last_updated": datetime.now(),
        }

    def _analyze_sota_solutions(self, state: KaggleState) -> dict[str, Any]:
        """
        Analyze SOTA solutions to extract patterns.

        Args:
            state: Current state with SOTA solutions

        Returns:
            Dictionary with analysis results
        """
        sota_solutions = state.get("sota_solutions", [])

        if not sota_solutions:
            return {
                "common_models": [],
                "feature_patterns": [],
                "ensemble_strategies": [],
                "unique_tricks": [],
                "success_factors": [],
            }

        # Format SOTA solutions for analysis
        sota_summary = self._format_sota_solutions(sota_solutions)

        if self.use_dspy and hasattr(self, "sota_analyzer"):
            # Use DSPy module
            result = self.sota_analyzer(sota_solutions=sota_summary)

            analysis = {
                "common_models": result.common_models.split(", ") if result.common_models else [],
                "feature_patterns": result.feature_patterns.split(", ")
                if result.feature_patterns
                else [],
                "ensemble_strategies": result.ensemble_strategies
                if result.ensemble_strategies
                else "",
                "unique_tricks": result.unique_tricks.split(", ") if result.unique_tricks else [],
                "success_factors": result.success_factors.split(", ")
                if result.success_factors
                else [],
            }
        else:
            # Use direct LLM call
            prompt = ANALYZE_SOTA_PROMPT.format(sota_solutions=sota_summary)
            messages = [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)

            # Parse JSON from response
            try:
                content = get_text_content(response.content)
                # Strip optional markdown fences
                if isinstance(content, str):
                    content = content.strip()
                    if "```json" in content:
                        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
                    elif content.startswith("```") and content.endswith("```"):
                        content = content.strip("` \n")
                analysis = json.loads(content)
            except Exception:
                # Fallback to empty analysis
                analysis = {
                    "common_models": [],
                    "feature_patterns": [],
                    "ensemble_strategies": "",
                    "unique_tricks": [],
                    "success_factors": [],
                }

        print(f"   Found {len(analysis.get('common_models', []))} common models")
        print(f"   Found {len(analysis.get('feature_patterns', []))} feature patterns")

        return analysis

    def _extract_curriculum_insights(self, state: KaggleState) -> str:
        """
        Extract curriculum learning insights from iteration memory.

        Builds on successful patterns and avoids known failures from previous iterations.

        Args:
            state: Current workflow state containing iteration_memory

        Returns:
            Formatted string with learned patterns to inject in prompt
        """
        iteration_memory = state.get("iteration_memory", [])

        if not iteration_memory:
            return "No previous iteration insights available."

        # Aggregate patterns across all iterations
        all_worked = []
        all_failed = []

        for memory in iteration_memory:
            all_worked.extend(memory.what_worked)
            all_failed.extend(memory.what_failed)

        # Build insights string
        insights = ["\n🧠 CURRICULUM LEARNING INSIGHTS (from previous iterations):"]

        if all_worked:
            insights.append("\n✅ What Worked (prioritize these approaches):")
            # Get last 5 unique successful patterns
            unique_worked = list(dict.fromkeys(all_worked))[-5:]
            for pattern in unique_worked:
                insights.append(f"   - {pattern}")

        if all_failed:
            insights.append("\n❌ CRITICAL: What Failed (DO NOT REPEAT these approaches):")
            # Get last 5 unique failure patterns
            unique_failed = list(dict.fromkeys(all_failed))[-5:]
            for pattern in unique_failed:
                insights.append(f"   - {pattern}")

        # Add failure analysis insights from latest iteration
        if iteration_memory:
            latest_memory = iteration_memory[-1]
            if "failure_analysis" in latest_memory.results:
                analysis = latest_memory.results["failure_analysis"]

                if analysis.get("common_errors"):
                    insights.append("\n⚠️  Common Errors to Avoid:")
                    # Get top 3 most common errors
                    common_errors = sorted(
                        analysis["common_errors"].items(), key=lambda x: x[1], reverse=True
                    )[:3]
                    for error_type, count in common_errors:
                        insights.append(f"   - {error_type}: {count} occurrences")

                # Add component-specific success patterns
                if analysis.get("success_by_component"):
                    insights.append("\n📊 Component Success Rates:")
                    for comp_type, success_info in list(analysis["success_by_component"].items())[
                        :3
                    ]:
                        rate = success_info.get("success_rate", 0.0)
                        insights.append(f"   - {comp_type}: {rate:.0%} success rate")

        # Add score improvement trend
        if len(iteration_memory) >= 2:
            score_improvements = [m.score_improvement for m in iteration_memory[-3:]]
            avg_improvement = sum(score_improvements) / len(score_improvements)
            insights.append(f"\n📈 Recent Score Trend: {avg_improvement:+.4f} avg improvement")

        return "\n".join(insights)

    def _generate_ablation_plan(
        self,
        state: KaggleState,
        sota_analysis: dict[str, Any],
    ) -> list[AblationComponent]:
        """
        Generate ablation plan based on competition info and SOTA analysis.

        Uses curriculum learning to incorporate insights from previous iterations.

        Args:
            state: Current state
            sota_analysis: Analysis of SOTA solutions

        Returns:
            List of ablation components
        """
        competition_info = state["competition_info"]
        domain = state.get("domain_detected", "tabular")

        # Extract curriculum learning insights from previous iterations
        curriculum_insights = self._extract_curriculum_insights(state)

        # Get raw SOTA solutions for "Adopt & Improve" strategy
        sota_solutions = state.get("sota_solutions", [])
        sota_details = self._format_sota_details(sota_solutions)
        memory_summary = get_memory_summary_for_planning(state)

        # Prepare inputs
        comp_info_str = f"""
Name: {competition_info.name}
Description: {competition_info.description}
Problem Type: {competition_info.problem_type}
Metric: {competition_info.evaluation_metric}
Domain: {domain}
"""

        sota_summary = json.dumps(sota_analysis, indent=2)
        domain_guidance = get_domain_guidance(domain)

        # Resolve max components (allow override via env)
        run_mode = state.get("run_mode", "")
        objective = state.get("objective", "") or ""
        fast_mode = bool(state.get("fast_mode")) or run_mode == "mlebench" or "medal" in objective
        max_components = 3 if fast_mode else 6
        override = os.getenv("KAGGLE_AGENTS_MAX_COMPONENTS")
        if override:
            try:
                override_val = int(override)
                if override_val >= 2:
                    max_components = override_val
            except ValueError:
                print(f"  ⚠️ Invalid KAGGLE_AGENTS_MAX_COMPONENTS='{override}', using default")

        # Print curriculum insights
        if "No previous iteration" not in curriculum_insights:
            print(curriculum_insights)

        if self.use_dspy:
            # DSPy path: Use AblationPlannerModule with Adopt & Improve strategy
            print("  🧠 Using DSPy for ablation plan generation (Adopt & Improve)...")
            try:
                try:
                    result = self.planner_module(
                        competition_info=comp_info_str,
                        domain=domain,
                        sota_details=sota_details,
                        sota_summary=sota_summary,
                        domain_guidance=domain_guidance,
                        memory_summary=memory_summary,
                    )
                except TypeError:
                    # Backward-compatible fallback for older optimized signatures
                    result = self.planner_module(
                        competition_info=comp_info_str,
                        domain=domain,
                        sota_details=sota_details,
                        sota_summary=sota_summary,
                        domain_guidance=domain_guidance,
                    )
                plan_data = self._parse_llm_plan_response(result.ablation_plan, sota_analysis)
                if len(plan_data) < 3:
                    print(f"  ⚠️ DSPy generated only {len(plan_data)} components, using fallback")
                    plan_data = self._create_fallback_plan(
                        domain, sota_analysis, curriculum_insights, state=state
                    )
            except Exception as e:
                print(f"  ⚠️ DSPy plan generation failed: {e}, using fallback")
                plan_data = self._create_fallback_plan(
                    domain, sota_analysis, curriculum_insights, state=state
                )

        else:
            # Use direct LLM call with Adopt & Improve strategy
            prompt = CREATE_ABLATION_PLAN_PROMPT.format(
                competition_info=comp_info_str,
                domain=domain,
                sota_details=sota_details,
                sota_summary=sota_summary,
                memory_summary=memory_summary,
            )

            # Inject curriculum learning insights
            enhanced_prompt = f"""{prompt}

{curriculum_insights}

IMPORTANT: Use the curriculum insights above to:
1. Prioritize components and techniques that worked in previous iterations
2. Avoid approaches that consistently failed or caused errors
3. Learn from common error patterns and component success rates
4. Build upon successful patterns while exploring new improvements

Generate a plan that leverages proven successful strategies and avoids known pitfalls.

Return a JSON array with up to {max_components} components. Each component must have:
- name: unique identifier
- component_type: one of [feature_engineering, model, preprocessing, ensemble]
- description: what this component does
- estimated_impact: float 0.0-1.0
- code_outline: brief implementation sketch"""

            # INJECT FAILURE ANALYSIS (if available from previous runs)
            failure_analysis = state.get("failure_analysis", {})
            if failure_analysis:
                error_patterns = failure_analysis.get("error_patterns", [])
                if error_patterns:
                    failure_text = "\n\n## ⚠️ ERROR PATTERNS FROM PREVIOUS RUNS\n"
                    failure_text += "Avoid these known issues when planning components:\n"
                    for pattern in error_patterns[:5]:
                        failure_text += f"- {pattern}\n"
                    enhanced_prompt += failure_text
                    print(f"  ⚠️ Added {len(error_patterns[:5])} error patterns to initial plan")

            messages = [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT + "\n\n" + domain_guidance),
                HumanMessage(content=enhanced_prompt),
            ]

            # Invoke LLM and parse response
            print("  🧠 Using LLM for ablation plan generation (Adopt & Improve strategy)...")
            try:
                response = self.llm.invoke(messages)
                # Normalize response content to text (handles list/dict responses from some LLM backends)
                try:
                    from ..utils.llm_utils import get_text_content

                    plan_text = get_text_content(response.content)
                except Exception:
                    plan_text = response.content if hasattr(response, "content") else response
                if not isinstance(plan_text, str):
                    # Coerce lists/dicts to string for parsing
                    try:
                        plan_text = "\n".join(map(str, plan_text))  # type: ignore[arg-type]
                    except Exception:
                        plan_text = str(plan_text)
                plan_data = self._parse_llm_plan_response(plan_text, sota_analysis)
                if len(plan_data) < 3:
                    print(f"  ⚠️ LLM generated only {len(plan_data)} components, using fallback")
                    plan_data = self._create_fallback_plan(
                        domain, sota_analysis, curriculum_insights, state=state
                    )
            except Exception as e:
                print(f"  ⚠️ LLM plan generation failed: {e}, using fallback")
                plan_data = self._create_fallback_plan(
                    domain, sota_analysis, curriculum_insights, state=state
                )

        components = self._coerce_components(plan_data)

        # Sort by estimated impact (descending)
        components.sort(key=lambda x: x.estimated_impact, reverse=True)

        return components

    def _coerce_components(
        self,
        plan_data: list[Any],
    ) -> list[AblationComponent]:
        """Normalize plan data into AblationComponent objects."""
        components: list[AblationComponent] = []
        for i, item in enumerate(plan_data):
            if isinstance(item, AblationComponent):
                components.append(item)
                continue
            if not isinstance(item, dict):
                continue

            name = item.get("name")
            if not name or name == "unnamed":
                comp_type = item.get("component_type", "component")
                name = f"{comp_type}_{i + 1}"
                if item.get("description"):
                    desc = str(item["description"]).lower()
                    for word in desc.split():
                        if len(word) > 4 and word not in ["using", "apply", "create", "implement"]:
                            name = f"{word}_{comp_type}"
                            break

            try:
                estimated_impact = float(item.get("estimated_impact", 0.05))
            except (TypeError, ValueError):
                estimated_impact = 0.05

            component = AblationComponent(
                name=name,
                component_type=item.get("component_type", "preprocessing"),
                code=item.get("code_outline", item.get("description", "")),
                estimated_impact=estimated_impact,
                tested=False,
                actual_impact=None,
            )
            components.append(component)

        return components

    def _refine_ablation_plan(
        self, state: KaggleState, sota_analysis: dict[str, Any]
    ) -> list[AblationComponent]:
        """
        Refine the ablation plan based on previous results using RL prompts.

        Args:
            state: Current state with previous results
            sota_analysis: SOTA analysis results

        Returns:
            Refined ablation plan
        """

        # Gather previous results
        previous_plan = state.get("ablation_plan", [])
        dev_results = state.get("development_results", [])
        best_score = state.get("best_score", 0.0)
        current_score = state.get("current_performance_score", best_score)

        # Build test results summary
        test_results_summary = []
        for i, component in enumerate(previous_plan):
            if i < len(dev_results):
                result = dev_results[i]
                test_results_summary.append(
                    {
                        "component": component.name,
                        "type": component.component_type,
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "impact": "positive" if result.success else "failed",
                    }
                )

        # Format previous plan for prompt
        previous_plan_str = json.dumps(
            [
                {
                    "name": c.name,
                    "type": c.component_type,
                    "description": c.code[:100] + "...",
                    "estimated_impact": c.estimated_impact,
                }
                for c in previous_plan
            ],
            indent=2,
        )

        # Format test results
        test_results_str = json.dumps(test_results_summary, indent=2)

        # Perform Gap Analysis (NEW STEP)
        print("  🔍 Performing Gap Analysis...")
        gap_analysis = self._analyze_gaps(
            state=state, previous_plan_str=previous_plan_str, test_results_str=test_results_str
        )
        gap_analysis_str = json.dumps(gap_analysis, indent=2)
        print(f"  🔍 Gap Analysis: {gap_analysis.get('improvement_strategy', 'No strategy found')}")

        # Use the refinement prompt
        prompt = REFINE_ABLATION_PLAN_PROMPT.format(
            gap_analysis=gap_analysis_str,
            previous_plan=previous_plan_str,
            test_results=test_results_str,
            current_score=current_score,
            memory_summary=get_memory_summary_for_planning(state),
        )

        # INJECT FAILURE ANALYSIS DIRECTLY (for explicit error prevention)
        failure_analysis = state.get("failure_analysis", {})
        if failure_analysis:
            error_patterns = failure_analysis.get("error_patterns", [])
            failed_components = failure_analysis.get("failed_components", [])

            if error_patterns or failed_components:
                failure_text = (
                    "\n\n## ⚠️ ERROR PATTERNS TO PREVENT (from MetaEvaluator analysis)\n\n"
                )
                failure_text += (
                    "These errors occurred in previous components - plan to AVOID them:\n\n"
                )

                if error_patterns:
                    failure_text += "**Recurring Error Types:**\n"
                    for i, pattern in enumerate(error_patterns[:5], 1):
                        failure_text += f"{i}. {pattern}\n"
                    failure_text += "\n"

                if failed_components:
                    failure_text += "**Failed Components (DO NOT repeat similar approaches):**\n"
                    for comp in failed_components[:5]:
                        comp_name = comp.get("name", "Unknown")
                        comp_type = comp.get("type", "unknown")
                        error = comp.get("error", "Unknown error")[:100]
                        failure_text += f"- {comp_name} ({comp_type}): {error}\n"
                    failure_text += "\n"

                failure_text += "**CRITICAL**: When generating new components, ensure they handle these cases properly.\n"

                prompt += failure_text
                print(f"  ⚠️ Injected {len(error_patterns)} error patterns into prompt")

        # INJECT META-EVALUATOR GUIDANCE (RL Pattern)
        refinement_guidance = state.get("refinement_guidance", {})
        if refinement_guidance:
            guidance_text = "\n\n## 🎯 META-EVALUATOR GUIDANCE (from RL analysis)\n\n"

            if "planner_guidance" in refinement_guidance:
                guidance_text += (
                    f"**Strategic Guidance:**\n{refinement_guidance['planner_guidance']}\n\n"
                )

            if refinement_guidance.get("priority_fixes"):
                guidance_text += "**Priority Error Fixes:**\n"
                for error in refinement_guidance["priority_fixes"]:
                    guidance_text += f"- Avoid components that cause: {error}\n"
                guidance_text += "\n"

            if refinement_guidance.get("success_amplification"):
                guidance_text += "**Amplify These Successes:**\n"
                for success in refinement_guidance["success_amplification"]:
                    guidance_text += f"- {success}\n"
                guidance_text += "\n"

            if "component_type_guidance" in refinement_guidance:
                guidance_text += "**Component-Specific Guidance:**\n"
                for comp_type, guide in refinement_guidance["component_type_guidance"].items():
                    guidance_text += f"- {comp_type}: {guide}\n"

            prompt += guidance_text
            print("  🧠 Injected Meta-Evaluator guidance into prompt")

        try:
            if self.use_dspy:
                # For now, use fallback in refinement mode too
                # TODO: Create DSPy refinement module
                print("  🔧 Using enhanced fallback with refinement logic")
                plan_data = self._create_refined_fallback_plan(
                    state,
                    sota_analysis,
                    test_results_summary,
                    previous_plan,
                    dev_results,
                )
            else:
                # Use LLM with refinement prompt
                from langchain.schema import HumanMessage, SystemMessage

                messages = [
                    SystemMessage(
                        content="You are a Kaggle Grandmaster expert at refining ML solutions based on test results."
                    ),
                    HumanMessage(content=prompt),
                ]

                response = self.llm.invoke(messages)
                plan_text = get_text_content(response.content).strip()

                # Parse JSON
                # Remove markdown code blocks if present
                if "```json" in plan_text:
                    plan_text = plan_text.split("```json")[1].split("```")[0].strip()
                elif "```" in plan_text:
                    plan_text = plan_text.split("```")[1].split("```")[0].strip()

                plan_data = json.loads(plan_text)

        except Exception as e:
            print(f"  ⚠️  Refinement failed: {e!s}")
            print("  🔧 Using enhanced fallback with refinement logic")
            plan_data = self._create_refined_fallback_plan(
                state,
                sota_analysis,
                test_results_summary,
                previous_plan,
                dev_results,
            )

        # Convert to AblationComponent objects
        components = []
        for i, item in enumerate(plan_data):
            # Get code from code_outline or description (fallback for compatibility)
            code = item.get("code_outline", item.get("description", ""))

            component = AblationComponent(
                name=item.get("name", f"refined_component_{i + 1}"),
                component_type=item.get("component_type", "model"),
                code=code,
                estimated_impact=item.get("estimated_impact", 0.15),
            )
            components.append(component)

        # Sort by estimated impact
        components.sort(key=lambda x: x.estimated_impact, reverse=True)

        return components

    def _create_refined_fallback_plan(
        self,
        state: KaggleState,
        sota_analysis: dict[str, Any],
        test_results: list[dict],
        previous_plan: list[AblationComponent],
        dev_results: list[DevelopmentResult],
    ) -> list[dict[str, Any]]:
        """
        Create a refined fallback plan based on what worked in previous iteration.

        Args:
            state: Current state
            sota_analysis: SOTA analysis
            test_results: Previous test results

        Returns:
            Refined plan as list of dicts
        """
        # Bandit-lite: keep top-2 successful arms by reward, explore one new arm
        arms = []
        for idx, comp in enumerate(previous_plan):
            score = None
            if idx < len(dev_results):
                score = self._extract_validation_score(dev_results[idx].stdout)
            reward = score if score is not None else comp.estimated_impact
            success = idx < len(dev_results) and dev_results[idx].success
            arms.append(
                {
                    "component": comp,
                    "reward": reward if reward is not None else 0.0,
                    "success": success,
                }
            )

        # Exploit: keep top-2 successful arms
        successful_arms = [a for a in arms if a["success"]]
        successful_arms.sort(key=lambda a: a["reward"], reverse=True)
        keep = successful_arms[:2]

        plan = []

        # Ensure a strong feature engineering arm is present
        fe_in_keep = any(a["component"].component_type == "feature_engineering" for a in keep)
        if not fe_in_keep:
            plan.append(
                {
                    "name": "advanced_feature_engineering",
                    "component_type": "feature_engineering",
                    "description": "Polynomial + interaction features with leak-safe pipelines (imputer/encoder in CV)",
                    "estimated_impact": 0.15,
                    "rationale": "Consistently strong FE baseline",
                    "code_outline": "Pipeline with ColumnTransformer, SimpleImputer, OneHot/TargetEncoder, interactions",
                }
            )

        # Add kept winners
        for arm in keep:
            comp = arm["component"]
            plan.append(
                {
                    "name": comp.name,
                    "component_type": comp.component_type,
                    "description": comp.code or comp.component_type,
                    "estimated_impact": float(comp.estimated_impact)
                    if comp.estimated_impact
                    else max(0.12, arm["reward"]),
                    "rationale": "Kept from previous iteration (top reward)",
                    "code_outline": comp.code or comp.component_type,
                }
            )

        # Ensure at least two model components - DOMAIN AWARE
        domain = str(state.get("domain_detected", "tabular")).lower()

        # Define domain groups for fallback model selection
        NLP_DOMAINS = {"text_classification", "text_regression", "seq_to_seq", "nlp"}
        IMAGE_DOMAINS = {"image_classification", "image_regression", "image_segmentation",
                         "object_detection", "computer_vision", "image"}
        AUDIO_DOMAINS = {"audio_classification", "audio_regression"}

        model_count = sum(1 for p in plan if p["component_type"] == "model")

        if domain in NLP_DOMAINS:
            # NLP-specific fallback models
            if model_count < 2:
                plan.append(
                    {
                        "name": "tfidf_logreg_baseline",
                        "component_type": "model",
                        "description": "TF-IDF (1-3 ngrams) + LogisticRegression, 5-fold StratifiedKFold",
                        "estimated_impact": 0.22,
                        "rationale": "Strong NLP baseline - fast and interpretable",
                        "code_outline": "TfidfVectorizer(ngram_range=(1,3), max_features=50000) + LogisticRegression(C=1.0, solver='saga')",
                    }
                )
                model_count += 1

            if model_count < 2:
                plan.append(
                    {
                        "name": "tfidf_svm_classifier",
                        "component_type": "model",
                        "description": "TF-IDF + LinearSVC with calibration for probability outputs",
                        "estimated_impact": 0.18,
                        "rationale": "Adds diversity for text ensemble",
                        "code_outline": "TfidfVectorizer + CalibratedClassifierCV(LinearSVC(), cv=3)",
                    }
                )

        elif domain in IMAGE_DOMAINS:
            # Image-specific fallback models
            if model_count < 2:
                plan.append(
                    {
                        "name": "efficientnet_b0_baseline",
                        "component_type": "model",
                        "description": "EfficientNet-B0 pretrained, fine-tune all layers, 5-fold CV",
                        "estimated_impact": 0.22,
                        "rationale": "Strong pretrained CNN baseline",
                        "code_outline": "timm.create_model('efficientnet_b0', pretrained=True, num_classes=N)",
                    }
                )
                model_count += 1

            if model_count < 2:
                plan.append(
                    {
                        "name": "resnet34_classifier",
                        "component_type": "model",
                        "description": "ResNet34 pretrained with custom head",
                        "estimated_impact": 0.18,
                        "rationale": "Adds diversity for image ensemble",
                        "code_outline": "timm.create_model('resnet34', pretrained=True, num_classes=N)",
                    }
                )

        elif domain in AUDIO_DOMAINS:
            # Audio-specific fallback models
            if model_count < 2:
                plan.append(
                    {
                        "name": "melspec_cnn_baseline",
                        "component_type": "model",
                        "description": "Mel-spectrogram + EfficientNet-B0 for audio classification",
                        "estimated_impact": 0.22,
                        "rationale": "Standard audio classification approach",
                        "code_outline": "librosa.feature.melspectrogram + timm.create_model('efficientnet_b0')",
                    }
                )
                model_count += 1

            if model_count < 2:
                plan.append(
                    {
                        "name": "audio_feature_lgbm",
                        "component_type": "model",
                        "description": "Handcrafted audio features (MFCC, spectral) + LightGBM",
                        "estimated_impact": 0.16,
                        "rationale": "Fast baseline with interpretable features",
                        "code_outline": "librosa MFCC/chroma/spectral + LGBMClassifier",
                    }
                )

        else:
            # Default tabular fallback models
            if model_count < 2:
                plan.append(
                    {
                        "name": "lightgbm_fast_cv",
                        "component_type": "model",
                        "description": "LightGBM with OHE pipeline, 5-fold StratifiedKFold, early stopping via callbacks",
                        "estimated_impact": 0.20,
                        "rationale": "High-ROI baseline model",
                        "code_outline": "ColumnTransformer + LGBMClassifier(num_leaves=63, learning_rate=0.03, n_estimators=1200)",
                    }
                )
                model_count += 1

            if model_count < 2:
                plan.append(
                    {
                        "name": "xgboost_fast_cv",
                        "component_type": "model",
                        "description": "XGBoost with OHE pipeline, 5-fold CV, moderate depth",
                        "estimated_impact": 0.18,
                        "rationale": "Adds diversity for ensemble",
                        "code_outline": "XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=800, subsample=0.8)",
                    }
                )

        # Exploration arm if capacity allows
        if len(plan) < 4:
            plan.append(
                {
                    "name": "stacking_light",
                    "component_type": "ensemble",
                    "description": "Weighted average of top models using CV rewards as weights; validate submission vs sample",
                    "estimated_impact": 0.12,
                    "rationale": "Cheap ensemble leveraging existing predictions",
                    "code_outline": "Load saved preds, weight by CV reward, validate sample_submission shape/ids",
                }
            )

        return plan[:4]

    def _extract_validation_score(self, stdout: str) -> float | None:
        """Parse validation score from stdout if present."""
        import re

        if not stdout:
            return None

        match = re.search(r"Final Validation Performance:\s*([0-9\\.]+)", stdout)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _validate_plan(
        self,
        plan: list[AblationComponent],
        *,
        state: KaggleState | None = None,
    ) -> list[AblationComponent]:
        """
        Validate and enhance the ablation plan.

        Args:
            plan: Initial plan
            state: Current workflow state (optional, for fast-mode constraints)

        Returns:
            Validated plan
        """
        # Normalize any raw dict entries before applying validation rules.
        plan = self._coerce_components(plan)

        run_mode = str((state or {}).get("run_mode", "")).lower()
        objective = str((state or {}).get("objective", "")).lower()
        domain = str((state or {}).get("domain_detected", "tabular")).lower()
        timeout_cap = (state or {}).get("timeout_per_component")

        if isinstance(timeout_cap, str):
            try:
                timeout_cap = int(timeout_cap)
            except ValueError:
                timeout_cap = None

        fast_mode = (
            bool((state or {}).get("fast_mode")) or run_mode == "mlebench" or "medal" in objective
        )
        if isinstance(timeout_cap, int) and timeout_cap <= 1200:
            fast_mode = True

        # In fast mode, allow smaller-impact but cheap components (e.g., TTA inference).
        min_impact = 0.05 if fast_mode else 0.10

        # Filter out invalid components
        valid_plan = [c for c in plan if c.estimated_impact >= min_impact]

        # Guardrail: block tabular models for image competitions without features.
        if self._is_image_competition_without_features(state):
            tabular_signals = [
                "lightgbm",
                "lgbm",
                "xgboost",
                "catboost",
                "randomforest",
                "logistic",
                "svm",
                "naive",
                "optuna",
                "stacking",
                "ridge",
            ]
            filtered_plan = []
            removed = []
            for comp in valid_plan:
                text = f"{comp.name} {comp.code}".lower()
                if any(sig in text for sig in tabular_signals):
                    removed.append(comp.name)
                    continue
                filtered_plan.append(comp)
            if removed:
                print(
                    f"  ⚠️  Removed tabular components for image competition without features: {', '.join(removed)}"
                )
                valid_plan = filtered_plan

        # NOTE: NN-dominant tree removal logic REMOVED (was harmful for tabular tasks).
        # Tree models (LightGBM, XGBoost, CatBoost) are typically the best performers
        # for tabular classification/regression. Removing them when an MLP happened
        # to score well in one iteration caused severe performance degradation.
        # Tree models should ALWAYS be available for tabular competitions.

        # Limit components (quality over quantity)
        max_components = 3 if fast_mode else 6
        override = os.getenv("KAGGLE_AGENTS_MAX_COMPONENTS")
        if override:
            try:
                override_val = int(override)
                if override_val >= 2:
                    max_components = override_val
            except ValueError:
                print(f"  ⚠️ Invalid KAGGLE_AGENTS_MAX_COMPONENTS='{override}', using default")
        if len(valid_plan) > max_components:
            print(
                f"  ⚠️  Plan has {len(valid_plan)} components - limiting to top {max_components} by impact"
            )
            valid_plan = sorted(valid_plan, key=lambda x: x.estimated_impact, reverse=True)[
                :max_components
            ]

        # Ensure we have enough model components to produce predictions.
        model_count = sum(1 for c in valid_plan if c.component_type == "model")
        tabular_domain = domain.startswith("tabular") or domain in {
            "tabular",
            "tabular_classification",
            "tabular_regression",
        }
        require_two_models = tabular_domain and not fast_mode

        if model_count == 0:
            print("  ⚠️  No 'model' components found - adding a baseline model")
            if domain == "image_to_image" or domain == "image_segmentation":
                # CRITICAL: Use encoder-decoder for image-to-image, not classifier
                baseline = AblationComponent(
                    name="baseline_unet_encoder_decoder",
                    component_type="model",
                    code="U-Net encoder-decoder for pixel-level prediction. Output must be same size as input. Flatten to pixel-level CSV format.",
                    estimated_impact=0.30 if not fast_mode else 0.20,
                    tested=False,
                    actual_impact=None,
                )
            elif domain.startswith("image_"):
                baseline = AblationComponent(
                    name="baseline_resnet18",
                    component_type="model",
                    code="",
                    estimated_impact=0.20 if not fast_mode else 0.10,
                    tested=False,
                    actual_impact=None,
                )
            elif avoid_tree_models and tabular_domain:
                baseline = AblationComponent(
                    name="baseline_tabular_mlp",
                    component_type="model",
                    code="Tabular MLP with StandardScaler, Dropout, softmax for multiclass.",
                    estimated_impact=0.18 if not fast_mode else 0.12,
                    tested=False,
                    actual_impact=None,
                )
            else:
                baseline = AblationComponent(
                    name="baseline_lightgbm",
                    component_type="model",
                    code="",
                    estimated_impact=0.20,
                    tested=False,
                    actual_impact=None,
                )
            valid_plan.append(baseline)
            print(f"     Added: {baseline.name}")

        elif model_count == 1 and require_two_models:
            print("  ⚠️  Only 1 'model' component found - adding second baseline model")
            if avoid_tree_models and tabular_domain:
                baseline_model = AblationComponent(
                    name="baseline_tabular_mlp_2",
                    component_type="model",
                    code="Tabular MLP variant with wider layers or batchnorm.",
                    estimated_impact=0.16 if not fast_mode else 0.10,
                    tested=False,
                    actual_impact=None,
                )
            else:
                baseline_model = AblationComponent(
                    name="baseline_xgboost",
                    component_type="model",
                    code="",
                    estimated_impact=0.18,
                    tested=False,
                    actual_impact=None,
                )
            valid_plan.append(baseline_model)
            print(f"     Added: {baseline_model.name}")

        # Ensure 2-5 components total (fast mode allows 2)
        if len(valid_plan) < 2:
            print("  ⚠️  Plan has fewer than 2 components")
        elif len(valid_plan) > 5 and not fast_mode:
            print(f"  ⚠️  Plan still has {len(valid_plan)} components after filtering")

        # Sort by type: preprocessing first, then models, then ensembles
        # This ensures data is prepared before models are trained
        preprocessing_components = [
            c for c in valid_plan if c.component_type in ["preprocessing", "feature_engineering"]
        ]
        model_components = [c for c in valid_plan if c.component_type == "model"]
        other_components = [
            c
            for c in valid_plan
            if c.component_type not in ["preprocessing", "feature_engineering", "model"]
        ]

        # Reorder: preprocessing first, then models, then ensembles
        valid_plan = preprocessing_components + model_components + other_components

        # Debug log: Show final plan composition
        print(
            f"  📊 Final plan: {len(preprocessing_components)} FE + {len(model_components)} models + {len(other_components)} ensemble = {len(valid_plan)} total"
        )

        return valid_plan

    def _analyze_gaps(
        self, state: KaggleState, previous_plan_str: str, test_results_str: str
    ) -> dict[str, Any]:
        """
        Analyze gaps between results and goal.

        Args:
            state: Current state
            previous_plan_str: JSON string of previous plan
            test_results_str: JSON string of test results

        Returns:
            Dictionary with gap analysis
        """
        competition_info = state["competition_info"]
        current_score = state.get("current_performance_score", 0.0)

        prompt = ANALYZE_GAPS_PROMPT.format(
            previous_plan=previous_plan_str,
            test_results=test_results_str,
            metric=competition_info.evaluation_metric,
            current_score=current_score,
            target_score="SOTA (typically Top 10%)",
            memory_summary=get_memory_summary_for_planning(state),
        )

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            if not getattr(self, "llm", None):
                self.llm = get_llm_for_role(
                    role="planner",
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                )
            response = self.llm.invoke(messages)
            content = get_text_content(response.content).strip()

            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif content.startswith("```") and content.endswith("```"):
                content = content.strip("` \n")

            return json.loads(content)
        except Exception as e:
            print(f"  ⚠️ Gap analysis failed: {e}")
            return {
                "root_causes": ["Unknown (analysis failed)"],
                "missed_opportunities": ["Standard baselines"],
                "improvement_strategy": "Focus on fixing any errors shown in logs.",
            }

    def _parse_llm_plan_response(
        self,
        response_text: str,
        sota_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Parse LLM response into a list of component dictionaries.

        Handles both raw JSON and markdown-wrapped JSON responses.

        Args:
            response_text: Raw LLM response text
            sota_analysis: SOTA analysis for fallback enrichment

        Returns:
            List of component dictionaries
        """
        import re

        # Coerce non-string responses (e.g., list) to string safely
        if not isinstance(response_text, str):
            try:
                response_text = "\n".join(map(str, response_text))  # type: ignore[arg-type]
            except Exception:
                response_text = str(response_text)

        # Try to extract JSON array from response
        text = response_text.strip()

        # Remove markdown code blocks if present
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Find JSON array
        json_start = text.find("[")
        json_end = text.rfind("]") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = text[json_start:json_end]
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    # Validate and normalize each component
                    valid_components = []
                    for item in parsed:
                        if isinstance(item, dict):
                            component = {
                                "name": item.get("name", "unnamed"),
                                "component_type": item.get("component_type", "model"),
                                "description": item.get("description", ""),
                                "estimated_impact": float(item.get("estimated_impact", 0.1)),
                                "code_outline": item.get(
                                    "code_outline", item.get("description", "")
                                ),
                                "rationale": item.get("rationale", ""),
                            }
                            valid_components.append(component)
                    return valid_components
            except json.JSONDecodeError as e:
                print(f"  ⚠️ JSON parse error: {e}")

        # If JSON parsing fails, return empty (caller will use fallback)
        return []

    def _is_image_competition_without_features(self, state: KaggleState | None) -> bool:
        """
        Detect if competition is image-based but has no tabular features.

        This catches cases where domain detection fails but the competition
        is clearly image-based (has image files and minimal train.csv columns).

        Args:
            state: Current workflow state

        Returns:
            True if this appears to be an image competition without tabular features
        """
        if state is None:
            return False

        # Check for image files in data directory
        data_dir = state.get("data_dir", "")
        has_images = False
        if data_dir:
            from pathlib import Path

            data_path = Path(data_dir)
            if data_path.exists():
                # Check for common image directories (train/, test/, images/)
                for subdir in ["train", "test", "images", "train_images", "test_images"]:
                    subdir_path = data_path / subdir
                    if subdir_path.exists() and subdir_path.is_dir():
                        # Check if directory contains image files
                        image_extensions = {
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".gif",
                            ".bmp",
                            ".tiff",
                            ".webp",
                        }
                        for f in subdir_path.iterdir():
                            if f.suffix.lower() in image_extensions:
                                has_images = True
                                break
                    if has_images:
                        break

        # Check if train.csv has minimal columns (only id + label)
        train_csv_minimal = False
        train_csv_path = state.get("train_csv_path", "")
        if train_csv_path:
            from pathlib import Path

            import pandas as pd

            train_path = Path(train_csv_path)
            if train_path.exists():
                try:
                    train_df = pd.read_csv(train_path, nrows=5)  # Only read header
                    # If train.csv has 2 or fewer columns, it's likely just id + label
                    train_csv_minimal = len(train_df.columns) <= 2
                except Exception:
                    pass

        if has_images and train_csv_minimal:
            print("  [WARNING] Detected IMAGE competition without tabular features!")
            print(f"            - Has image files: {has_images}")
            print(f"            - train.csv minimal (<=2 cols): {train_csv_minimal}")
            return True

        return False

    def _detect_multimodal_competition(self, state: KaggleState | None) -> dict[str, Any]:
        """
        Detect if competition has both images AND rich tabular features.

        Multi-modal competitions (like leaf-classification) have:
        - Images in train/ or test/ directories
        - Rich tabular features in train.csv (>10 columns)

        Returns guidance for hybrid model strategies.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with detection results and strategy recommendations
        """
        if state is None:
            return {"type": "unknown", "is_multimodal": False}

        from pathlib import Path

        # Check for image files
        data_dir = state.get("data_dir", "")
        has_images = False
        image_count = 0

        if data_dir:
            data_path = Path(data_dir)
            if data_path.exists():
                for subdir in ["train", "test", "images", "train_images", "test_images"]:
                    subdir_path = data_path / subdir
                    if subdir_path.exists() and subdir_path.is_dir():
                        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
                        for f in subdir_path.iterdir():
                            if f.suffix.lower() in image_extensions:
                                has_images = True
                                image_count += 1
                                if image_count > 10:  # Found enough images
                                    break
                    if has_images:
                        break

        # Check if train.csv has rich tabular features
        has_rich_tabular = False
        tabular_feature_count = 0
        train_csv_path = state.get("train_csv_path", "") or state.get("train_data_path", "")

        if train_csv_path:
            import pandas as pd

            train_path = Path(train_csv_path)
            if train_path.exists():
                try:
                    train_df = pd.read_csv(train_path, nrows=5)
                    # Count non-ID, non-target columns
                    exclude_cols = {"id", "target", "species", "label", "class", "image", "image_id"}
                    feature_cols = [
                        c for c in train_df.columns if c.lower() not in exclude_cols
                    ]
                    tabular_feature_count = len(feature_cols)
                    has_rich_tabular = tabular_feature_count >= 10
                except Exception:
                    pass

        # Determine competition type and strategy
        if has_images and has_rich_tabular:
            print("\n  🔍 MULTI-MODAL COMPETITION DETECTED:")
            print(f"      - Has image files: {has_images}")
            print(f"      - Tabular features: {tabular_feature_count}")

            return {
                "type": "multi_modal",
                "is_multimodal": True,
                "has_images": True,
                "has_rich_tabular": True,
                "tabular_features": tabular_feature_count,
                "strategy": "hybrid_cnn_tabular",
                "recommendation": (
                    "Use Keras Functional API with multi-input model: "
                    "CNN branch (EfficientNet) for images + MLP branch for tabular features. "
                    "Alternatively, the pre-extracted tabular features may be sufficient "
                    "for competitive performance with LightGBM/XGBoost alone."
                ),
                "priority_models": [
                    "LightGBM with all tabular features (fast, often competitive)",
                    "XGBoost with all tabular features",
                    "Hybrid CNN+Tabular (best but slower)",
                ],
            }
        if has_images:
            return {
                "type": "image_only",
                "is_multimodal": False,
                "has_images": True,
                "has_rich_tabular": False,
                "strategy": "efficientnet",
                "recommendation": "Use transfer learning with EfficientNet or ResNet.",
            }
        return {
            "type": "tabular_only",
            "is_multimodal": False,
            "has_images": False,
            "has_rich_tabular": has_rich_tabular,
            "tabular_features": tabular_feature_count,
            "strategy": "lightgbm_xgboost",
            "recommendation": "Use gradient boosting (LightGBM, XGBoost, CatBoost).",
        }

    def _handle_debug_loop_trigger(self, state: KaggleState) -> dict[str, Any] | None:
        """
        Handle debug loop trigger from Meta-Evaluator.

        When a performance gap is detected, create a focused debug plan
        for the underperforming model instead of the normal ablation plan.

        Args:
            state: Current workflow state

        Returns:
            Debug plan if triggered, None otherwise
        """
        trigger_debug = state.get("trigger_debug_loop", False)
        if not trigger_debug:
            return None

        debug_target = state.get("debug_target_model", "unknown")
        debug_hints = state.get("debug_hints", [])
        performance_gap = state.get("performance_gap", 0.0)

        print(f"\n  🔧 DEBUG LOOP TRIGGERED for: {debug_target}")
        print(f"      Performance gap: {performance_gap:.2f}")
        print("      Debug hints:")
        for hint in debug_hints[:3]:
            print(f"        - {hint}")

        # Create focused debug plan
        debug_components = [
            AblationComponent(
                name=f"debug_{debug_target}_labelencoder",
                component_type="debug",
                code=(
                    f"# Debug {debug_target}: Verify LabelEncoder class order\n"
                    "# Compare with other models' class_order.npy files\n"
                    "# Ensure consistent encoding across all models"
                ),
                estimated_impact=0.8,
            ),
            AblationComponent(
                name=f"debug_{debug_target}_preprocessing",
                component_type="debug",
                code=(
                    f"# Debug {debug_target}: Check preprocessing pipeline\n"
                    "# Verify train/val splits use same random_state\n"
                    "# Compare feature scaling and encoding"
                ),
                estimated_impact=0.7,
            ),
            AblationComponent(
                name=f"debug_{debug_target}_hyperparams",
                component_type="debug",
                code=(
                    f"# Debug {debug_target}: Review hyperparameters\n"
                    "# Check class_weight setting\n"
                    "# Verify objective function matches metric"
                ),
                estimated_impact=0.6,
            ),
        ]

        return {
            "ablation_plan": debug_components,
            "is_debug_iteration": True,
            "debug_target": debug_target,
            "debug_hints": debug_hints,
            "skip_normal_planning": True,
        }

    def _create_fallback_plan(
        self,
        domain: str,
        sota_analysis: dict[str, Any],
        curriculum_insights: str = "",
        *,
        state: KaggleState | None = None,
    ) -> list[dict[str, Any]]:
        """
        Create domain-specific fallback plan when LLM parsing fails.

        Routes to appropriate domain-specific fallback method based on domain type.

        Args:
            domain: Competition domain (e.g., 'image_classification', 'text_classification', 'tabular')
            sota_analysis: SOTA analysis results
            curriculum_insights: Insights from previous iterations (optional)

        Returns:
            List of component dictionaries (3-5 components depending on domain)
        """
        print(f"  [DEBUG] Creating fallback plan for domain: '{domain}'")

        # SAFETY CHECK: Prevent tabular models for image competitions
        # This catches cases where domain detection fails but we can clearly see
        # image files exist and train.csv has minimal columns (just id + label)
        if self._is_image_competition_without_features(state):
            print(
                "  [WARNING] Forcing IMAGE fallback plan (detected image competition without features)"
            )
            print("            Tree models (LightGBM/XGBoost) require tabular features!")
            # Force image classification fallback
            return self._create_image_fallback_plan(
                "image_classification", sota_analysis, fast_mode=False
            )

        run_mode = str((state or {}).get("run_mode", "")).lower()
        objective = str((state or {}).get("objective", "")).lower()
        timeout_cap = (state or {}).get("timeout_per_component")
        if isinstance(timeout_cap, str):
            try:
                timeout_cap = int(timeout_cap)
            except ValueError:
                timeout_cap = None

        # Speed-first when optimizing for MLE-bench medals or tight component caps.
        fast_mode = (
            run_mode == "mlebench"
            or "medal" in objective
            or (isinstance(timeout_cap, int) and timeout_cap <= 1200)
        )

        # Define domain sets for routing
        IMAGE_DOMAINS = {
            "image_classification",
            "image_regression",
            "object_detection",
            "image_to_image",
            "image_segmentation",
        }
        TEXT_DOMAINS = {"text_classification", "text_regression", "seq_to_seq"}
        AUDIO_DOMAINS = {"audio_classification", "audio_regression"}

        # Route to domain-specific fallback method
        if domain in ("image_to_image", "image_segmentation"):
            # CRITICAL: Image-to-image tasks require pixel-level predictions
            # Use specialized fallback plan with encoder-decoder architectures
            return self._create_image_to_image_fallback_plan(
                domain, sota_analysis, fast_mode=fast_mode
            )
        if domain in IMAGE_DOMAINS or domain.startswith("image_"):
            # Includes object_detection which uses CNN-based approaches
            return self._create_image_fallback_plan(domain, sota_analysis, fast_mode=fast_mode)
        if domain in TEXT_DOMAINS or domain.startswith("text_"):
            return self._create_text_fallback_plan(domain, sota_analysis)
        if domain in AUDIO_DOMAINS or domain.startswith("audio_"):
            return self._create_audio_fallback_plan(domain, sota_analysis)
        # Tabular (existing logic)
        return self._create_tabular_fallback_plan(
            domain,
            sota_analysis,
            curriculum_insights,
            fast_mode=fast_mode,
            state=state,
        )

    def _create_tabular_fallback_plan(
        self,
        domain: str,
        sota_analysis: dict[str, Any],
        curriculum_insights: str = "",
        *,
        fast_mode: bool = False,
        state: KaggleState | None = None,
    ) -> list[dict[str, Any]]:
        """
        Create fallback plan for tabular competitions (classification/regression).

        Uses tree-based models (LightGBM, XGBoost, CatBoost) with ensemble.

        Args:
            domain: Competition domain
            sota_analysis: SOTA analysis results
            curriculum_insights: Insights from previous iterations (optional)
            fast_mode: Whether to use speed-optimized plan
            state: Current workflow state (used to filter failed components)

        Returns:
            List of component dictionaries (5 components: 1 FE + 4 models + 1 ensemble)
        """
        # Get failed components to avoid repeating them
        failed_names = set()
        if state:
            failed_names = set(state.get("failed_component_names", []))
            if failed_names:
                print(f"   📋 Filtering out previously failed components: {failed_names}")

        # Note: Curriculum insights are logged but fallback uses fixed plan
        # In future iterations, could use insights to reorder components
        if fast_mode:
            # Full candidate pool with alternatives for failed components
            all_fast_candidates = [
                {
                    "name": "lightgbm_fast_cv",
                    "component_type": "model",
                    "description": "LightGBM baseline tuned for speed (no Optuna). Use fewer estimators + early stopping/callbacks. Respect KAGGLE_AGENTS_CV_FOLDS for faster iteration.",
                    "estimated_impact": 0.18,
                    "rationale": "High ROI baseline for tabular tasks; fast enough to iterate under tight time budgets (MLE-bench).",
                    "code_outline": "LGBMClassifier/Regressor with sane defaults, 3-fold CV when FAST_MODE, save OOF/test preds",
                },
                {
                    "name": "xgboost_fast_cv",
                    "component_type": "model",
                    "description": "XGBoost baseline tuned for speed (no Optuna). Use hist/gpu_hist where available. Respect time budget and fold count env vars.",
                    "estimated_impact": 0.16,
                    "rationale": "Provides diversity vs LightGBM with similar compute budget; useful for a quick ensemble.",
                    "code_outline": "XGBClassifier/Regressor with fixed params, 3-fold CV when FAST_MODE, save OOF/test preds",
                },
                # ALTERNATIVE MODELS (used when primary models fail)
                {
                    "name": "catboost_fast_cv",
                    "component_type": "model",
                    "description": "CatBoost baseline tuned for speed (no Optuna). Handles categoricals natively. Respect time budget and fold count env vars.",
                    "estimated_impact": 0.15,
                    "rationale": "Alternative to XGBoost/LightGBM with different regularization; handles categoricals well.",
                    "code_outline": "CatBoostClassifier/Regressor with sane defaults, 3-fold CV when FAST_MODE, save OOF/test preds",
                },
                {
                    "name": "logistic_tfidf",
                    "component_type": "model",
                    "description": "Logistic Regression with TF-IDF features. Very fast fallback for text-heavy tabular data.",
                    "estimated_impact": 0.12,
                    "rationale": "Extremely fast linear model; useful when tree models timeout.",
                    "code_outline": "TfidfVectorizer + LogisticRegression/Ridge, save OOF/test preds",
                },
                {
                    "name": "random_forest_fast",
                    "component_type": "model",
                    "description": "Random Forest baseline with limited trees (n_estimators=200) for speed.",
                    "estimated_impact": 0.13,
                    "rationale": "Robust tree ensemble that rarely fails; good fallback option.",
                    "code_outline": "RandomForestClassifier/Regressor with n_estimators=200, 3-fold CV, save OOF/test preds",
                },
                {
                    "name": "stacking_ensemble",
                    "component_type": "ensemble",
                    "description": "Stack OOF predictions from available models with LogisticRegression/Ridge meta-learner. Fallback to weighted average if needed.",
                    "estimated_impact": 0.10,
                    "rationale": "Cheap ensemble step that often improves generalization without additional heavy training.",
                    "code_outline": "Load models/oof_*.npy + models/test_*.npy, fit meta-model on OOF, predict test, write submission",
                },
            ]

            # Filter out failed components
            filtered_plan = [c for c in all_fast_candidates if c["name"] not in failed_names]

            # Ensure we have at least 2 models (excluding ensemble) for meaningful stacking
            model_count = sum(1 for c in filtered_plan if c["component_type"] == "model")
            if model_count < 2:
                print("   ⚠️ Less than 2 models available after filtering. Adding simple baseline.")
                # Add a very simple baseline that's unlikely to fail
                filtered_plan.insert(0, {
                    "name": "simple_ridge_baseline",
                    "component_type": "model",
                    "description": "Simple Ridge regression baseline. Cannot fail, always produces predictions.",
                    "estimated_impact": 0.08,
                    "rationale": "Failsafe baseline that always works.",
                    "code_outline": "StandardScaler + Ridge, 3-fold CV, save OOF/test preds",
                })

            # Keep top 2 models + ensemble (avoid bloated plans)
            models = [c for c in filtered_plan if c["component_type"] == "model"][:2]
            ensemble = [c for c in filtered_plan if c["component_type"] == "ensemble"][:1]
            return models + ensemble

        plan = []

        # ALWAYS add feature engineering first (high impact)
        plan.append(
            {
                "name": "advanced_feature_engineering",
                "component_type": "feature_engineering",
                "description": "Create polynomial features (degree 2), feature interactions (ratio, diff, product), statistical transformations (log, sqrt), and target encoding for categorical features",
                "estimated_impact": 0.15,
                "rationale": "Comprehensive feature engineering improves scores by 10-20% in tabular competitions",
                "code_outline": "Use PolynomialFeatures(degree=2), create ratio/diff/product features, apply log/sqrt transforms, use TargetEncoder",
            }
        )

        # ALWAYS add 3 diverse models for ensemble diversity
        plan.extend(
            [
                {
                    "name": "lightgbm_optuna_tuned",
                    "component_type": "model",
                    "description": "LightGBM with Optuna hyperparameter optimization: 15 trials, tuning learning_rate, num_leaves, max_depth, min_child_samples",
                    "estimated_impact": 0.22,
                    "rationale": "LightGBM consistently wins tabular competitions. Optuna finds better parameters than manual tuning.",
                    "code_outline": "LGBMRegressor/Classifier with OptunaSearchCV, 5-fold CV, early_stopping_rounds=100",
                },
                {
                    "name": "xgboost_optuna_tuned",
                    "component_type": "model",
                    "description": "XGBoost with Optuna hyperparameter optimization: 15 trials, tuning max_depth, learning_rate, subsample, colsample_bytree",
                    "estimated_impact": 0.20,
                    "rationale": "XGBoost provides different regularization than LightGBM. Optuna ensures optimal capacity.",
                    "code_outline": "XGBRegressor/Classifier with OptunaSearchCV, 5-fold CV, early_stopping_rounds=50",
                },
                {
                    "name": "catboost_optuna_tuned",
                    "component_type": "model",
                    "description": "CatBoost with Optuna hyperparameter optimization: 15 trials, tuning depth, learning_rate, l2_leaf_reg",
                    "estimated_impact": 0.19,
                    "rationale": "CatBoost handles categorical features natively. Tuning depth is critical for performance.",
                    "code_outline": "CatBoostRegressor/Classifier with OptunaSearchCV, cat_features parameter, 5-fold CV",
                },
                {
                    "name": "neural_network_mlp",
                    "component_type": "model",
                    "description": "Simple MLP Neural Network using Scikit-Learn or PyTorch (if available). Standard scaling is CRITICAL.",
                    "estimated_impact": 0.15,
                    "rationale": "Neural Networks capture different patterns than tree-based models, adding valuable diversity to the ensemble.",
                    "code_outline": "MLPClassifier/Regressor or PyTorch simple net. Must use StandardScaler/MinMaxScaler on inputs. Early stopping.",
                },
            ]
        )

        # ALWAYS add stacking ensemble (combines the 4 models above)
        plan.append(
            {
                "name": "stacking_ensemble",
                "component_type": "ensemble",
                "description": "Stack LightGBM, XGBoost, CatBoost, and NN predictions using Ridge/Logistic regression as meta-learner",
                "estimated_impact": 0.25,
                "rationale": "Stacking combines diverse models (Trees + NN) and typically improves scores by 5-10%",
                "code_outline": "StackingRegressor/Classifier with base_estimators=[lgb, xgb, cat, nn], final_estimator=Ridge/LogisticRegression, cv=5",
            }
        )

        # Filter out failed components (if any)
        if failed_names:
            plan = [c for c in plan if c["name"] not in failed_names]
            # Ensure we still have at least 1 model
            model_count = sum(1 for c in plan if c["component_type"] == "model")
            if model_count == 0:
                print("   ⚠️ All models filtered out! Adding simple baseline.")
                plan.insert(0, {
                    "name": "simple_ridge_baseline",
                    "component_type": "model",
                    "description": "Simple Ridge regression baseline. Cannot fail, always produces predictions.",
                    "estimated_impact": 0.08,
                    "rationale": "Failsafe baseline that always works.",
                    "code_outline": "StandardScaler + Ridge, 5-fold CV, save OOF/test preds",
                })

        return plan

    def _create_image_fallback_plan(
        self,
        domain: str,
        sota_analysis: dict[str, Any],
        *,
        fast_mode: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Create fallback plan for image competitions (PyTorch/TensorFlow DataLoaders).

        Uses transfer learning with pre-trained CNNs (EfficientNet, ResNet).

        Args:
            domain: Competition domain (image_classification, image_regression, etc.)
            sota_analysis: SOTA analysis results
            fast_mode: If True, return minimal 2-component plan for speed

        Returns:
            List of component dictionaries (2 in fast mode, 3 normally)
        """
        is_regression = "regression" in domain
        task = "regression" if is_regression else "classification"

        # FAST MODE: Only 2 components for maximum speed (MLE-bench optimization)
        if fast_mode:
            return [
                {
                    "name": f"efficientnet_b0_fast_{task}",
                    "component_type": "model",
                    "description": "EfficientNet-B0 with FROZEN backbone. Only train classifier head for 2-3 epochs. Use 2-fold CV (KAGGLE_AGENTS_CV_FOLDS=2). Mixed precision training. Lightweight augmentations only (flip, normalize). IMPLEMENT soft-deadline pattern.",
                    "estimated_impact": 0.30,
                    "rationale": "Frozen backbone = fastest training. 2 epochs is enough for head fine-tuning. This prioritizes getting a valid submission quickly.",
                    "code_outline": "efficientnet_b0(pretrained=True), freeze all backbone layers, train head only, 2 epochs, 2-fold CV, save best checkpoint, implement _check_deadline() pattern",
                },
                {
                    "name": "tta_inference_only",
                    "component_type": "ensemble",
                    "description": "Test-Time Augmentation ONLY (no additional training). Load the single trained model and apply 5 simple transforms (original, hflip, vflip, rotate90, rotate180), average predictions. Write submission.csv.",
                    "estimated_impact": 0.05,
                    "rationale": "Free accuracy boost without additional training time. Just inference with multiple transforms.",
                    "code_outline": "Load models/best_model.* (auto-detect extension), for each test image: apply transforms, average predictions, clip to [0,1], write submission.csv",
                },
            ]

        # NORMAL MODE: 3 components (2 models + TTA ensemble)
        return [
            {
                "name": f"efficientnet_b0_{task}",
                "component_type": "model",
                "description": f"EfficientNet-B0 pre-trained fine-tuned for {task}. PyTorch DataLoader with ImageFolder or custom Dataset. Data augmentation (rotation, flip, color jitter). Use transfer learning from ImageNet weights.",
                "estimated_impact": 0.28,
                "rationale": "EfficientNet achieves SOTA on ImageNet with excellent efficiency. Transfer learning transfers learned features. Data augmentation prevents overfitting on small datasets.",
                "code_outline": "torchvision.models.efficientnet_b0(pretrained=True), replace classifier head, train with CrossEntropyLoss/MSELoss, CV folds via KAGGLE_AGENTS_CV_FOLDS, save OOF predictions for ensemble",
            },
            {
                "name": f"resnet50_{task}",
                "component_type": "model",
                "description": "ResNet50 fine-tuned with different augmentation strategy (Cutout, Mixup) for architectural diversity in ensemble.",
                "estimated_impact": 0.24,
                "rationale": "ResNet provides complementary features vs EfficientNet. Ensemble benefits from architectural diversity.",
                "code_outline": "torchvision.models.resnet50(pretrained=True), replace head, Cutout + Mixup augmentations, AdamW, CV folds via KAGGLE_AGENTS_CV_FOLDS",
            },
            {
                "name": "tta_ensemble",
                "component_type": "ensemble",
                "description": "Test-time augmentation (TTA) + weighted ensemble of EfficientNet and ResNet predictions. Apply multiple transforms to test images and average predictions.",
                "estimated_impact": 0.15,
                "rationale": "TTA averages predictions over multiple augmented views of each test image, reducing variance. Weighted ensemble (by CV score) combines different architectures. Typical +2-5% improvement.",
                "code_outline": "For each test image: apply 5 transforms (original, hflip, vflip, rotate90, rotate270), get predictions from each model, average TTA predictions per model, then weighted average models by CV score",
            },
        ]

    def _create_image_to_image_fallback_plan(
        self,
        domain: str,
        sota_analysis: dict[str, Any],
        *,
        fast_mode: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Create fallback plan for image-to-image tasks (denoising, super-resolution, style transfer).

        CRITICAL: These tasks require PIXEL-LEVEL predictions, not per-image predictions.
        The submission format is typically one row per pixel: id=image_row_col, value=pixel_intensity.

        Args:
            domain: Competition domain (image_to_image)
            sota_analysis: SOTA analysis results
            fast_mode: If True, return minimal plan for speed

        Returns:
            List of component dictionaries with encoder-decoder architectures
        """
        if fast_mode:
            return [
                {
                    "name": "simple_autoencoder_denoiser",
                    "component_type": "model",
                    "description": """Simple convolutional autoencoder for image-to-image transformation.

CRITICAL - THIS IS A PIXEL-LEVEL PREDICTION TASK:
- Model must output FULL IMAGE (same H x W as input), NOT a single value
- Use encoder-decoder architecture (Conv2d -> ConvTranspose2d)
- DO NOT use classifiers (EfficientNet, ResNet with FC head)

Architecture:
- Encoder: 3-4 Conv2d layers with ReLU, max pooling
- Decoder: 3-4 ConvTranspose2d layers with ReLU
- Output: Same size as input (H x W) for grayscale

Training:
- Input: noisy/degraded images
- Target: clean images
- Loss: MSE or L1 loss between output and target image

SUBMISSION FORMAT (CRITICAL - MUST FOLLOW):
```python
sample_sub = pd.read_csv(sample_submission_path)
expected_rows = len(sample_sub)  # Typically MILLIONS of rows

submission_rows = []
for img_path in sorted(test_images):
    img_id = img_path.stem  # e.g., "1" from "1.png"
    pred = model(preprocess(img))  # OUTPUT: (H, W) image
    H, W = pred.shape
    for row in range(H):
        for col in range(W):
            pixel_id = f"{img_id}_{row+1}_{col+1}"
            submission_rows.append({"id": pixel_id, "value": float(pred[row, col])})

assert len(submission_rows) == expected_rows
pd.DataFrame(submission_rows).to_csv("submission.csv", index=False)
```""",
                    "estimated_impact": 0.35,
                    "rationale": "Simple autoencoder is fast to train and provides baseline for denoising. Pixel-level output is critical for correct submission format.",
                    "code_outline": "Conv2d encoder, ConvTranspose2d decoder, MSE loss, output same size as input, flatten to pixel-level CSV",
                },
                {
                    "name": "submission_format_validator",
                    "component_type": "ensemble",
                    "description": "Validate pixel-level submission format matches sample_submission.csv exactly.",
                    "estimated_impact": 0.05,
                    "rationale": "Critical validation to catch format errors before submission.",
                    "code_outline": "Load sample_sub, verify row count matches, verify ID format matches exactly",
                },
            ]

        # Full mode: U-Net and ensemble
        return [
            {
                "name": "unet_encoder_decoder",
                "component_type": "model",
                "description": """U-Net architecture for image-to-image transformation with skip connections.

CRITICAL - THIS IS A PIXEL-LEVEL PREDICTION TASK:
- Model must output FULL IMAGE (same H x W as input)
- U-Net preserves fine details through skip connections
- DO NOT use classifiers (EfficientNet, ResNet with FC head)

U-Net Architecture:
- Encoder: 4 blocks of (Conv2d, BatchNorm, ReLU, MaxPool)
- Bottleneck: Conv2d block
- Decoder: 4 blocks of (ConvTranspose2d, concat skip, Conv2d, BatchNorm, ReLU)
- Output: Conv2d(1, 1, 1) for single-channel grayscale output

SUBMISSION FORMAT (CRITICAL):
Read sample_submission.csv to get expected row count (millions of rows).
Flatten each output image to pixel format: {img_id}_{row}_{col} -> value""",
                "estimated_impact": 0.40,
                "rationale": "U-Net is SOTA for image-to-image tasks. Skip connections preserve fine details crucial for denoising/super-resolution.",
                "code_outline": "PyTorch U-Net with skip connections, MSE loss, 3-5 epochs, output full image, flatten to pixel CSV",
            },
            {
                "name": "residual_autoencoder",
                "component_type": "model",
                "description": """Residual autoencoder that predicts the NOISE (residual) rather than clean image.

Architecture:
- Similar to U-Net but predicts: clean = noisy - predicted_noise
- Residual learning makes training more stable
- Output: Same size as input

This provides model diversity for ensemble.""",
                "estimated_impact": 0.35,
                "rationale": "Residual learning (predicting noise) often works better than direct denoising. Provides ensemble diversity.",
                "code_outline": "Conv encoder-decoder, predict residual, output = input - residual, same pixel-level submission format",
            },
            {
                "name": "pixel_ensemble_average",
                "component_type": "ensemble",
                "description": """Average predictions from U-Net and Residual autoencoder at pixel level.

1. Load predictions from both models
2. Average pixel values: final[i,j] = (unet[i,j] + residual[i,j]) / 2
3. Flatten to submission format
4. Validate row count matches sample_submission.csv""",
                "estimated_impact": 0.10,
                "rationale": "Ensembling reduces prediction variance. Simple average works well for image tasks.",
                "code_outline": "Load both model outputs, pixel-wise average, flatten to CSV, validate format",
            },
        ]

    def _create_text_fallback_plan(
        self,
        domain: str,
        sota_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Create fallback plan for text/NLP competitions (HuggingFace transformers).

        Uses pre-trained language models (RoBERTa, DistilBERT, or T5 for seq2seq).

        Args:
            domain: Competition domain (text_classification, seq_to_seq, etc.)
            sota_analysis: SOTA analysis results

        Returns:
            List of component dictionaries (3 components for classification, 1 for seq2seq)
        """
        if domain == "seq_to_seq":
            # Sequence-to-sequence tasks (translation, text normalization, summarization)
            return [
                {
                    "name": "t5_base_seq2seq",
                    "component_type": "model",
                    "description": "T5-base fine-tuned for seq2seq task using HuggingFace Trainer API. T5 is designed for text-to-text tasks.",
                    "estimated_impact": 0.30,
                    "rationale": "T5 (Text-to-Text Transfer Transformer) is specifically designed for seq2seq tasks. Achieves SOTA on translation, summarization, and text normalization benchmarks.",
                    "code_outline": "transformers.T5ForConditionalGeneration.from_pretrained('t5-base'), T5Tokenizer, Seq2SeqTrainer with DataCollatorForSeq2Seq, train with learning_rate=1e-4, evaluate with BLEU/ROUGE metrics",
                }
            ]
        # Classification or regression tasks
        return [
            {
                "name": "roberta_classifier",
                "component_type": "model",
                "description": "RoBERTa-base fine-tuned for text classification with learning rate warmup and linear decay schedule.",
                "estimated_impact": 0.28,
                "rationale": "RoBERTa improves on BERT with dynamic masking and larger training corpus. Achieves SOTA on GLUE, SuperGLUE, and many NLP benchmarks. Warmup stabilizes training.",
                "code_outline": "transformers.RobertaForSequenceClassification.from_pretrained('roberta-base'), AutoTokenizer, Trainer API with TrainingArguments, AdamW optimizer with warmup_steps=500, 5-fold StratifiedKFold CV, save OOF predictions",
            },
            {
                "name": "distilbert_classifier",
                "component_type": "model",
                "description": "DistilBERT fine-tuned (60% faster than BERT, lighter for ensemble diversity).",
                "estimated_impact": 0.22,
                "rationale": "DistilBERT is 60% faster and 40% smaller than BERT while retaining 97% of performance through knowledge distillation. Provides architectural diversity for ensemble while being computationally efficient.",
                "code_outline": "transformers.DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased'), similar training setup to RoBERTa, 5-fold CV",
            },
            {
                "name": "transformer_ensemble",
                "component_type": "ensemble",
                "description": "Weighted average of RoBERTa and DistilBERT predictions using CV scores as weights.",
                "estimated_impact": 0.12,
                "rationale": "Different architectures (RoBERTa vs DistilBERT) capture different linguistic patterns. Ensemble reduces variance and overfitting to specific model biases.",
                "code_outline": "Load OOF predictions from both models, compute optimal weights via Ridge regression on validation fold, apply weighted average to test predictions",
            },
        ]

    def _create_audio_fallback_plan(
        self,
        domain: str,
        sota_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Create fallback plan for audio competitions (mel-spectrograms + CNNs).

        Converts audio to spectrograms, then uses image models.

        Args:
            domain: Competition domain (audio_classification, audio_regression)
            sota_analysis: SOTA analysis results

        Returns:
            List of component dictionaries (4 components: 1 preprocessing + 2 models + 1 ensemble)
        """
        return [
            {
                "name": "mel_spectrogram_preprocessing",
                "component_type": "preprocessing",
                "description": "Convert audio files to mel-spectrograms using librosa. Save as PNG images for CNN input.",
                "estimated_impact": 0.20,
                "rationale": "Mel-spectrograms are the standard time-frequency representation for audio. Convert audio problem to computer vision problem, enabling use of powerful pre-trained image models.",
                "code_outline": "Use librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128), convert to dB scale with librosa.power_to_db(), normalize to [0, 255], save as 3-channel PNG to spectrograms/ directory",
            },
            {
                "name": "efficientnet_audio",
                "component_type": "model",
                "description": "EfficientNet-B0 trained on mel-spectrogram images. Transfer learning from ImageNet.",
                "estimated_impact": 0.25,
                "rationale": "CNNs excel at recognizing patterns in spectrograms (frequency bands, temporal patterns). EfficientNet provides excellent accuracy with computational efficiency.",
                "code_outline": "Load mel-spectrogram images with PyTorch DataLoader, torchvision.models.efficientnet_b0(pretrained=True), replace classifier, train with data augmentation on spectrograms",
            },
            {
                "name": "resnet_audio",
                "component_type": "model",
                "description": "ResNet50 for architectural diversity in ensemble.",
                "estimated_impact": 0.20,
                "rationale": "ResNet learns different features than EfficientNet due to different architecture (residual connections). Ensemble benefits from this diversity.",
                "code_outline": "Similar pipeline to EfficientNet but with torchvision.models.resnet50(pretrained=True)",
            },
            {
                "name": "audio_ensemble",
                "component_type": "ensemble",
                "description": "Weighted average of EfficientNet and ResNet predictions.",
                "estimated_impact": 0.12,
                "rationale": "Ensemble reduces overfitting to specific architecture biases and improves generalization.",
                "code_outline": "Load OOF predictions, compute weights by CV score, weighted average for test predictions",
            },
        ]

    def _format_sota_solutions(self, solutions: list[SOTASolution]) -> str:
        """Format SOTA solutions for prompts (summary version without code)."""
        formatted = []
        for sol in solutions[:5]:  # Top 5
            formatted.append(f"""
Title: {sol.title}
Votes: {sol.votes}
Models: {", ".join(sol.models_used) if sol.models_used else "N/A"}
Features: {", ".join(sol.feature_engineering) if sol.feature_engineering else "N/A"}
Ensemble: {sol.ensemble_approach or "N/A"}
""")
        return "\n---\n".join(formatted)

    def _estimate_complexity(self, sol: SOTASolution) -> str:
        """
        Estimate time complexity based on code patterns.

        Args:
            sol: SOTA solution to analyze

        Returns:
            Complexity level: "Low", "Medium", or "High" with explanation
        """
        high_complexity_signals = [
            "Ensemble",
            "Stacking",
            "stacking",
            "VotingClassifier",
            "BaggingClassifier",
            "StackingClassifier",
            "StackingRegressor",
            "neural",
            "deep",
            "LSTM",
            "Transformer",
            "BERT",
            "CNN",
            "optuna",
            "hyperopt",
            "GridSearchCV",
            "RandomizedSearchCV",
            "n_estimators=5000",
            "n_estimators=10000",
            "epochs=100",
        ]

        medium_complexity_signals = [
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "RandomForest",
            "n_estimators=1000",
            "n_estimators=2000",
            "cross_val",
            "KFold",
            "StratifiedKFold",
        ]

        # Build text to check from all solution fields
        text_to_check = " ".join(sol.models_used or [])
        text_to_check += " " + (sol.ensemble_approach or "")
        text_to_check += " " + " ".join(sol.strategies or [])
        if sol.code_snippets:
            text_to_check += " " + " ".join(sol.code_snippets[:2])

        text_lower = text_to_check.lower()

        # Count signals
        high_count = sum(1 for signal in high_complexity_signals if signal.lower() in text_lower)
        medium_count = sum(
            1 for signal in medium_complexity_signals if signal.lower() in text_lower
        )

        if high_count >= 3:
            return "High (likely slow - heavy ensembles/optimization/deep learning)"
        if high_count >= 1 or medium_count >= 2:
            return "Medium (moderate training time - standard ML pipeline)"
        return "Low (fast - simple models, quick iteration)"

    def _format_sota_details(self, solutions: list[SOTASolution]) -> str:
        """
        Format SOTA solutions with code snippets, votes, and complexity estimation.

        This provides detailed information for the "Adopt & Improve" strategy,
        allowing the planner to directly copy successful approaches.

        Args:
            solutions: List of SOTA solutions from search

        Returns:
            Formatted string with detailed solution info including code snippets
        """
        if not solutions:
            return "No SOTA solutions found. Create a baseline plan using domain best practices."

        details = []
        for i, sol in enumerate(solutions[:3], 1):  # Top 3 to save tokens
            # Estimate complexity based on code patterns
            complexity = self._estimate_complexity(sol)

            # Get code snippet (truncated to 1500 chars as per user preference)
            code_snippet = ""
            if sol.code_snippets:
                code_snippet = sol.code_snippets[0][:1500]
                if len(sol.code_snippets[0]) > 1500:
                    code_snippet += "\n... (truncated)"

            details.append(f"""
### Candidate {i}: {sol.title}
- **Votes**: {sol.votes} (Quality Signal - higher is better)
- **Estimated Complexity**: {complexity}
- **Models Used**: {", ".join(sol.models_used) if sol.models_used else "N/A"}
- **Feature Engineering**: {", ".join(sol.feature_engineering) if sol.feature_engineering else "N/A"}
- **Ensemble Approach**: {sol.ensemble_approach or "N/A"}
- **Key Strategies**: {", ".join(sol.strategies[:3]) if sol.strategies else "N/A"}

**Code Snippet** (use this as reference for your implementation):
```python
{code_snippet if code_snippet else "# No code available"}
```
""")

        return "\n".join(details)

    def _print_summary(self, plan: list[AblationComponent]) -> None:
        """Print plan summary."""
        print(f"\n= Ablation Plan Created: {len(plan)} components")
        print("-" * 60)

        for i, comp in enumerate(plan, 1):
            print(f"\n{i}. {comp.name} ({comp.component_type})")
            print(f"   Estimated Impact: {comp.estimated_impact:.1%}")
            if comp.code:
                print(f"   Code: {comp.code[:80]}...")

        print("\n" + "=" * 60)

    # ==================== Eureka: Multi-Candidate Evolutionary Planning ====================

    def _generate_multiple_plans(
        self,
        state: KaggleState,
        sota_analysis: dict[str, Any],
        n_candidates: int = 3,
    ) -> list[tuple[list[AblationComponent], str, float]]:
        """
        Eureka-style: Generate multiple candidate plans with different strategies.

        Args:
            state: Current workflow state
            sota_analysis: SOTA analysis results
            n_candidates: Number of candidate plans to generate

        Returns:
            List of (plan, strategy, fitness_score) tuples
        """
        print(f"\n   Eureka: Generating {n_candidates} candidate plans...")

        # Domain-aware strategy selection
        domain = state.get("domain_detected", "tabular")

        # Define domain groups for cleaner matching
        IMAGE_CLASSIFICATION_DOMAINS = {
            "image_classification",
            "image_regression",
            "computer_vision",
            "image",  # legacy
        }
        IMAGE_SEGMENTATION_DOMAINS = {
            "image_segmentation",
            "image_to_image",
            "object_detection",
        }
        NLP_DOMAINS = {
            "nlp",
            "text_classification",
            "text_regression",
            "seq_to_seq",
        }
        AUDIO_DOMAINS = {
            "audio_classification",
            "audio_regression",
        }

        if domain in IMAGE_CLASSIFICATION_DOMAINS:
            # IMAGE CLASSIFICATION strategies - use CNN backbones
            strategies = [
                {
                    "name": "conservative",
                    "prompt_modifier": "Use proven CNN architectures: EfficientNet-B0/B3, ResNet50. Focus on stable training with pretrained ImageNet weights.",
                    "model_preference": ["efficientnet_b0", "resnet50", "efficientnet_b3"],
                },
                {
                    "name": "aggressive",
                    "prompt_modifier": "Use SOTA architectures: ResNet200d, EfficientNet-B5/B7, ConvNeXt. Apply heavy augmentation (Cutmix, Mixup). Full fine-tuning.",
                    "model_preference": ["resnet200d", "efficientnet_b5", "convnext", "swin"],
                },
                {
                    "name": "balanced",
                    "prompt_modifier": "Use mid-size models: EfficientNet-B3/B4, ResNet101. Balance speed with accuracy. Use TTA for inference.",
                    "model_preference": ["efficientnet_b3", "efficientnet_b4", "resnet101"],
                },
            ]
        elif domain in IMAGE_SEGMENTATION_DOMAINS:
            # IMAGE SEGMENTATION / OBJECT DETECTION strategies - use encoder-decoder architectures
            strategies = [
                {
                    "name": "conservative",
                    "prompt_modifier": "Use proven segmentation architectures: U-Net with ResNet34 encoder, FPN. Focus on stable training with pretrained encoders.",
                    "model_preference": ["unet_resnet34", "fpn", "deeplabv3"],
                },
                {
                    "name": "aggressive",
                    "prompt_modifier": "Use SOTA segmentation: U-Net++ with EfficientNet-B5 encoder, HRNet, Mask R-CNN. Apply heavy augmentation.",
                    "model_preference": ["unet_plusplus", "hrnet", "mask_rcnn", "segformer"],
                },
                {
                    "name": "balanced",
                    "prompt_modifier": "Use mid-size segmentation models: U-Net with EfficientNet-B3 encoder. Balance speed with accuracy.",
                    "model_preference": ["unet_effb3", "deeplabv3_plus", "pan"],
                },
            ]
        elif domain in NLP_DOMAINS:
            # NLP domain strategies
            strategies = [
                {
                    "name": "conservative",
                    "prompt_modifier": "Use proven NLP models: DistilBERT, RoBERTa-base. Focus on stable training.",
                    "model_preference": ["distilbert", "roberta_base", "bert_base"],
                },
                {
                    "name": "aggressive",
                    "prompt_modifier": "Use large models: DeBERTa-v3, RoBERTa-large. Apply advanced techniques like MLM pretraining.",
                    "model_preference": ["deberta_v3", "roberta_large", "longformer"],
                },
                {
                    "name": "balanced",
                    "prompt_modifier": "Mix efficient models with strong performance. Use ensemble of BERT variants.",
                    "model_preference": ["roberta_base", "deberta", "albert"],
                },
            ]
        elif domain in AUDIO_DOMAINS:
            # AUDIO domain strategies
            strategies = [
                {
                    "name": "conservative",
                    "prompt_modifier": "Use proven audio models: mel-spectrogram + EfficientNet, simple CNN. Focus on stable preprocessing.",
                    "model_preference": ["efficientnet_audio", "resnet_audio", "simple_cnn"],
                },
                {
                    "name": "aggressive",
                    "prompt_modifier": "Use SOTA audio: AST (Audio Spectrogram Transformer), wav2vec2, PANN. Heavy augmentation (SpecAugment, mixup).",
                    "model_preference": ["ast", "wav2vec2", "pann", "whisper"],
                },
                {
                    "name": "balanced",
                    "prompt_modifier": "Use mid-size audio models: EfficientNet-B2 on mel-specs. Balance preprocessing with model complexity.",
                    "model_preference": ["efficientnet_b2_audio", "sed_model", "cnn_transformer"],
                },
            ]
        else:
            # TABULAR domain strategies (default for tabular, time_series, multi_modal, unknown)
            strategies = [
                {
                    "name": "conservative",
                    "prompt_modifier": "Focus on proven, reliable approaches. Use well-established models like XGBoost, LightGBM. Prioritize stability over novelty.",
                    "model_preference": ["xgboost", "lightgbm", "random_forest"],
                },
                {
                    "name": "aggressive",
                    "prompt_modifier": "Focus on innovative approaches. Prioritize novel feature engineering, creative ensembles, and cutting-edge techniques.",
                    "model_preference": ["catboost", "neural_network", "stacking"],
                },
                {
                    "name": "balanced",
                    "prompt_modifier": "Mix proven models with creative features. Balance stability with innovation.",
                    "model_preference": ["xgboost", "lightgbm", "catboost"],
                },
            ]

        print(f"   📊 Domain: {domain}, using domain-specific strategies")

        # Get current iteration to determine if we should use extended strategies
        current_iteration = state.get("current_iteration", 0)

        # After iteration 2, add extended strategies for more diversity
        if current_iteration >= 2:
            # Select domain-appropriate extended strategies
            if domain in IMAGE_CLASSIFICATION_DOMAINS or domain in IMAGE_SEGMENTATION_DOMAINS:
                extended = EXTENDED_STRATEGIES_CV
            elif domain in NLP_DOMAINS:
                extended = EXTENDED_STRATEGIES_NLP
            else:
                extended = EXTENDED_STRATEGIES_TABULAR

            # Add extended strategies to the base strategies
            extended_list = [
                extended.get("feature_engineering_heavy"),
                extended.get("neural_exploration"),
                extended.get("hyperparameter_variant"),
            ]
            strategies.extend([s for s in extended_list if s is not None])
            print(f"   🔄 Iteration {current_iteration}: Added extended strategies for diversity")

        # Dynamically adjust n_candidates based on iteration
        if current_iteration >= 3:
            n_candidates = min(5, len(strategies))  # More candidates in later iterations
        else:
            n_candidates = min(n_candidates, len(strategies))

        candidate_plans = []

        for i, strategy in enumerate(strategies[:n_candidates]):
            print(f"   - Generating {strategy['name']} plan...")

            # Generate plan with strategy-specific modifications
            plan = self._generate_plan_with_strategy(state, sota_analysis, strategy)

            # Apply hyperparameter mutation for variant strategies
            if strategy.get("inherit_from_best") and current_iteration >= 2:
                plan = self._mutate_plan_hyperparameters(plan, state)

            # Evaluate fitness
            fitness = self._evaluate_plan_fitness(plan, state)

            candidate_plans.append((plan, strategy["name"], fitness))
            print(f"     Fitness: {fitness:.3f}")

        # Sort by fitness (highest first)
        candidate_plans.sort(key=lambda x: x[2], reverse=True)

        return candidate_plans

    def _generate_plan_with_strategy(
        self,
        state: KaggleState,
        sota_analysis: dict[str, Any],
        strategy: dict[str, Any],
    ) -> list[AblationComponent]:
        """
        Generate a single plan with a specific strategy.

        Args:
            state: Current workflow state
            sota_analysis: SOTA analysis results
            strategy: Strategy configuration

        Returns:
            List of ablation components
        """
        competition_info = state["competition_info"]
        domain = state.get("domain_detected", "tabular")

        # Create strategy-modified prompt
        strategy_prompt = f"""
Strategy: {strategy["name"].upper()}
{strategy["prompt_modifier"]}

Preferred approaches: {", ".join(strategy["model_preference"])}
"""

        # Use fallback plan generation with strategy bias
        plan = self._create_fallback_plan(domain, sota_analysis, state=state)
        plan = self._coerce_components(plan)

        # Modify plan based on strategy
        if strategy["name"] == "conservative":
            # Filter to keep only well-established models
            plan = [
                c
                for c in plan
                if any(
                    m in c.name.lower()
                    for m in ["xgboost", "lightgbm", "random", "logistic", "baseline"]
                )
            ] or plan[:2]

        elif strategy["name"] == "aggressive":
            # Boost feature engineering and ensemble components
            for comp in plan:
                if comp.component_type in ["feature_engineering", "ensemble"]:
                    comp.estimated_impact = min(comp.estimated_impact * 1.3, 1.0)

        return plan

    def _mutate_plan_hyperparameters(
        self,
        plan: list[AblationComponent],
        state: KaggleState,
        mutation_rate: float = 0.3,
    ) -> list[AblationComponent]:
        """
        Apply hyperparameter mutations to plan components.

        Eureka-style: Introduce controlled randomness to explore hyperparameter space.

        Args:
            plan: Original plan components
            state: Current workflow state (for accessing best hyperparameters)
            mutation_rate: Probability of mutating each component

        Returns:
            Plan with mutated hyperparameters
        """
        iteration_memory = state.get("iteration_memory", [])

        # Get best hyperparameters from previous iterations
        best_hyperparams = {}
        if iteration_memory:
            for memory in iteration_memory:
                if hasattr(memory, "best_hyperparameters") and memory.best_hyperparameters:
                    best_hyperparams.update(memory.best_hyperparameters)

        mutated_plan = []
        for comp in plan:
            # Only mutate model components with some probability
            if comp.component_type == "model" and random.random() < mutation_rate:
                # Create a mutated version
                mutated_name = f"{comp.name}_hp_variant"

                # Define mutation suggestions for common hyperparameters
                mutation_hints = self._get_hyperparameter_mutations(comp.name)

                # Add mutation hint to code description
                mutation_note = f"\n# HYPERPARAMETER VARIANT: {mutation_hints}"

                mutated_comp = AblationComponent(
                    name=mutated_name,
                    component_type=comp.component_type,
                    description=f"{comp.description} (hyperparameter variant: {mutation_hints})",
                    code=comp.code,  # Code will be regenerated by Developer
                    estimated_impact=comp.estimated_impact * 0.95,  # Slight uncertainty penalty
                    dependencies=comp.dependencies,
                    ablatable=comp.ablatable,
                )
                mutated_plan.append(mutated_comp)
            else:
                mutated_plan.append(comp)

        return mutated_plan

    def _get_hyperparameter_mutations(self, model_name: str) -> str:
        """Get suggested hyperparameter mutations for a model type."""
        model_lower = model_name.lower()

        if "lightgbm" in model_lower or "lgb" in model_lower:
            mutations = [
                "learning_rate: [0.01, 0.03, 0.05]",
                "num_leaves: [31, 63, 127]",
                "max_depth: [5, 7, 9]",
                "reg_alpha: [0, 0.1, 0.5]",
            ]
            return random.choice(mutations)

        elif "xgboost" in model_lower or "xgb" in model_lower:
            mutations = [
                "learning_rate: [0.01, 0.05, 0.1]",
                "max_depth: [4, 6, 8]",
                "subsample: [0.7, 0.8, 0.9]",
                "colsample_bytree: [0.7, 0.8, 0.9]",
            ]
            return random.choice(mutations)

        elif "catboost" in model_lower:
            mutations = [
                "learning_rate: [0.01, 0.03, 0.1]",
                "depth: [4, 6, 8]",
                "l2_leaf_reg: [1, 3, 5]",
            ]
            return random.choice(mutations)

        elif "neural" in model_lower or "mlp" in model_lower or "tabnet" in model_lower:
            mutations = [
                "learning_rate: [1e-4, 1e-3, 1e-2]",
                "dropout: [0.1, 0.2, 0.3]",
                "hidden_dims: [128, 256, 512]",
            ]
            return random.choice(mutations)

        else:
            # Generic mutations
            return "try different hyperparameter values"

    def _evaluate_plan_fitness(
        self,
        plan: list[AblationComponent],
        state: KaggleState,
    ) -> float:
        """
        Eureka-style: Evaluate fitness of a plan based on history.

        Considers:
        - Past success/failure patterns from iteration_memory
        - Component type diversity
        - Estimated impact scores
        - Crossover guidance from meta-evaluator

        Args:
            plan: Candidate plan to evaluate
            state: Current workflow state

        Returns:
            Fitness score (0-1)
        """
        score = 0.0
        iteration_memory = state.get("iteration_memory", [])
        crossover_guidance = state.get("crossover_guidance", {})

        # 1. Historical success/failure (40% weight)
        historical_score = 0.0
        for comp in plan:
            for memory in iteration_memory:
                # Reward components similar to what worked
                if comp.component_type in memory.what_worked:
                    historical_score += 0.2
                # Penalize components similar to what failed
                if comp.component_type in memory.what_failed:
                    historical_score -= 0.1

        historical_score = max(0, min(historical_score, 1.0))
        score += 0.4 * historical_score

        # 2. Diversity bonus (20% weight)
        unique_types = len(set(c.component_type for c in plan))
        diversity_score = min(unique_types / 4.0, 1.0)  # Target: 4 different types
        score += 0.2 * diversity_score

        # 3. Estimated impact (25% weight)
        if plan:
            avg_impact = sum(c.estimated_impact for c in plan) / len(plan)
            score += 0.25 * avg_impact

        # 4. Crossover guidance alignment (15% weight)
        if crossover_guidance:
            preserve_components = crossover_guidance.get("preserve_components", [])
            avoid_components = crossover_guidance.get("avoid_components", [])

            alignment_score = 0.0
            for comp in plan:
                if comp.component_type in preserve_components:
                    alignment_score += 0.3
                if comp.component_type in avoid_components:
                    alignment_score -= 0.2

            alignment_score = max(0, min(alignment_score, 1.0))
            score += 0.15 * alignment_score

        return min(max(score, 0.0), 1.0)

    def _select_best_plan(
        self,
        candidate_plans: list[tuple[list[AblationComponent], str, float]],
    ) -> tuple[list[AblationComponent], str]:
        """
        Select the best plan from candidates.

        Args:
            candidate_plans: List of (plan, strategy, fitness) tuples

        Returns:
            Tuple of (best_plan, strategy_name)
        """
        if not candidate_plans:
            return [], "none"

        best_plan, strategy, fitness = candidate_plans[0]
        print(f"\n   Eureka: Selected '{strategy}' plan (fitness: {fitness:.3f})")

        return best_plan, strategy

    def generate_with_eureka(
        self,
        state: KaggleState,
        sota_analysis: dict[str, Any],
        n_candidates: int = 3,
    ) -> dict[str, Any]:
        """
        Eureka-style plan generation with multiple candidates.

        Args:
            state: Current workflow state
            sota_analysis: SOTA analysis results
            n_candidates: Number of candidates to generate

        Returns:
            State updates with plan and candidate info
        """
        print("\n   Eureka: Multi-candidate evolutionary planning...")

        # Generate multiple candidate plans
        candidate_plans = self._generate_multiple_plans(state, sota_analysis, n_candidates)

        # Select the best plan
        best_plan, strategy = self._select_best_plan(candidate_plans)

        # Store all candidates for potential crossover in next iteration
        from ..core.state import CandidatePlan

        stored_candidates = [
            CandidatePlan(
                components=plan,
                strategy=strat,
                fitness_score=fitness,
                generation=state.get("evolutionary_generation", 0) + 1,
            )
            for plan, strat, fitness in candidate_plans
        ]

        return {
            "ablation_plan": best_plan,
            "candidate_plans": stored_candidates,
            "current_plan_index": 0,
            "evolutionary_generation": state.get("evolutionary_generation", 0) + 1,
            "optimization_strategy": f"eureka_{strategy}",
        }


# ==================== LangGraph Node Function ====================


def planner_agent_node(state: KaggleState) -> dict[str, Any]:
    """
    LangGraph node function for the planner agent.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = PlannerAgent()
    return agent(state)
