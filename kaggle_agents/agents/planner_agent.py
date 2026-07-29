"""
Planner Agent with Ablation-Driven Optimization.

This agent implements the ablation planning strategy from Google ADK,
identifying high-impact components for systematic improvement.
"""

from __future__ import annotations

import json
import os
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
from .planner.domain_patterns import (
    extract_domain_specific_patterns,
    format_seq2seq_insights,
)
from .planner.fallback_plans.audio import create_audio_fallback_plan
from .planner.fallback_plans.seq2seq import create_seq2seq_fallback_plan
from .planner.fallback_plans.text import create_text_fallback_plan
from .planner.sota_analysis import (
    format_sota_details as format_external_sota_details,
)
from .planner.sota_analysis import (
    format_sota_solutions as format_external_sota_solutions,
)


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

    def _extract_domain_specific_patterns(
        self, sota_solutions: list[SOTASolution], domain: str
    ) -> dict[str, Any]:
        """Extract patterns using the maintained, data-driven implementation."""
        return extract_domain_specific_patterns(sota_solutions, domain)

    def _format_domain_insights(self, domain: str, domain_patterns: dict[str, Any]) -> str:
        """Format domain-specific insights for the planner prompt.

        Returns empty string for domains without specific insights.

        Args:
            domain: Detected domain
            domain_patterns: Patterns extracted from SOTA solutions

        Returns:
            Formatted string with domain insights (or empty string)
        """
        if domain == "seq_to_seq":
            return self._format_seq2seq_insights(domain_patterns)
        # Add more domains as needed:
        # elif domain == "tabular":
        #     return self._format_tabular_insights(domain_patterns)
        return ""

    def _format_seq2seq_insights(self, patterns: dict[str, Any]) -> str:
        """Format seq2seq insights using locally validated routing guidance."""
        return format_seq2seq_insights(patterns)

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

        # Extract domain-specific patterns and format insights
        domain_patterns = self._extract_domain_specific_patterns(sota_solutions, domain)
        domain_insights = self._format_domain_insights(domain, domain_patterns)

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
        run_mode = str(state.get("run_mode", "")).lower()
        fast_mode = bool(state.get("fast_mode"))
        default_max_components = 3 if run_mode == "mlebench" else 3 if fast_mode else 6
        max_components = int(
            state.get("max_components") or default_max_components
        )
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

        # Combine domain_guidance with domain_insights for DSPy path
        # (Appending avoids modifying the DSPy signature which would require retraining)
        enhanced_domain_guidance = domain_guidance + "\n\n" + domain_insights if domain_insights else domain_guidance

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
                        domain_guidance=enhanced_domain_guidance,
                        memory_summary=memory_summary,
                    )
                except TypeError:
                    # Backward-compatible fallback for older optimized signatures
                    result = self.planner_module(
                        competition_info=comp_info_str,
                        domain=domain,
                        sota_details=sota_details,
                        sota_summary=sota_summary,
                        domain_guidance=enhanced_domain_guidance,
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
                domain_insights=domain_insights,
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

Generate a plan using strategies supported by allowed evidence and local validation.

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
                if not plan_data:
                    print("  ⚠️ LLM generated no components, using fallback")
                    plan_data = self._create_fallback_plan(
                        domain, sota_analysis, curriculum_insights, state=state
                    )
            except Exception as e:
                print(f"  ⚠️ LLM plan generation failed: {e}, using fallback")
                plan_data = self._create_fallback_plan(
                    domain, sota_analysis, curriculum_insights, state=state
                )

        components = self._coerce_components(plan_data)

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

        # FIX: Plan diversity check - detect and avoid repeating same plan
        previous_plan_hashes = state.get("previous_plan_hashes", [])
        plan_hash = hash(tuple(sorted(c.name for c in components)))

        max_diversity_retries = 3
        retry_count = 0

        while plan_hash in previous_plan_hashes and retry_count < max_diversity_retries:
            # Select strategy BEFORE incrementing retry_count to avoid off-by-one
            diversity_strategies = [
                {"name": "neural_exploration", "focus": "deep_learning"},
                {"name": "feature_heavy", "focus": "feature_engineering"},
                {"name": "ensemble_focus", "focus": "ensemble"},
            ]
            strategy = diversity_strategies[retry_count % len(diversity_strategies)]

            retry_count += 1
            print(f"   [Planner] ⚠️ Plan already tried (attempt {retry_count}/{max_diversity_retries}) - trying {strategy['name']}...")

            # Try to generate a different plan using fallback with new strategy hint
            alternative_plan = self._create_diversified_fallback_plan(
                state, sota_analysis, strategy["focus"]
            )

            # Convert to components
            components = []
            for i, item in enumerate(alternative_plan):
                code = item.get("code_outline", item.get("description", ""))
                component = AblationComponent(
                    name=item.get("name", f"diverse_component_{i + 1}"),
                    component_type=item.get("component_type", "model"),
                    code=code,
                    estimated_impact=item.get("estimated_impact", 0.15),
                )
                components.append(component)

            plan_hash = hash(tuple(sorted(c.name for c in components)))

        # FIX: Force exploration if plan is still a duplicate after max retries
        if plan_hash in previous_plan_hashes and retry_count >= max_diversity_retries:
            # Count consecutive identical plans
            consecutive_duplicates = sum(1 for h in previous_plan_hashes[-3:] if h == plan_hash)

            if consecutive_duplicates >= 2:
                print("   [Planner] ⚠️ STAGNATION DETECTED: 2+ identical plans. Forcing novel exploration.")
                # Force completely different approach with components not used recently
                exploration_plan = self._create_exploration_plan(state, sota_analysis)
                if exploration_plan:
                    components = []
                    for i, item in enumerate(exploration_plan):
                        code = item.get("code_outline", item.get("description", ""))
                        component = AblationComponent(
                            name=item.get("name", f"explore_component_{i + 1}"),
                            component_type=item.get("component_type", "model"),
                            code=code,
                            estimated_impact=item.get("estimated_impact", 0.15),
                        )
                        components.append(component)
                    plan_hash = hash(tuple(sorted(c.name for c in components)))
                    print(f"   [Planner] ✓ Forced exploration plan: {[c.name for c in components]}")

        # Record this plan hash (keep last 10)
        previous_plan_hashes.append(plan_hash)
        state["previous_plan_hashes"] = previous_plan_hashes[-10:]

        if retry_count > 0:
            print(f"   [Planner] ✓ Found diverse plan after {retry_count} retries")

        return components

    def _extract_sota_recommendations(
        self,
        sota_analysis: dict[str, Any] | None,
    ) -> list[str]:
        """
        Extract model and technique recommendations from SOTA analysis.

        Parses SOTA search results to identify recommended approaches.
        Used by both exploration plans and fallback plans.

        Args:
            sota_analysis: SOTA analysis dict from search agent

        Returns:
            List of recommendation categories (e.g., ["catboost", "feature_engineering"])
        """
        if not sota_analysis:
            return []

        recommendations = []
        sota_str = str(sota_analysis).lower()

        # Model recommendations
        if "catboost" in sota_str:
            recommendations.append("catboost")
        if "lightgbm" in sota_str or "lgbm" in sota_str:
            recommendations.append("lightgbm")
        if "xgboost" in sota_str or "xgb" in sota_str:
            recommendations.append("xgboost")
        if "neural" in sota_str or "tabnet" in sota_str or "mlp" in sota_str or "nn" in sota_str:
            recommendations.append("neural_network")
        if "random forest" in sota_str or "randomforest" in sota_str:
            recommendations.append("random_forest")

        # Technique recommendations
        if "feature" in sota_str or "encoding" in sota_str:
            recommendations.append("feature_engineering")
        if "optuna" in sota_str or "hyperparameter" in sota_str or "tuning" in sota_str:
            recommendations.append("hyperparameter_tuning")
        if "ensemble" in sota_str or "stacking" in sota_str or "blend" in sota_str:
            recommendations.append("ensemble")

        return recommendations

    def _create_exploration_plan(
        self,
        state: KaggleState,
        sota_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Create plan with novel components not used in recent iterations.

        This is the nuclear option for breaking stagnation loops.
        Combines:
        1. SOTA guidance from search results
        2. Intensive training options (more epochs/folds)
        3. Novel component combinations

        Args:
            state: Current state
            sota_analysis: SOTA analysis

        Returns:
            Exploration plan with novel components
        """
        domain = str(state.get("domain_detected", "tabular")).lower()
        # Extract SOTA guidance if available
        sota_guidance = {}
        if state:
            refinement_guidance = state.get("refinement_guidance", {})
            sota_guidance = refinement_guidance.get("sota_guidance", {})

        # Check what components were used recently to avoid them
        used_components = set()

        # 1. Check development_history for executed components
        development_history = state.get("development_history", [])
        for entry in development_history[-6:]:  # Last 6 components
            if isinstance(entry, dict):
                used_components.add(entry.get("component_name", ""))

        # 2. Check current ablation_plan for planned components (may not be executed yet)
        current_plan = state.get("ablation_plan", [])
        if isinstance(current_plan, list):
            for component in current_plan:
                if isinstance(component, dict):
                    used_components.add(component.get("name", ""))
                elif hasattr(component, "name"):  # AblationComponent object
                    used_components.add(component.name)

        # 3. Check failed components to avoid retrying failures
        # NOTE: The state uses THREE different keys for failed components:
        #   - failed_components: list of component names (used by tests and some code paths)
        #   - failed_component_names: list of component name strings (state definition)
        #   - failure_analysis["failed_components"]: list of component dicts
        all_failed_names = []

        # Check state["failed_components"] (used by tests and some code paths)
        failed_components_list = state.get("failed_components", [])
        for comp in failed_components_list:
            if isinstance(comp, str):
                used_components.add(comp)
                all_failed_names.append(comp)
            elif isinstance(comp, dict):
                comp_name = comp.get("name") or comp.get("component_name", "")
                if comp_name:
                    used_components.add(comp_name)
                    all_failed_names.append(comp_name)

        # Check state["failed_component_names"] (state definition)
        failed_names = state.get("failed_component_names", [])
        for comp_name in failed_names:
            if isinstance(comp_name, str):
                used_components.add(comp_name)
                all_failed_names.append(comp_name)

        # Also check failure_analysis for additional failed component info
        failure_analysis = state.get("failure_analysis", {})
        failed_component_dicts = failure_analysis.get("failed_components", [])
        for comp in failed_component_dicts:
            if isinstance(comp, dict):
                comp_name = comp.get("name") or comp.get("component_name", "")
                if comp_name:
                    used_components.add(comp_name)
                    all_failed_names.append(comp_name)
            elif isinstance(comp, str):
                used_components.add(comp)
                all_failed_names.append(comp)

        if all_failed_names:
            # Deduplicate for logging
            unique_failed = list(dict.fromkeys(all_failed_names))
            print(f"   [EXPLORATION] Excluding failed components: {unique_failed}")

        print(f"   [EXPLORATION] Domain: {domain}, Recently used/planned/failed: {used_components}")

        # SOTA-guided exploration: extract recommendations using helper
        sota_recommended = self._extract_sota_recommendations(sota_analysis)
        if sota_recommended:
            print(f"   [EXPLORATION] SOTA recommends: {sota_recommended}")

        # For tabular: dynamic exploration based on SOTA + unused components
        if "tabular" in domain:
            plan = []

            # PRIORITY 1: SOTA-recommended approaches
            # Check exact component name to avoid re-adding used/failed components
            if "feature_engineering" in sota_recommended and "sota_feature_engineering" not in used_components:
                plan.append({
                    "name": "sota_feature_engineering",
                    "component_type": "feature_engineering",
                    "description": "Advanced feature engineering based on SOTA analysis. Includes target encoding, interactions, and aggregations.",
                    "estimated_impact": 0.18,
                    "code_outline": """SOTA-guided feature engineering:
# Install category_encoders if not available
try:
    from category_encoders import TargetEncoder
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'category_encoders', '-q'], timeout=60)
    from category_encoders import TargetEncoder

# Target encoding with smoothing
te = TargetEncoder(cols=cat_cols, smoothing=0.3)

# Feature interactions (identified by SOTA)
for c1, c2 in high_importance_pairs:
    X[f'{c1}_x_{c2}'] = X[c1] * X[c2]
    X[f'{c1}_div_{c2}'] = X[c1] / (X[c2] + 1e-8)

# Aggregation features by categorical groups
for cat in important_cats:
    for num in important_nums:
        X[f'{cat}_{num}_mean'] = X.groupby(cat)[num].transform('mean')
        X[f'{cat}_{num}_std'] = X.groupby(cat)[num].transform('std')""",
                })

            # Add CatBoost Optuna if SOTA recommends tuning AND component not already used/failed
            if ("hyperparameter_tuning" in sota_recommended or "catboost" in sota_recommended) and "catboost_optuna_intensive" not in used_components:
                plan.append({
                    "name": "catboost_optuna_intensive",
                    "component_type": "model",
                    "description": "CatBoost with intensive Optuna tuning (20 trials). SOTA recommends hyperparameter optimization.",
                    "estimated_impact": 0.21,
                    "code_outline": """INTENSIVE CatBoost tuning with timeout awareness:
import optuna
import time
import os

TIMEOUT = int(os.environ.get('KAGGLE_AGENTS_COMPONENT_TIMEOUT_S', 3600))
START = time.time()
TUNING_BUDGET = min(TIMEOUT * 0.4, 1200)  # Max 40% of timeout for tuning

def objective(trial):
    if time.time() - START > TUNING_BUDGET:
        raise optuna.TrialPruned()
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'iterations': trial.suggest_int('iterations', 500, 2000),
    }
    model = CatBoostClassifier(**params, verbose=0, early_stopping_rounds=100)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model.best_score_['validation']['Logloss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, timeout=TUNING_BUDGET)
best_params = study.best_params""",
                })

            # PRIORITY 2: Intensive training if score is plateauing
            # Check exact component name that will be added
            if "lgbm_intensive_7fold" not in used_components:
                plan.append({
                    "name": "lgbm_intensive_7fold",
                    "component_type": "model",
                    "description": "LightGBM with 7-fold CV and 3000 iterations. For breaking score plateau.",
                    "estimated_impact": 0.19,
                    "code_outline": """INTENSIVE 7-fold LightGBM with checkpoint saving:
import time
import os

TIMEOUT = int(os.environ.get('KAGGLE_AGENTS_COMPONENT_TIMEOUT_S', 3600))
START = time.time()
SAVE_BUFFER = 180

# Adaptive fold count based on remaining time
remaining = TIMEOUT - (time.time() - START)
n_folds = 7 if remaining > 3000 else 5 if remaining > 1800 else 3

params = {
    'n_estimators': 3000,
    'learning_rate': 0.015,
    'num_leaves': 128,
    'min_child_samples': 15,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'early_stopping_rounds': 200,
}

# Train with per-fold checkpoints
for fold in range(n_folds):
    remaining = TIMEOUT - (time.time() - START) - SAVE_BUFFER
    if remaining < 400:
        print(f"[INTENSIVE] Early stop at fold {fold}")
        break
    # ... train fold ...
    np.save(f'models/oof_lgbm_intensive_fold{fold}.npy', oof_preds)
    np.save(f'models/test_lgbm_intensive_fold{fold}.npy', test_preds)""",
                })

            # PRIORITY 3: Weighted ensemble if not tried
            # Check exact component name to avoid re-adding used/failed components
            if len(plan) < 3 and "weighted_rank_blend" not in used_components:
                plan.append({
                    "name": "weighted_rank_blend",
                    "component_type": "ensemble",
                    "description": "Rank-based weighted blending optimized with scipy.minimize.",
                    "estimated_impact": 0.12,
                    "code_outline": """Rank-based weighted ensemble:
from scipy.optimize import minimize
from scipy.stats import rankdata

# Convert predictions to ranks
def to_ranks(preds):
    return rankdata(preds) / len(preds)

# Load all OOF predictions
oof_files = list(Path('models').glob('oof_*.npy'))
oofs = [np.load(f) for f in oof_files]
oofs_ranked = [to_ranks(oof) for oof in oofs]

# Optimize weights
def loss(weights):
    weights = np.abs(weights) / np.sum(np.abs(weights))
    blend = np.average(oofs_ranked, axis=0, weights=weights)
    return -roc_auc_score(y_true, blend)

result = minimize(loss, x0=np.ones(len(oofs))/len(oofs), method='Nelder-Mead')
best_weights = np.abs(result.x) / np.sum(np.abs(result.x))""",
                })

            return plan[:3]  # Return top 3 components

        # For image: try different architectures with TTA and intensive training
        if "image" in domain:
            image_candidates = [
                {
                    "name": "efficientnet_b2_tta_intensive",
                    "component_type": "model",
                    "description": "EfficientNet-B2 with TTA and more epochs. Timeout-aware training.",
                    "estimated_impact": 0.19,
                    "code_outline": """EfficientNet-B2 with timeout-aware intensive training:
import time
import os

TIMEOUT = int(os.environ.get('KAGGLE_AGENTS_COMPONENT_TIMEOUT_S', 3600))
START = time.time()

# Adaptive epochs based on remaining time
remaining = TIMEOUT - (time.time() - START)
max_epochs = 15 if remaining > 3000 else 10 if remaining > 1800 else 5

model = timm.create_model('efficientnet_b2', pretrained=True, num_classes=num_classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

for epoch in range(max_epochs):
    remaining = TIMEOUT - (time.time() - START) - 120
    if remaining < 300:
        print(f"[TIMEOUT] Stopping at epoch {epoch}")
        break
    # ... train epoch ...
    torch.save(model.state_dict(), f'models/effnet_b2_epoch{epoch}.pth')

# TTA at inference
tta_preds = []
for transform in [identity, hflip, vflip, rotate90]:
    preds = model(transform(test_images))
    tta_preds.append(preds)
final_preds = np.mean(tta_preds, axis=0)""",
                },
                {
                    "name": "convnext_tiny_mixup",
                    "component_type": "model",
                    "description": "ConvNeXt-Tiny with MixUp augmentation. Modern architecture.",
                    "estimated_impact": 0.18,
                    "code_outline": "timm.create_model('convnext_tiny'), MixUp(alpha=0.4), CutMix(alpha=1.0)",
                },
                {
                    "name": "geometric_mean_ensemble",
                    "component_type": "ensemble",
                    "description": "Geometric mean of predictions. Better for log-loss metrics.",
                    "estimated_impact": 0.10,
                    "code_outline": "np.exp(np.mean(np.log(predictions + 1e-15), axis=0))",
                },
            ]
            # Filter out used/failed components
            return [c for c in image_candidates if c["name"] not in used_components][:3]

        # Default fallback with intensive options
        default_candidates = [
            {
                "name": "histgradient_intensive",
                "component_type": "model",
                "description": "HistGradientBoosting with max_iter=1000. Very fast sklearn implementation.",
                "estimated_impact": 0.16,
                "code_outline": "HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.1)",
            },
            {
                "name": "gradient_boosting_sklearn",
                "component_type": "model",
                "description": "Scikit-learn GradientBoosting with n_estimators=500.",
                "estimated_impact": 0.14,
                "code_outline": "GradientBoostingClassifier(n_estimators=500, max_depth=6, subsample=0.8)",
            },
            {
                "name": "simple_average_ensemble",
                "component_type": "ensemble",
                "description": "Simple average of all model predictions.",
                "estimated_impact": 0.08,
                "code_outline": "np.mean(all_predictions, axis=0)",
            },
        ]
        # Filter out used/failed components
        return [c for c in default_candidates if c["name"] not in used_components][:3]

    def _create_diversified_fallback_plan(
        self,
        state: KaggleState,
        sota_analysis: dict[str, Any],
        focus: str,
    ) -> list[dict[str, Any]]:
        """
        Create a diversified fallback plan with a specific focus.

        Args:
            state: Current state
            sota_analysis: SOTA analysis
            focus: Focus area ('deep_learning', 'feature_engineering', 'ensemble')

        Returns:
            Diversified plan as list of dicts
        """
        domain = str(state.get("domain_detected", "tabular")).lower()

        if focus == "deep_learning":
            return [
                {
                    "name": "nn_tabular",
                    "component_type": "model",
                    "description": "Neural network for tabular data (TabNet or MLP)",
                    "estimated_impact": 0.18,
                    "rationale": "Deep learning alternative to tree models",
                    "code_outline": "TabNet/MLP with entity embeddings, batch norm, dropout",
                },
                {
                    "name": "gradient_blend",
                    "component_type": "ensemble",
                    "description": "Gradient-based blending of NN and tree models",
                    "estimated_impact": 0.12,
                    "rationale": "Combine NN and tree strengths",
                    "code_outline": "Weighted average with learned weights via gradient descent",
                },
            ]
        if focus == "feature_engineering":
            return [
                {
                    "name": "target_encoding_cv",
                    "component_type": "feature_engineering",
                    "description": "Target encoding with proper CV to avoid leakage",
                    "estimated_impact": 0.15,
                    "rationale": "Powerful encoding for categorical features",
                    "code_outline": "category_encoders.TargetEncoder with cv=5 folds",
                },
                {
                    "name": "feature_selection",
                    "component_type": "feature_engineering",
                    "description": "Feature selection using importance + RFE",
                    "estimated_impact": 0.10,
                    "rationale": "Remove noise features",
                    "code_outline": "RFECV or SelectFromModel with LightGBM importances",
                },
                {
                    "name": "lightgbm_tuned",
                    "component_type": "model",
                    "description": "LightGBM with Optuna hyperparameter tuning",
                    "estimated_impact": 0.20,
                    "rationale": "Better hyperparameters",
                    "code_outline": "Optuna study with n_trials=50 for LGBM params",
                },
            ]
        # ensemble focus
        return [
            {
                "name": "stacking_meta",
                "component_type": "ensemble",
                "description": "Stacking ensemble with ridge meta-learner",
                "estimated_impact": 0.15,
                "rationale": "Combine diverse model predictions",
                "code_outline": "StackingClassifier/Regressor with Ridge meta",
            },
            {
                "name": "voting_diverse",
                "component_type": "ensemble",
                "description": "Voting ensemble with diverse base models",
                "estimated_impact": 0.10,
                "rationale": "Simple but effective ensemble",
                "code_outline": "VotingClassifier with LGBM, XGB, CatBoost",
            },
        ]

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
        # Compatibility path: use only scores independently recomputed from
        # canonical OOF artifacts. Generated stdout and declared impact are not
        # selection evidence.
        trusted_scores = state.get("trusted_component_scores", {})
        if not isinstance(trusted_scores, dict):
            trusted_scores = {}
        arms = []
        for idx, comp in enumerate(previous_plan):
            raw_score = trusted_scores.get(comp.name)
            try:
                reward = float(raw_score)
            except (TypeError, ValueError):
                reward = None
            success = (
                idx < len(dev_results)
                and dev_results[idx].success
                and reward is not None
            )
            arms.append(
                {
                    "component": comp,
                    "reward": reward,
                    "success": success,
                }
            )

        # Exploit: keep top-2 successful arms that ACTUALLY IMPROVED score
        # FIX: Check for actual improvement, not just "didn't crash"
        from ..core.config import is_metric_minimization

        previous_best = state.get("best_score", 0.0) or 0.0
        competition_info = state.get("competition_info")
        metric_name = ""
        if competition_info:
            metric_name = getattr(competition_info, "evaluation_metric", "") or ""
        is_minimize = is_metric_minimization(metric_name) if metric_name else True

        def actually_improved(arm: dict) -> bool:
            """Check if arm actually improved over previous best score."""
            if not arm["success"]:
                return False
            reward = arm["reward"]
            if reward is None or reward == 0.0:
                return False
            # For first iteration or no previous score, accept any result
            if previous_best == 0.0:
                return True
            if is_minimize:
                return reward < previous_best  # Lower is better
            return reward > previous_best      # Higher is better

        # First try to find arms that actually improved
        improved_arms = [a for a in arms if actually_improved(a)]

        if improved_arms:
            # Use arms that improved
            improved_arms.sort(key=lambda a: a["reward"], reverse=not is_minimize)
            keep = improved_arms[:2]
            print(f"   [Planner] Keeping {len(keep)} arm(s) that improved score")
        else:
            # No arms improved - fall back to successful arms but log warning
            successful_arms = [a for a in arms if a["success"]]
            if successful_arms:
                successful_arms.sort(key=lambda a: a["reward"], reverse=not is_minimize)
                keep = successful_arms[:1]  # Keep only 1 to encourage exploration
                print("   [Planner] ⚠️ No components improved score - keeping only 1 for stability")
            else:
                keep = []
                print("   [Planner] ⚠️ No successful components - forcing full exploration")

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
                    "rationale": (
                        "Leakage-safe feature candidate retained only when its ablation "
                        "improves held-out performance."
                    ),
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
                    "description": "TF-IDF (1-3 ngrams) + LogisticRegression on the injected canonical folds",
                    "estimated_impact": 0.22,
                    "rationale": "Strong NLP baseline - fast and interpretable",
                    "code_outline": "Fit TF-IDF + LogisticRegression independently inside every injected canonical fold",
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
                    "code_outline": "Fit TF-IDF + calibrated LinearSVC wholly inside each injected canonical training partition",
                }
            )

        elif domain in IMAGE_DOMAINS:
            # Image-specific fallback models
            if model_count < 2:
                plan.append(
                {
                    "name": "efficientnet_b0_baseline",
                    "component_type": "model",
                    "description": "EfficientNet-B0 pretrained, fine-tuned on the injected canonical folds",
                    "estimated_impact": 0.22,
                    "rationale": "Strong pretrained CNN baseline",
                    "code_outline": "Train timm EfficientNet-B0 independently on every injected canonical fold",
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
                    "description": "LightGBM with fold-local OHE and early stopping on the injected canonical folds",
                    "estimated_impact": 0.20,
                    "rationale": "High-ROI baseline model",
                    "code_outline": "Fit the task-appropriate LightGBM pipeline independently inside every injected canonical fold with budget-derived capacity",
                }
            )
                model_count += 1

            if model_count < 2:
                plan.append(
                {
                    "name": "xgboost_fast_cv",
                    "component_type": "model",
                    "description": "XGBoost with fold-local OHE on the injected canonical folds",
                    "estimated_impact": 0.18,
                    "rationale": "Adds diversity for ensemble",
                    "code_outline": "Fit the task-appropriate XGBoost pipeline independently inside every injected canonical fold with budget-derived capacity",
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
        from .planner.validation import validate_plan

        return validate_plan(
            plan,
            state=state,
            coerce_components_fn=self._coerce_components,
            is_image_competition_without_features_fn=(
                self._is_image_competition_without_features
            ),
        )

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
            target_score="Improve the held-out validation metric",
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

        Multi-modal datasets can have:
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
                    # Count features from explicit public-data roles instead of
                    # target-name recipes.
                    canonical_contract = state.get("canonical_contract") or {}
                    contract_targets = canonical_contract.get("target_cols") or []
                    exclude_cols = {
                        str(value).lower()
                        for value in (
                            state.get("target_col"),
                            canonical_contract.get("target_col"),
                            canonical_contract.get("id_col"),
                            *contract_targets,
                        )
                        if isinstance(value, str) and value
                    }
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
                    "Compare the hybrid model with a tabular-only baseline on identical "
                    "folds before selecting either path."
                ),
                "priority_models": [
                    "LightGBM with all tabular features",
                    "XGBoost with all tabular features",
                    "Hybrid CNN+Tabular candidate",
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

        timeout_cap = (state or {}).get("timeout_per_component")
        if isinstance(timeout_cap, str):
            try:
                timeout_cap = int(timeout_cap)
            except ValueError:
                timeout_cap = None

        # Speed-first is an explicit runtime/budget decision.
        fast_mode = (
            bool((state or {}).get("fast_mode"))
            or (isinstance(timeout_cap, int) and timeout_cap <= 1200)
        )

        # Calculate stagnation_iteration from previous plan hashes
        # This is used to rotate model selection and break stagnation loops
        stagnation_iteration = 0
        if state:
            previous_plan_hashes = state.get("previous_plan_hashes", [])
            stagnation_iteration = len(previous_plan_hashes)
            if stagnation_iteration > 0:
                print(f"  [DEBUG] Stagnation iteration: {stagnation_iteration} (previous plans: {len(previous_plan_hashes)})")

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
            return self._create_text_fallback_plan(
                domain,
                sota_analysis,
            )
        if domain in AUDIO_DOMAINS or domain.startswith("audio_"):
            return self._create_audio_fallback_plan(domain, sota_analysis)
        # Tabular (existing logic)
        return self._create_tabular_fallback_plan(
            domain,
            sota_analysis,
            curriculum_insights,
            fast_mode=fast_mode,
            state=state,
            stagnation_iteration=stagnation_iteration,
        )

    def _create_tabular_fallback_plan(
        self,
        domain: str,
        sota_analysis: dict[str, Any],
        curriculum_insights: str = "",
        *,
        fast_mode: bool = False,
        state: KaggleState | None = None,
        stagnation_iteration: int = 0,
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
            stagnation_iteration: Number of times fallback was used (rotates model selection)

        Returns:
            List of component dictionaries (5 components: 1 FE + 4 models + 1 ensemble)
        """
        # Get failed components to avoid repeating them
        failed_names = set()
        if state:
            failed_names = set(state.get("failed_component_names", []))
            if failed_names:
                print(f"   📋 Filtering out previously failed components: {failed_names}")

        # Extract SOTA guidance from state for informed rotation
        sota_guidance = None
        sota_recommended_models = []
        if state:
            refinement_guidance = state.get("refinement_guidance", {})
            sota_guidance = refinement_guidance.get("sota_guidance", {})
            if sota_guidance:
                print(f"   📊 SOTA guidance available: {list(sota_guidance.keys())}")
                # Use helper method to extract recommendations
                sota_recommended_models = self._extract_sota_recommendations(sota_guidance)
                if sota_recommended_models:
                    print(f"   📊 SOTA recommends: {sota_recommended_models}")

        # STAGNATION-AWARE ROTATION: Different model combinations per iteration
        # This breaks the loop of repeating the same 3 components
        if fast_mode:
            print(f"   🔄 Stagnation iteration: {stagnation_iteration}")

            # Define model rotation sets - each iteration uses different combination
            model_rotations = [
                # Iteration 0: Default (LGBM + XGB)
                ["lightgbm_fast_cv", "xgboost_fast_cv"],
                # Iteration 1: CatBoost + LGBM tuned
                ["catboost_fast_cv", "lightgbm_tuned_cv"],
                # Iteration 2: Neural Network + Random Forest (different model families)
                ["neural_network_mlp", "random_forest_fast"],
                # Iteration 3: evaluate feature engineering with CatBoost.
                ["target_encoding_fe", "catboost_fast_cv"],
                # Iteration 4: Intensive training - more folds/epochs
                ["lightgbm_intensive", "catboost_fast_cv"],
            ]

            # SOTA-guided rotation override: if SOTA recommends specific approach, prioritize it
            if "catboost" in sota_recommended_models and stagnation_iteration > 0:
                print("   📊 SOTA override: prioritizing CatBoost rotation")
                model_rotations[stagnation_iteration % len(model_rotations)] = ["catboost_fast_cv", "lightgbm_tuned_cv"]
            if "neural_network" in sota_recommended_models and stagnation_iteration > 1:
                print("   📊 SOTA override: prioritizing Neural Network rotation")
                model_rotations[stagnation_iteration % len(model_rotations)] = ["neural_network_mlp", "catboost_fast_cv"]
            if "feature_engineering" in sota_recommended_models and stagnation_iteration > 0:
                print("   📊 SOTA override: prioritizing Feature Engineering rotation")
                model_rotations[stagnation_iteration % len(model_rotations)] = ["target_encoding_fe", "lightgbm_fast_cv"]

            # Component definitions with MANDATORY StandardScaler for MLP
            component_defs = {
                "lightgbm_fast_cv": {
                    "name": "lightgbm_fast_cv",
                    "component_type": "model",
                    "description": "LightGBM baseline tuned for speed (no Optuna). Use fewer estimators + early stopping/callbacks.",
                    "estimated_impact": 0.18,
                    "rationale": "High ROI baseline for tabular tasks.",
                    "code_outline": "LGBMClassifier/Regressor with sane defaults, 3-fold CV, save OOF/test preds",
                },
                "xgboost_fast_cv": {
                    "name": "xgboost_fast_cv",
                    "component_type": "model",
                    "description": "XGBoost baseline tuned for speed (no Optuna). Use hist/gpu_hist where available.",
                    "estimated_impact": 0.16,
                    "rationale": "Provides diversity vs LightGBM with similar compute budget.",
                    "code_outline": "XGBClassifier/Regressor with fixed params, 3-fold CV, save OOF/test preds",
                },
                "catboost_fast_cv": {
                    "name": "catboost_fast_cv",
                    "component_type": "model",
                    "description": "CatBoost baseline tuned for speed (no Optuna). Handles categoricals natively.",
                    "estimated_impact": 0.17,
                    "rationale": "Different regularization than XGBoost/LightGBM; handles categoricals well.",
                    "code_outline": "CatBoostClassifier/Regressor with sane defaults, 3-fold CV, save OOF/test preds",
                },
                "lightgbm_tuned_cv": {
                    "name": "lightgbm_tuned_cv",
                    "component_type": "model",
                    "description": "LightGBM with light Optuna tuning (5 trials). Better than defaults, still fast.",
                    "estimated_impact": 0.19,
                    "rationale": (
                        "A small search tests whether the default configuration is locally "
                        "suboptimal; keep the result only when held-out validation improves."
                    ),
                    "code_outline": "LGBMClassifier/Regressor with Optuna (5 trials), tune learning_rate/num_leaves, 3-fold CV",
                },
                "neural_network_mlp": {
                    "name": "neural_network_mlp",
                    "component_type": "model",
                    "description": "MLP Neural Network with MANDATORY StandardScaler. Different pattern capture than trees.",
                    "estimated_impact": 0.14,
                    "rationale": "Neural Networks capture non-linear patterns differently, adds diversity.",
                    "code_outline": """MANDATORY: Use StandardScaler for ALL numeric features (without this, AUC will be ~0.50!).
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier  # or MLPRegressor

# Per-fold scaling (fit on train, transform val and test)
scaler = StandardScaler()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Inside CV loop:
X_train_fold_scaled = scaler.fit_transform(X_train_fold[numeric_cols])
X_val_fold_scaled = scaler.transform(X_val_fold[numeric_cols])
X_test_scaled = scaler.transform(X_test[numeric_cols])

model = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, validation_fraction=0.1, max_iter=500, random_state=42)
model.fit(X_train_fold_scaled, y_train_fold)
oof_preds[val_idx] = model.predict_proba(X_val_fold_scaled)[:, 1]  # Binary
test_preds += model.predict_proba(X_test_scaled)[:, 1] / n_folds""",
                },
                "random_forest_fast": {
                    "name": "random_forest_fast",
                    "component_type": "model",
                    "description": "Random Forest baseline with limited trees (n_estimators=200) for speed.",
                    "estimated_impact": 0.13,
                    "rationale": (
                        "A dependency-light tree ensemble candidate; it must still pass "
                        "held-out and submission validation."
                    ),
                    "code_outline": "RandomForestClassifier/Regressor with n_estimators=200, 3-fold CV, save OOF/test preds",
                },
                "stacking_ensemble": {
                    "name": "stacking_ensemble",
                    "component_type": "ensemble",
                    "description": "Stack OOF predictions from available models with LogisticRegression/Ridge meta-learner.",
                    "estimated_impact": 0.10,
                    "rationale": (
                        "A cheap ensemble is worth evaluating when base-model OOF errors "
                        "differ; retain it only if held-out performance improves."
                    ),
                    "code_outline": "Load models/oof_*.npy + models/test_*.npy, fit meta-model on OOF, predict test, write submission",
                },
                "simple_ridge_baseline": {
                    "name": "simple_ridge_baseline",
                    "component_type": "model",
                    "description": "Simple Ridge regression baseline with StandardScaler.",
                    "estimated_impact": 0.08,
                    "rationale": (
                        "Low-complexity fallback when optional libraries are unavailable; "
                        "it must still pass validation and submission checks."
                    ),
                    "code_outline": "StandardScaler + Ridge, 3-fold CV, save OOF/test preds",
                },
                "target_encoding_fe": {
                    "name": "target_encoding_fe",
                    "component_type": "feature_engineering",
                    "description": "Target encoding with K-fold CV to prevent leakage. Creates powerful features from categoricals.",
                    "estimated_impact": 0.16,
                    "rationale": (
                        "Leakage-safe target encoding can represent categorical signal; "
                        "compare it with the existing encoding on identical folds."
                    ),
                    "code_outline": """Target encoding with CV to prevent leakage:
# Install category_encoders if not available
try:
    from category_encoders import TargetEncoder
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'category_encoders', '-q'], timeout=60)
    from category_encoders import TargetEncoder

from sklearn.model_selection import KFold

# Initialize with CV to prevent leakage
te = TargetEncoder(cols=categorical_cols, smoothing=0.3)

# Encode within CV folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_encoded = X.copy()
for train_idx, val_idx in kf.split(X):
    te.fit(X.iloc[train_idx], y.iloc[train_idx])
    X_encoded.iloc[val_idx] = te.transform(X.iloc[val_idx])

# Fit final encoder for test data
te.fit(X, y)
X_test_encoded = te.transform(X_test)

# Save encoded features for downstream models
np.save(MODELS_DIR / 'X_train_encoded.npy', X_encoded.values)
np.save(MODELS_DIR / 'X_test_encoded.npy', X_test_encoded.values)""",
                },
                "lightgbm_intensive": {
                    "name": "lightgbm_intensive",
                    "component_type": "model",
                    "description": "LightGBM with MORE folds (5-7) and iterations for better score. Uses timeout-aware training.",
                    "estimated_impact": 0.20,
                    "rationale": (
                        "Additional folds and iterations may reduce estimation variance; "
                        "use them only when the time budget and validation results justify it."
                    ),
                    "code_outline": """INTENSIVE training with timeout awareness:
import time
import os

TIMEOUT = int(os.environ.get('KAGGLE_AGENTS_COMPONENT_TIMEOUT_S', 3600))
START_TIME = time.time()
SAVE_BUFFER = 180  # Reserve 3 min for saving

# Increase folds for better CV estimate (only if time permits)
remaining = TIMEOUT - (time.time() - START_TIME)
n_folds = 7 if remaining > 2400 else 5 if remaining > 1200 else 3
print(f"[INTENSIVE] Using {n_folds} folds (remaining time: {remaining:.0f}s)")

# More iterations with early stopping
params = {
    'n_estimators': 2000,  # More iterations
    'learning_rate': 0.02,  # Lower LR for better convergence
    'num_leaves': 64,
    'min_child_samples': 20,
    'early_stopping_rounds': 150,  # More patience
    'verbose': -1,
}

# Per-fold timeout check
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    fold_start = time.time()
    remaining = TIMEOUT - (time.time() - START_TIME) - SAVE_BUFFER
    if remaining < 300:  # Need at least 5 min per fold
        print(f"[INTENSIVE] Stopping at fold {fold} (not enough time)")
        break

    model = LGBMClassifier(**params)
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])
    oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / n_folds

    # Save checkpoint after each fold
    np.save(MODELS_DIR / f'oof_checkpoint_fold{fold}.npy', oof_preds)
    np.save(MODELS_DIR / f'test_checkpoint_fold{fold}.npy', test_preds)
    print(f"[INTENSIVE] Fold {fold} done in {time.time()-fold_start:.0f}s")""",
                },
            }

            # Select rotation based on stagnation_iteration (with wrap-around)
            rotation_idx = stagnation_iteration % len(model_rotations)
            selected_model_names = model_rotations[rotation_idx]
            print(f"   📋 Selected rotation {rotation_idx}: {selected_model_names}")

            # Build plan from selected models, filtering out failed ones
            models = []
            for name in selected_model_names:
                if name not in failed_names and name in component_defs:
                    models.append(component_defs[name])
                else:
                    print(f"      ⚠️ Skipping {name} (failed or unavailable)")

            # Ensure the plan remains executable; diversity is optional.
            if not models:
                fallback_order = ["catboost_fast_cv", "lightgbm_fast_cv", "xgboost_fast_cv", "random_forest_fast", "simple_ridge_baseline"]
                for fallback_name in fallback_order:
                    if fallback_name not in failed_names and fallback_name not in [m["name"] for m in models]:
                        models.append(component_defs[fallback_name])
                        break

            # Ensembling requires at least two validated prediction sources.
            ensemble = [component_defs["stacking_ensemble"]] if len(models) >= 2 else []
            final_plan = models[:2] + ensemble
            print(f"   ✅ Final plan: {[c['name'] for c in final_plan]}")
            return final_plan

        plan = []

        # ALWAYS add feature engineering first (high impact)
        plan.append(
            {
                "name": "advanced_feature_engineering",
                "component_type": "feature_engineering",
                "description": "Create polynomial features (degree 2), feature interactions (ratio, diff, product), statistical transformations (log, sqrt), and target encoding for categorical features",
                "estimated_impact": 0.15,
                "rationale": (
                    "Feature engineering is a candidate source of tabular signal; retain "
                    "transformations only when leakage-safe validation improves."
                ),
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
                    "rationale": (
                        "LightGBM is an efficient nonlinear baseline. Compare tuned parameters "
                        "against the untuned baseline using the same validation folds."
                    ),
                    "code_outline": "LGBMRegressor/Classifier with OptunaSearchCV, 5-fold CV, early_stopping_rounds=100",
                },
                {
                    "name": "xgboost_optuna_tuned",
                    "component_type": "model",
                    "description": "XGBoost with Optuna hyperparameter optimization: 15 trials, tuning max_depth, learning_rate, subsample, colsample_bytree",
                    "estimated_impact": 0.20,
                    "rationale": (
                        "XGBoost offers a different regularization profile. Retain tuned "
                        "parameters only when nested validation improves over its baseline."
                    ),
                    "code_outline": "XGBRegressor/Classifier with OptunaSearchCV, 5-fold CV, early_stopping_rounds=50",
                },
                {
                    "name": "catboost_optuna_tuned",
                    "component_type": "model",
                    "description": "CatBoost with Optuna hyperparameter optimization: 15 trials, tuning depth, learning_rate, l2_leaf_reg",
                    "estimated_impact": 0.19,
                    "rationale": (
                        "CatBoost provides native categorical handling. Select depth and "
                        "regularization using the same leakage-safe folds as other models."
                    ),
                    "code_outline": "CatBoostRegressor/Classifier with OptunaSearchCV, cat_features parameter, 5-fold CV",
                },
                {
                    "name": "neural_network_mlp",
                    "component_type": "model",
                    "description": "Simple MLP Neural Network using Scikit-Learn or PyTorch (if available). Standard scaling is CRITICAL.",
                    "estimated_impact": 0.15,
                    "rationale": (
                        "An MLP is a complementary candidate. Include it in an ensemble only "
                        "when its out-of-fold errors add validated diversity."
                    ),
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
                "rationale": (
                    "Stacking can exploit complementary out-of-fold errors. Keep the "
                    "meta-model only when it improves over the best base model."
                ),
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
                    "description": "Simple Ridge regression baseline with a minimal preprocessing pipeline.",
                    "estimated_impact": 0.08,
                    "rationale": (
                        "Low-complexity fallback when other candidates fail; it must still "
                        "pass validation and submission checks."
                    ),
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
                    "rationale": (
                        "A frozen backbone bounds training cost. Use the time budget and "
                        "validation curve to choose the number of head-training epochs."
                    ),
                    "code_outline": "efficientnet_b0(pretrained=True), freeze all backbone layers, train head only, 2 epochs, 2-fold CV, save best checkpoint, implement _check_deadline() pattern",
                },
                {
                    "name": "tta_inference_only",
                    "component_type": "ensemble",
                    "description": "Test-Time Augmentation ONLY (no additional training). Load the single trained model and apply 5 simple transforms (original, hflip, vflip, rotate90, rotate180), average predictions, save aligned evidence, and call write_submission(test_preds).",
                    "estimated_impact": 0.05,
                    "rationale": (
                        "TTA adds inference cost but no training. Retain it only when an OOF "
                        "simulation improves the declared metric."
                    ),
                    "code_outline": "Load models/best_model.* (auto-detect extension), for each test image: apply transforms, average predictions, clip to [0,1], call save_component_artifacts(oof_preds, test_preds), then write_submission(test_preds)",
                },
            ]

        # NORMAL MODE: 3 components (2 models + TTA ensemble)
        return [
            {
                "name": f"efficientnet_b0_{task}",
                "component_type": "model",
                "description": f"EfficientNet-B0 pre-trained fine-tuned for {task}. PyTorch DataLoader with ImageFolder or custom Dataset. Data augmentation (rotation, flip, color jitter). Use transfer learning from ImageNet weights.",
                "estimated_impact": 0.28,
                "rationale": (
                    "An efficient pretrained backbone is a practical transfer-learning "
                    "candidate. Select augmentation and fine-tuning depth from validation."
                ),
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
                "rationale": (
                    "TTA and model blending can reduce variance, but both must be evaluated "
                    "on held-out predictions and disabled when they do not improve the metric."
                ),
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
        Infer the submission schema and row mapping from sample_submission.csv; never
        synthesize identifiers or assume a coordinate encoding or index base.

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
save_component_artifacts(
    oof_images,
    test_images,
    train_ids=CANONICAL_TRAIN_IDS,
    test_ids=CANONICAL_TEST_IDS,
)
write_submission(None)
```
The injected writer maps the saved packed artifact to observed template IDs,
proves exact pixel coverage and coordinate base, and streams the CSV.
Never load the full template or write it manually.""",
                    "estimated_impact": 0.35,
                    "rationale": "Simple autoencoder is fast to train and provides baseline for denoising. Pixel-level output is critical for correct submission format.",
                    "code_outline": "Conv2d encoder, ConvTranspose2d decoder, MSE loss, output same size as input, align keyed predictions to sample_submission",
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
Read sample_submission.csv as the schema and ordering contract. Map model outputs
to identifiers through supplied test metadata, validate exact key-set equality,
then reorder to sample order. Do not synthesize or parse identifier strings.""",
                "estimated_impact": 0.40,
                "rationale": (
                    "U-Net is a suitable image-to-image candidate because skip connections "
                    "preserve spatial detail; compare it with the baseline on validation."
                ),
                "code_outline": "PyTorch U-Net with skip connections, validation-selected loss and training budget, output full image, align keyed predictions to sample_submission",
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
                "rationale": (
                    "Residual learning tests a complementary target parameterization. Compare "
                    "it with direct prediction before using it in an ensemble."
                ),
                "code_outline": "Conv encoder-decoder, predict residual, output = input - residual, align keyed predictions to sample_submission",
            },
        ]

    def _create_text_fallback_plan(
        self,
        domain: str,
        sota_analysis: dict[str, Any],
        competition_name: str = "",
    ) -> list[dict[str, Any]]:
        """
        Create fallback plan for text/NLP competitions (HuggingFace transformers).

        Uses pre-trained language models (RoBERTa, DistilBERT, or T5 for seq2seq).

        Args:
            domain: Competition domain (text_classification, seq_to_seq, etc.)
            sota_analysis: SOTA analysis results
            competition_name: Deprecated compatibility argument; ignored

        Returns:
            Domain-appropriate component list
        """
        del competition_name
        if domain in {"seq_to_seq", "text_normalization"}:
            return create_seq2seq_fallback_plan(
                domain,
                sota_analysis,
            )
        return create_text_fallback_plan(domain, sota_analysis)

    def _create_audio_fallback_plan(
        self,
        domain: str,
        sota_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Delegate to the shared benchmark-neutral audio fallback."""
        return create_audio_fallback_plan(domain, sota_analysis)

    def _format_sota_solutions(self, solutions: list[SOTASolution]) -> str:
        """Format external facts through the shared untrusted-data boundary."""
        return format_external_sota_solutions(solutions)

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
        return format_external_sota_details(solutions)

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
                    "prompt_modifier": (
                        "Evaluate efficient pretrained CNN baselines such as "
                        "EfficientNet-B0/B3 or ResNet50. Select by local validation."
                    ),
                    "model_preference": ["efficientnet_b0", "resnet50", "efficientnet_b3"],
                },
                {
                    "name": "aggressive",
                    "prompt_modifier": (
                        "Evaluate higher-capacity CNN candidates such as ResNet200d, "
                        "EfficientNet-B5/B7, or ConvNeXt. Validate augmentation and full "
                        "fine-tuning rather than assuming they help."
                    ),
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
                    "prompt_modifier": (
                        "Evaluate stable segmentation baselines such as U-Net with a ResNet34 "
                        "encoder or FPN, and select them using held-out masks."
                    ),
                    "model_preference": ["unet_resnet34", "fpn", "deeplabv3"],
                },
                {
                    "name": "aggressive",
                    "prompt_modifier": (
                        "Evaluate higher-capacity segmentation candidates such as U-Net++, "
                        "HRNet, or Mask R-CNN. Keep heavy augmentation only if validated."
                    ),
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
                    "prompt_modifier": (
                        "Evaluate compact pretrained text baselines such as DistilBERT and "
                        "RoBERTa-base using the supplied validation split."
                    ),
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
                    "prompt_modifier": (
                        "Evaluate a summary-feature baseline and a compact "
                        "time-frequency model on the same verified folds."
                    ),
                    "model_preference": [
                        "summary_feature_baseline",
                        "compact_spectrogram_model",
                    ],
                },
                {
                    "name": "aggressive",
                    "prompt_modifier": (
                        "Evaluate installed higher-capacity audio candidates only "
                        "when input compatibility and remaining budget permit. "
                        "Keep augmentation only when trusted CV improves."
                    ),
                    "model_preference": ["pretrained_audio_candidate"],
                },
                {
                    "name": "balanced",
                    "prompt_modifier": (
                        "Select a medium-capacity model from observed feature "
                        "shape, sample count, installed resources, and budget."
                    ),
                    "model_preference": ["budget_matched_audio_model"],
                },
            ]
        else:
            # TABULAR domain strategies (default for tabular, time_series, multi_modal, unknown)
            strategies = [
                {
                    "name": "conservative",
                    "prompt_modifier": (
                        "Start with reliable baselines such as XGBoost or LightGBM and "
                        "prioritize stable held-out performance over novelty."
                    ),
                    "model_preference": ["xgboost", "lightgbm", "random_forest"],
                },
                {
                    "name": "aggressive",
                    "prompt_modifier": "Focus on innovative approaches. Prioritize novel feature engineering, creative ensembles, and cutting-edge techniques.",
                    "model_preference": ["catboost", "neural_network", "stacking"],
                },
                {
                    "name": "balanced",
                    "prompt_modifier": (
                        "Combine validated baseline models with new features, retaining each "
                        "addition only when its ablation improves held-out performance."
                    ),
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

            # Record a measured actual-impact proxy, never self-declared impact.
            fitness = self._evaluate_plan_fitness(plan, state)

            candidate_plans.append((plan, strategy["name"], fitness))
            print(f"     Measured impact proxy: {fitness:.3f}")

        from .planner.eureka import _plan_selection_key

        candidate_plans.sort(
            key=lambda item: _plan_selection_key(item[0], state),
            reverse=True,
        )

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
        domain = state.get("domain_detected", "tabular")

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
        from .planner.eureka import mutate_plan_hyperparameters

        return mutate_plan_hyperparameters(plan, state, mutation_rate)

    def _get_hyperparameter_mutations(self, model_name: str) -> str:
        """Return validation-gated mutation guidance without numeric recipes."""
        from .planner.eureka import _get_hyperparameter_mutations

        return _get_hyperparameter_mutations(model_name)

    def _evaluate_plan_fitness(
        self,
        plan: list[AblationComponent],
        state: KaggleState,
    ) -> float:
        """Return the shared measured actual-impact proxy."""
        from .planner.eureka import evaluate_plan_fitness

        return evaluate_plan_fitness(plan, state)

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
