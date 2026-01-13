"""Plan refinement functions for the planner agent."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ...core.state import AblationComponent, DevelopmentResult, KaggleState


def refine_ablation_plan(
    state: KaggleState,
    sota_analysis: dict[str, Any],
    llm,
    use_dspy: bool,
    refine_ablation_plan_prompt: str,
    analyze_gaps_fn,
    create_refined_fallback_plan_fn,
    create_diversified_fallback_plan_fn,
    get_memory_summary_for_planning_fn,
) -> list[AblationComponent]:
    """
    Refine the ablation plan based on previous results using RL prompts.

    Args:
        state: Current state with previous results
        sota_analysis: SOTA analysis results
        llm: LLM instance
        use_dspy: Whether to use DSPy modules
        refine_ablation_plan_prompt: Prompt template for refinement
        analyze_gaps_fn: Function to analyze gaps
        create_refined_fallback_plan_fn: Function to create refined fallback plan
        create_diversified_fallback_plan_fn: Function to create diversified fallback plan
        get_memory_summary_for_planning_fn: Function to get memory summary

    Returns:
        Refined ablation plan
    """
    from langchain.schema import HumanMessage, SystemMessage

    from ...core.state import AblationComponent
    from ...utils.llm_utils import get_text_content

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

    # Perform Gap Analysis
    print("  🔍 Performing Gap Analysis...")
    gap_analysis = analyze_gaps_fn(
        state=state, previous_plan_str=previous_plan_str, test_results_str=test_results_str
    )
    gap_analysis_str = json.dumps(gap_analysis, indent=2)
    print(f"  🔍 Gap Analysis: {gap_analysis.get('improvement_strategy', 'No strategy found')}")

    # Use the refinement prompt
    prompt = refine_ablation_plan_prompt.format(
        gap_analysis=gap_analysis_str,
        previous_plan=previous_plan_str,
        test_results=test_results_str,
        current_score=current_score,
        memory_summary=get_memory_summary_for_planning_fn(state),
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
        if use_dspy:
            # For now, use fallback in refinement mode too
            print("  🔧 Using enhanced fallback with refinement logic")
            plan_data = create_refined_fallback_plan_fn(
                state,
                sota_analysis,
                test_results_summary,
                previous_plan,
                dev_results,
            )
        else:
            # Use LLM with refinement prompt
            messages = [
                SystemMessage(
                    content="You are a Kaggle Grandmaster expert at refining ML solutions based on test results."
                ),
                HumanMessage(content=prompt),
            ]

            response = llm.invoke(messages)
            plan_text = get_text_content(response.content).strip()

            # Parse JSON
            if "```json" in plan_text:
                plan_text = plan_text.split("```json")[1].split("```")[0].strip()
            elif "```" in plan_text:
                plan_text = plan_text.split("```")[1].split("```")[0].strip()

            plan_data = json.loads(plan_text)

    except Exception as e:
        print(f"  ⚠️  Refinement failed: {e!s}")
        print("  🔧 Using enhanced fallback with refinement logic")
        plan_data = create_refined_fallback_plan_fn(
            state,
            sota_analysis,
            test_results_summary,
            previous_plan,
            dev_results,
        )

    # Convert to AblationComponent objects
    components = []
    for i, item in enumerate(plan_data):
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

    # FIX: Plan diversity check - detect and avoid repeating same plan
    previous_plan_hashes = state.get("previous_plan_hashes", [])
    plan_hash = hash(tuple(sorted(c.name for c in components)))

    max_diversity_retries = 3
    retry_count = 0

    while plan_hash in previous_plan_hashes and retry_count < max_diversity_retries:
        diversity_strategies = [
            {"name": "neural_exploration", "focus": "deep_learning"},
            {"name": "feature_heavy", "focus": "feature_engineering"},
            {"name": "ensemble_focus", "focus": "ensemble"},
        ]
        strategy = diversity_strategies[retry_count % len(diversity_strategies)]

        retry_count += 1
        print(f"   [Planner] ⚠️ Plan already tried (attempt {retry_count}/{max_diversity_retries}) - trying {strategy['name']}...")

        # Try to generate a different plan using fallback with new strategy hint
        alternative_plan = create_diversified_fallback_plan_fn(
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

        components.sort(key=lambda x: x.estimated_impact, reverse=True)
        plan_hash = hash(tuple(sorted(c.name for c in components)))

    # Record this plan hash (keep last 10)
    previous_plan_hashes.append(plan_hash)
    state["previous_plan_hashes"] = previous_plan_hashes[-10:]

    if retry_count > 0:
        print(f"   [Planner] ✓ Found diverse plan after {retry_count} retries")

    return components


def create_refined_fallback_plan(
    state: KaggleState,
    sota_analysis: dict[str, Any],
    test_results: list[dict],
    previous_plan: list[AblationComponent],
    dev_results: list[DevelopmentResult],
    extract_validation_score_fn,
) -> list[dict[str, Any]]:
    """
    Create a refined fallback plan based on what worked in previous iteration.

    Args:
        state: Current state
        sota_analysis: SOTA analysis
        test_results: Previous test results
        previous_plan: Previous ablation plan
        dev_results: Development results
        extract_validation_score_fn: Function to extract validation score from stdout

    Returns:
        Refined plan as list of dicts
    """
    from ...core.config import is_metric_minimization

    # Bandit-lite: keep top-2 successful arms by reward, explore one new arm
    arms = []
    for idx, comp in enumerate(previous_plan):
        score = None
        if idx < len(dev_results):
            score = extract_validation_score_fn(dev_results[idx].stdout)
        reward = score if score is not None else comp.estimated_impact
        success = idx < len(dev_results) and dev_results[idx].success
        arms.append(
            {
                "component": comp,
                "reward": reward if reward is not None else 0.0,
                "success": success,
            }
        )

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
        if previous_best == 0.0:
            return True
        if is_minimize:
            return reward < previous_best
        return reward > previous_best

    # First try to find arms that actually improved
    improved_arms = [a for a in arms if actually_improved(a)]

    if improved_arms:
        improved_arms.sort(key=lambda a: a["reward"], reverse=not is_minimize)
        keep = improved_arms[:2]
        print(f"   [Planner] Keeping {len(keep)} arm(s) that improved score")
    else:
        successful_arms = [a for a in arms if a["success"]]
        if successful_arms:
            successful_arms.sort(key=lambda a: a["reward"], reverse=not is_minimize)
            keep = successful_arms[:1]
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

    NLP_DOMAINS = {"text_classification", "text_regression", "seq_to_seq", "nlp"}
    IMAGE_DOMAINS = {"image_classification", "image_regression", "image_segmentation",
                     "object_detection", "computer_vision", "image"}
    AUDIO_DOMAINS = {"audio_classification", "audio_regression"}

    model_count = sum(1 for p in plan if p["component_type"] == "model")

    plan = _add_domain_models(plan, domain, model_count, NLP_DOMAINS, IMAGE_DOMAINS, AUDIO_DOMAINS)

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


def _add_domain_models(
    plan: list[dict],
    domain: str,
    model_count: int,
    nlp_domains: set,
    image_domains: set,
    audio_domains: set,
) -> list[dict]:
    """Add domain-specific models to ensure model diversity."""
    if domain in nlp_domains:
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

    elif domain in image_domains:
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

    elif domain in audio_domains:
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

    return plan


def extract_validation_score(stdout: str) -> float | None:
    """Parse validation score from stdout if present."""
    if not stdout:
        return None

    match = re.search(r"Final Validation Performance:\s*([0-9\\.]+)", stdout)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def analyze_gaps(
    state: KaggleState,
    previous_plan_str: str,
    test_results_str: str,
    llm,
    planner_system_prompt: str,
    analyze_gaps_prompt: str,
    get_memory_summary_for_planning_fn,
) -> dict[str, Any]:
    """
    Analyze gaps between results and goal.

    Args:
        state: Current state
        previous_plan_str: JSON string of previous plan
        test_results_str: JSON string of test results
        llm: LLM instance
        planner_system_prompt: System prompt for planner
        analyze_gaps_prompt: Prompt template for gap analysis
        get_memory_summary_for_planning_fn: Function to get memory summary

    Returns:
        Dictionary with gap analysis
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from ...utils.llm_utils import get_text_content

    competition_info = state["competition_info"]
    current_score = state.get("current_performance_score", 0.0)

    prompt = analyze_gaps_prompt.format(
        previous_plan=previous_plan_str,
        test_results=test_results_str,
        metric=competition_info.evaluation_metric,
        current_score=current_score,
        target_score="SOTA (typically Top 10%)",
        memory_summary=get_memory_summary_for_planning_fn(state),
    )

    messages = [
        SystemMessage(content=planner_system_prompt),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
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
