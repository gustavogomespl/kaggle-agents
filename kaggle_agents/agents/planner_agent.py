"""
Planner Agent with Ablation-Driven Optimization.

This agent implements the ablation planning strategy from Google ADK,
identifying high-impact components for systematic improvement.
"""

import json
from typing import List, Dict, Any
from datetime import datetime

import dspy
from langchain_core.messages import HumanMessage, SystemMessage
from ..core.state import KaggleState, AblationComponent, DevelopmentResult, SOTASolution
from ..core.config import get_config, get_llm_for_role
from ..prompts.templates.planner_prompts import (
    PLANNER_SYSTEM_PROMPT,
    CREATE_ABLATION_PLAN_PROMPT,
    REFINE_ABLATION_PLAN_PROMPT,
    ANALYZE_SOTA_PROMPT,
    get_domain_guidance,
)
from ..optimization import create_optimizer, create_planner_metric


# ==================== DSPy Signatures ====================

class AblationPlannerSignature(dspy.Signature):
    """Signature for ablation plan generation."""

    competition_info: str = dspy.InputField(desc="Competition metadata and description")
    domain: str = dspy.InputField(desc="Competition domain (tabular, CV, NLP, etc.)")
    sota_summary: str = dspy.InputField(desc="Summary of SOTA solutions and strategies")
    domain_guidance: str = dspy.InputField(desc="Domain-specific guidance and priorities")

    ablation_plan: str = dspy.OutputField(desc="JSON list of ablation components")
    analysis: str = dspy.OutputField(desc="Analysis of SOTA patterns and rationale for plan")


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

    def forward(self, competition_info, domain, sota_summary, domain_guidance):
        """Generate ablation plan."""
        result = self.generate_plan(
            competition_info=competition_info,
            domain=domain,
            sota_summary=sota_summary,
            domain_guidance=domain_guidance,
        )
        return result


class SOTAAnalyzerModule(dspy.Module):
    """DSPy module for SOTA analysis."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(SOTAAnalysisSignature)

    def forward(self, sota_solutions):
        """Analyze SOTA solutions."""
        result = self.analyze(sota_solutions=sota_solutions)
        return result


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

        if self.use_dspy:
            # Try to load optimized module, fallback to base; if none, fall back to direct LLM
            optimizer = create_optimizer()
            self.planner_module = optimizer.load_optimized_prompt("planner")

            if self.planner_module is None:
                print("   No optimized planner module found -> using direct LLM path")
                self.use_dspy = False
                self.llm = get_llm_for_role(
                    role="planner",
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                )
            else:
                self.sota_analyzer = SOTAAnalyzerModule()
        else:
            # Use direct LLM calls
            self.llm = get_llm_for_role(
                role="planner",
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )

    def __call__(self, state: KaggleState) -> Dict[str, Any]:
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

        if is_refinement:
            print("\n" + "="*60)
            print("= PLANNER AGENT: Refining Ablation Plan (RL-based)")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("= PLANNER AGENT: Creating Ablation Plan")
            print("="*60)

        # 1. Analyze SOTA solutions
        print("\nAnalyzing SOTA patterns...")
        sota_analysis = self._analyze_sota_solutions(state)

        # 2. Generate ablation plan (initial or refinement)
        if is_refinement:
            print("\n🔄 Refining plan based on previous results...")
            ablation_plan = self._refine_ablation_plan(state, sota_analysis)
        else:
            print("\n📝 Generating ablation plan...")
            ablation_plan = self._generate_ablation_plan(state, sota_analysis)

        # 3. Validate and enhance plan
        validated_plan = self._validate_plan(ablation_plan)

        # 4. Print summary
        self._print_summary(validated_plan)

        # Return state updates
        return {
            "ablation_plan": validated_plan,
            "optimization_strategy": "ablation_driven",
            "last_updated": datetime.now(),
        }

    def _analyze_sota_solutions(self, state: KaggleState) -> Dict[str, Any]:
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

        if self.use_dspy and hasattr(self, 'sota_analyzer'):
            # Use DSPy module
            result = self.sota_analyzer(sota_solutions=sota_summary)

            analysis = {
                "common_models": result.common_models.split(", ") if result.common_models else [],
                "feature_patterns": result.feature_patterns.split(", ") if result.feature_patterns else [],
                "ensemble_strategies": result.ensemble_strategies if result.ensemble_strategies else "",
                "unique_tricks": result.unique_tricks.split(", ") if result.unique_tricks else [],
                "success_factors": result.success_factors.split(", ") if result.success_factors else [],
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
                analysis = json.loads(response.content)
            except json.JSONDecodeError:
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

    def _generate_ablation_plan(
        self,
        state: KaggleState,
        sota_analysis: Dict[str, Any],
    ) -> List[AblationComponent]:
        """
        Generate ablation plan based on competition info and SOTA analysis.

        Args:
            state: Current state
            sota_analysis: Analysis of SOTA solutions

        Returns:
            List of ablation components
        """
        competition_info = state["competition_info"]
        domain = state.get("domain_detected", "tabular")

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

        if self.use_dspy:
            # TEMPORARY FIX: Skip DSPy, use fallback directly
            # DSPy consistently generates only 2 components instead of 5
            print("  🔧 Using fallback plan (ensures 5 high-quality components)")
            plan_data = self._create_fallback_plan(domain, sota_analysis)

        else:
            # Use direct LLM call
            prompt = CREATE_ABLATION_PLAN_PROMPT.format(
                competition_info=comp_info_str,
                domain=domain,
                sota_summary=sota_summary,
            )

            messages = [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT + "\n\n" + domain_guidance),
                HumanMessage(content=prompt),
            ]

            # TEMPORARY FIX: Skip LLM call, use fallback directly
            print("  🔧 Using fallback plan (ensures 5 high-quality components)")
            plan_data = self._create_fallback_plan(domain, sota_analysis)

        # Convert to AblationComponent objects
        components = []
        for i, item in enumerate(plan_data):
            # Generate name if not provided by LLM
            name = item.get("name")
            if not name or name == "unnamed":
                comp_type = item.get("component_type", "component")
                # Create descriptive name from type and index
                name = f"{comp_type}_{i+1}"
                # If there's a description, try to extract key words
                if item.get("description"):
                    desc = item["description"].lower()
                    # Extract first meaningful word
                    for word in desc.split():
                        if len(word) > 4 and word not in ["using", "apply", "create", "implement"]:
                            name = f"{word}_{comp_type}"
                            break

            component = AblationComponent(
                name=name,
                component_type=item.get("component_type", "preprocessing"),
                code=item.get("code_outline", ""),
                estimated_impact=float(item.get("estimated_impact", 0.05)),
                tested=False,
                actual_impact=None,
            )
            components.append(component)

        # Sort by estimated impact (descending)
        components.sort(key=lambda x: x.estimated_impact, reverse=True)

        return components

    def _refine_ablation_plan(
        self, state: KaggleState, sota_analysis: Dict[str, Any]
    ) -> List[AblationComponent]:
        """
        Refine the ablation plan based on previous results using RL prompts.

        Args:
            state: Current state with previous results
            sota_analysis: SOTA analysis results

        Returns:
            Refined ablation plan
        """
        from ..prompts.templates.planner_prompts import REFINE_ABLATION_PLAN_PROMPT

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
                test_results_summary.append({
                    "component": component.name,
                    "type": component.component_type,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "impact": "positive" if result.success else "failed"
                })

        # Format previous plan for prompt
        previous_plan_str = json.dumps([
            {
                "name": c.name,
                "type": c.component_type,
                "description": c.code[:100] + "...",
                "estimated_impact": c.estimated_impact
            }
            for c in previous_plan
        ], indent=2)

        # Format test results
        test_results_str = json.dumps(test_results_summary, indent=2)

        # Use the refinement prompt
        prompt = REFINE_ABLATION_PLAN_PROMPT.format(
            previous_plan=previous_plan_str,
            test_results=test_results_str,
            current_score=current_score
        )

        # INJECT META-EVALUATOR GUIDANCE (RL Pattern)
        refinement_guidance = state.get("refinement_guidance", {})
        if refinement_guidance:
            guidance_text = "\n\n## 🎯 META-EVALUATOR GUIDANCE (from RL analysis)\n\n"

            if "planner_guidance" in refinement_guidance:
                guidance_text += f"**Strategic Guidance:**\n{refinement_guidance['planner_guidance']}\n\n"

            if "priority_fixes" in refinement_guidance and refinement_guidance["priority_fixes"]:
                guidance_text += f"**Priority Error Fixes:**\n"
                for error in refinement_guidance["priority_fixes"]:
                    guidance_text += f"- Avoid components that cause: {error}\n"
                guidance_text += "\n"

            if "success_amplification" in refinement_guidance and refinement_guidance["success_amplification"]:
                guidance_text += f"**Amplify These Successes:**\n"
                for success in refinement_guidance["success_amplification"]:
                    guidance_text += f"- {success}\n"
                guidance_text += "\n"

            if "component_type_guidance" in refinement_guidance:
                guidance_text += f"**Component-Specific Guidance:**\n"
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
                from langchain.schema import SystemMessage, HumanMessage

                messages = [
                    SystemMessage(content="You are a Kaggle Grandmaster expert at refining ML solutions based on test results."),
                    HumanMessage(content=prompt),
                ]

                response = self.llm.invoke(messages)
                plan_text = response.content.strip()

                # Parse JSON
                # Remove markdown code blocks if present
                if "```json" in plan_text:
                    plan_text = plan_text.split("```json")[1].split("```")[0].strip()
                elif "```" in plan_text:
                    plan_text = plan_text.split("```")[1].split("```")[0].strip()

                plan_data = json.loads(plan_text)

        except Exception as e:
            print(f"  ⚠️  Refinement failed: {str(e)}")
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
                name=item.get("name", f"refined_component_{i+1}"),
                component_type=item.get("component_type", "model"),
                code=code,
                estimated_impact=item.get("estimated_impact", 0.15)
            )
            components.append(component)

        # Sort by estimated impact
        components.sort(key=lambda x: x.estimated_impact, reverse=True)

        return components

    def _create_refined_fallback_plan(
        self,
        state: KaggleState,
        sota_analysis: Dict[str, Any],
        test_results: List[Dict],
        previous_plan: List[AblationComponent],
        dev_results: List[DevelopmentResult],
    ) -> List[Dict[str, Any]]:
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
            arms.append({
                "component": comp,
                "reward": reward if reward is not None else 0.0,
                "success": success,
            })

        # Exploit: keep top-2 successful arms
        successful_arms = [a for a in arms if a["success"]]
        successful_arms.sort(key=lambda a: a["reward"], reverse=True)
        keep = successful_arms[:2]

        plan = []

        # Ensure a strong feature engineering arm is present
        fe_in_keep = any(a["component"].component_type == "feature_engineering" for a in keep)
        if not fe_in_keep:
            plan.append({
                "name": "advanced_feature_engineering",
                "component_type": "feature_engineering",
                "description": "Polynomial + interaction features with leak-safe pipelines (imputer/encoder in CV)",
                "estimated_impact": 0.15,
                "rationale": "Consistently strong FE baseline",
                "code_outline": "Pipeline with ColumnTransformer, SimpleImputer, OneHot/TargetEncoder, interactions",
            })

        # Add kept winners
        for arm in keep:
            comp = arm["component"]
            plan.append({
                "name": comp.name,
                "component_type": comp.component_type,
                "description": comp.code or comp.component_type,
                "estimated_impact": float(comp.estimated_impact) if comp.estimated_impact else max(0.12, arm["reward"]),
                "rationale": "Kept from previous iteration (top reward)",
                "code_outline": comp.code or comp.component_type,
            })

        # Ensure at least two model components
        model_count = sum(1 for p in plan if p["component_type"] == "model")
        if model_count < 2:
            plan.append({
                "name": "lightgbm_fast_cv",
                "component_type": "model",
                "description": "LightGBM with OHE pipeline, 5-fold StratifiedKFold, early stopping via callbacks",
                "estimated_impact": 0.20,
                "rationale": "High-ROI baseline model",
                "code_outline": "ColumnTransformer + LGBMClassifier(num_leaves=63, learning_rate=0.03, n_estimators=1200)",
            })
            model_count += 1

        if model_count < 2:
            plan.append({
                "name": "xgboost_fast_cv",
                "component_type": "model",
                "description": "XGBoost with OHE pipeline, 5-fold CV, moderate depth",
                "estimated_impact": 0.18,
                "rationale": "Adds diversity for ensemble",
                "code_outline": "XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=800, subsample=0.8)",
            })

        # Exploration arm if capacity allows
        if len(plan) < 4:
            plan.append({
                "name": "stacking_light",
                "component_type": "ensemble",
                "description": "Weighted average of top models using CV rewards as weights; validate submission vs sample",
                "estimated_impact": 0.12,
                "rationale": "Cheap ensemble leveraging existing predictions",
                "code_outline": "Load saved preds, weight by CV reward, validate sample_submission shape/ids",
            })

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

    def _validate_plan(self, plan: List[AblationComponent]) -> List[AblationComponent]:
        """
        Validate and enhance the ablation plan.

        Args:
            plan: Initial plan

        Returns:
            Validated plan
        """
        # Filter out invalid components (keep only high impact >= 10%)
        valid_plan = [c for c in plan if c.estimated_impact >= 0.10]

        # Limit to maximum 6 components (quality over quantity)
        if len(valid_plan) > 6:
            print(f"  ⚠️  Plan has {len(valid_plan)} components - limiting to top 6 by impact")
            valid_plan = sorted(valid_plan, key=lambda x: x.estimated_impact, reverse=True)[:6]

        # CRITICAL: Ensure at least TWO model components exist
        # Model components are required to generate predictions and create ensembles
        model_count = sum(1 for c in valid_plan if c.component_type == "model")

        if model_count == 0:
            print("  ⚠️  No 'model' components found - adding 2 baseline models")
            # Add two different baseline models
            baseline_lgbm = AblationComponent(
                name="baseline_lightgbm",
                component_type="model",
                code="",  # Will be generated by developer agent
                estimated_impact=0.20,
                tested=False,
                actual_impact=None,
            )
            baseline_xgb = AblationComponent(
                name="baseline_xgboost",
                component_type="model",
                code="",
                estimated_impact=0.18,
                tested=False,
                actual_impact=None,
            )
            valid_plan.extend([baseline_lgbm, baseline_xgb])
            print(f"     Added: {baseline_lgbm.name} and {baseline_xgb.name}")

        elif model_count == 1:
            print("  ⚠️  Only 1 'model' component found - adding second baseline model")
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

        # Ensure 3-5 components total
        if len(valid_plan) < 3:
            print("  ⚠️  Plan has fewer than 3 components")
        elif len(valid_plan) > 5:
            print(f"  ⚠️  Plan still has {len(valid_plan)} components after filtering")

        # Sort by type: preprocessing first, then models, then ensembles
        # This ensures data is prepared before models are trained
        preprocessing_components = [c for c in valid_plan if c.component_type in ["preprocessing", "feature_engineering"]]
        model_components = [c for c in valid_plan if c.component_type == "model"]
        other_components = [c for c in valid_plan if c.component_type not in ["preprocessing", "feature_engineering", "model"]]

        # Reorder: preprocessing first, then models, then ensembles
        valid_plan = preprocessing_components + model_components + other_components

        # Debug log: Show final plan composition
        print(f"  📊 Final plan: {len(preprocessing_components)} FE + {len(model_components)} models + {len(other_components)} ensemble = {len(valid_plan)} total")

        return valid_plan


    def _create_fallback_plan(
        self,
        domain: str,
        sota_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Create a fallback plan when LLM parsing fails.

        Args:
            domain: Competition domain
            sota_analysis: SOTA analysis results

        Returns:
            List of component dictionaries (always 4-5 components)
        """
        plan = []

        # ALWAYS add feature engineering first (high impact)
        plan.append({
            "name": "advanced_feature_engineering",
            "component_type": "feature_engineering",
            "description": "Create polynomial features (degree 2), feature interactions (ratio, diff, product), statistical transformations (log, sqrt), and target encoding for categorical features",
            "estimated_impact": 0.15,
            "rationale": "Comprehensive feature engineering improves scores by 10-20% in tabular competitions",
            "code_outline": "Use PolynomialFeatures(degree=2), create ratio/diff/product features, apply log/sqrt transforms, use TargetEncoder"
        })

        # ALWAYS add 3 diverse models for ensemble diversity
        plan.extend([
            {
                "name": "lightgbm_optuna_tuned",
                "component_type": "model",
                "description": "LightGBM with Optuna hyperparameter optimization: 15 trials, tuning learning_rate, num_leaves, max_depth, min_child_samples",
                "estimated_impact": 0.22,
                "rationale": "LightGBM consistently wins tabular competitions. Optuna finds better parameters than manual tuning.",
                "code_outline": "LGBMRegressor/Classifier with OptunaSearchCV, 5-fold CV, early_stopping_rounds=100"
            },
            {
                "name": "xgboost_optuna_tuned",
                "component_type": "model",
                "description": "XGBoost with Optuna hyperparameter optimization: 15 trials, tuning max_depth, learning_rate, subsample, colsample_bytree",
                "estimated_impact": 0.20,
                "rationale": "XGBoost provides different regularization than LightGBM. Optuna ensures optimal capacity.",
                "code_outline": "XGBRegressor/Classifier with OptunaSearchCV, 5-fold CV, early_stopping_rounds=50"
            },
            {
                "name": "catboost_optuna_tuned",
                "component_type": "model",
                "description": "CatBoost with Optuna hyperparameter optimization: 15 trials, tuning depth, learning_rate, l2_leaf_reg",
                "estimated_impact": 0.19,
                "rationale": "CatBoost handles categorical features natively. Tuning depth is critical for performance.",
                "code_outline": "CatBoostRegressor/Classifier with OptunaSearchCV, cat_features parameter, 5-fold CV"
            },
            {
                "name": "neural_network_mlp",
                "component_type": "model",
                "description": "Simple MLP Neural Network using Scikit-Learn or PyTorch (if available). Standard scaling is CRITICAL.",
                "estimated_impact": 0.15,
                "rationale": "Neural Networks capture different patterns than tree-based models, adding valuable diversity to the ensemble.",
                "code_outline": "MLPClassifier/Regressor or PyTorch simple net. Must use StandardScaler/MinMaxScaler on inputs. Early stopping."
            }
        ])

        # ALWAYS add stacking ensemble (combines the 4 models above)
        plan.append({
            "name": "stacking_ensemble",
            "component_type": "ensemble",
            "description": "Stack LightGBM, XGBoost, CatBoost, and NN predictions using Ridge/Logistic regression as meta-learner",
            "estimated_impact": 0.25,
            "rationale": "Stacking combines diverse models (Trees + NN) and typically improves scores by 5-10%",
            "code_outline": "StackingRegressor/Classifier with base_estimators=[lgb, xgb, cat, nn], final_estimator=Ridge/LogisticRegression, cv=5"
        })

        return plan

    def _format_sota_solutions(self, solutions: List[SOTASolution]) -> str:
        """Format SOTA solutions for prompts."""
        formatted = []
        for sol in solutions[:5]:  # Top 5
            formatted.append(f"""
Title: {sol.title}
Votes: {sol.votes}
Models: {', '.join(sol.models_used) if sol.models_used else 'N/A'}
Features: {', '.join(sol.feature_engineering) if sol.feature_engineering else 'N/A'}
Ensemble: {sol.ensemble_approach or 'N/A'}
""")
        return "\n---\n".join(formatted)

    def _print_summary(self, plan: List[AblationComponent]) -> None:
        """Print plan summary."""
        print(f"\n= Ablation Plan Created: {len(plan)} components")
        print("-" * 60)

        for i, comp in enumerate(plan, 1):
            print(f"\n{i}. {comp.name} ({comp.component_type})")
            print(f"   Estimated Impact: {comp.estimated_impact:.1%}")
            if comp.code:
                print(f"   Code: {comp.code[:80]}...")

        print("\n" + "="*60)


# ==================== LangGraph Node Function ====================

def planner_agent_node(state: KaggleState) -> Dict[str, Any]:
    """
    LangGraph node function for the planner agent.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = PlannerAgent()
    return agent(state)
