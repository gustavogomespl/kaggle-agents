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
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from ..core.state import KaggleState, AblationComponent, SOTASolution
from ..core.config import get_config
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
            # Try to load optimized module, fallback to base
            optimizer = create_optimizer()
            self.planner_module = optimizer.load_optimized_prompt("planner")

            if self.planner_module is None:
                print("   Using base (unoptimized) planner module")
                self.planner_module = AblationPlannerModule()

            self.sota_analyzer = SOTAAnalyzerModule()
        else:
            # Use direct LLM calls
            if self.config.llm.provider == "openai":
                self.llm = ChatOpenAI(
                    model=self.config.llm.model,
                    temperature=self.config.llm.temperature,
                )
            else:
                self.llm = ChatAnthropic(
                    model=self.config.llm.model,
                    temperature=self.config.llm.temperature,
                )

    def __call__(self, state: KaggleState) -> Dict[str, Any]:
        """
        Execute the planner agent.

        Args:
            state: Current workflow state

        Returns:
            State updates with ablation plan
        """
        print("\n" + "="*60)
        print("= PLANNER AGENT: Creating Ablation Plan")
        print("="*60)

        # 1. Analyze SOTA solutions
        print("\nAnalyzing SOTA patterns...")
        sota_analysis = self._analyze_sota_solutions(state)

        # 2. Generate ablation plan
        print("\n< Generating ablation plan...")
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

        if self.use_dspy:
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
            # Use DSPy module
            result = self.planner_module(
                competition_info=comp_info_str,
                domain=domain,
                sota_summary=sota_summary,
                domain_guidance=domain_guidance,
            )

            # Parse JSON plan
            try:
                plan_data = json.loads(result.ablation_plan)
            except json.JSONDecodeError:
                print("  Failed to parse DSPy plan, using fallback")
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

            response = self.llm.invoke(messages)

            # Parse JSON
            try:
                # Extract JSON from markdown code block if present
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                plan_data = json.loads(content.strip())
            except (json.JSONDecodeError, IndexError):
                print("  Failed to parse LLM plan, using fallback")
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

    def _validate_plan(self, plan: List[AblationComponent]) -> List[AblationComponent]:
        """
        Validate and enhance the ablation plan.

        Args:
            plan: Initial plan

        Returns:
            Validated plan
        """
        # Filter out invalid components (keep only high impact)
        valid_plan = [c for c in plan if c.estimated_impact > 0.05]

        # Limit to maximum 5 components (quality over quantity)
        if len(valid_plan) > 5:
            print(f"  ⚠️  Plan has {len(valid_plan)} components - limiting to top 5 by impact")
            valid_plan = sorted(valid_plan, key=lambda x: x.estimated_impact, reverse=True)[:5]

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
            List of component dictionaries
        """
        plan = []

        # Add model components based on SOTA
        common_models = sota_analysis.get("common_models", [])
        if common_models:
            for model in common_models[:2]:
                plan.append({
                    "name": f"train_{model.lower().replace(' ', '_')}",
                    "component_type": "model",
                    "description": f"Train {model} model",
                    "estimated_impact": 0.10,
                    "code_outline": f"from sklearn import ...; model = {model}()",
                })

        # Add feature engineering
        plan.append({
            "name": "advanced_feature_engineering",
            "component_type": "feature_engineering",
            "description": "Advanced feature engineering techniques",
            "estimated_impact": 0.15,
            "code_outline": "Create interaction features, polynomial features, etc.",
        })

        # Add ensemble
        plan.append({
            "name": "model_ensemble",
            "component_type": "ensemble",
            "description": "Ensemble multiple models",
            "estimated_impact": 0.08,
            "code_outline": "Combine predictions from multiple models",
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
