"""
Planner Agent with Ablation-Driven Optimization.

This agent implements the ablation planning strategy from Google ADK,
identifying high-impact components for systematic improvement.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from ...core.config import get_config, get_llm_for_role
from ...core.state import (
    AblationComponent,
    KaggleState,
    get_memory_summary_for_planning,
)
from ...optimization import create_optimizer
from ...prompts.templates.planner_prompts import (
    ANALYZE_GAPS_PROMPT,
    ANALYZE_SOTA_PROMPT,
    CREATE_ABLATION_PLAN_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REFINE_ABLATION_PLAN_PROMPT,
    get_domain_guidance,
)
from ...utils.llm_utils import get_text_content
from .curriculum import extract_curriculum_insights
from .debug_loop import handle_debug_loop_trigger
from .domain_patterns import extract_domain_specific_patterns, format_domain_insights
from .dspy_modules import SOTAAnalyzerModule
from .eureka import generate_with_eureka
from .fallback_plans import (
    create_diversified_fallback_plan,
    create_fallback_plan,
)
from .plan_refinement import (
    analyze_gaps,
    create_refined_fallback_plan,
    extract_validation_score,
    refine_ablation_plan,
)
from .sota_analysis import (
    analyze_sota_solutions,
    format_sota_details,
)
from .validation import (
    detect_multimodal_competition,
    is_image_competition_without_features,
    validate_plan,
)


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

        # ALWAYS initialize self.llm
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
        crossover_guidance = state.get("crossover_guidance", {})
        evolutionary_generation = state.get("evolutionary_generation", 0)
        use_eureka = bool(crossover_guidance) or evolutionary_generation > 0 or is_refinement

        # Check for debug loop trigger from Meta-Evaluator
        debug_result = handle_debug_loop_trigger(state)
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

        # Detect multi-modal competition
        multimodal_info = detect_multimodal_competition(state)
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
            print("\n🧬 Using Eureka multi-candidate evolutionary planning...")
            eureka_result = generate_with_eureka(
                state,
                sota_analysis,
                n_candidates=3,
                create_fallback_plan_fn=self._create_fallback_plan,
                coerce_components_fn=self._coerce_components,
            )

            # Validate the selected plan
            validated_plan = self._validate_plan(eureka_result["ablation_plan"], state=state)

            # Print summary
            self._print_summary(validated_plan)

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

        return {
            "ablation_plan": validated_plan,
            "optimization_strategy": "ablation_driven",
            "last_updated": datetime.now(),
        }

    def _analyze_sota_solutions(self, state: KaggleState) -> dict[str, Any]:
        """Analyze SOTA solutions to extract patterns."""
        return analyze_sota_solutions(
            state=state,
            llm=self.llm,
            use_dspy=self.use_dspy,
            sota_analyzer=getattr(self, "sota_analyzer", None),
            planner_system_prompt=PLANNER_SYSTEM_PROMPT,
            analyze_sota_prompt=ANALYZE_SOTA_PROMPT,
        )

    def _generate_ablation_plan(
        self,
        state: KaggleState,
        sota_analysis: dict[str, Any],
    ) -> list[AblationComponent]:
        """Generate ablation plan based on competition info and SOTA analysis."""
        from langchain_core.messages import HumanMessage, SystemMessage

        competition_info = state["competition_info"]
        domain = state.get("domain_detected", "tabular")

        # Extract curriculum learning insights
        curriculum_insights = extract_curriculum_insights(state)

        # Get raw SOTA solutions for "Adopt & Improve" strategy
        sota_solutions = state.get("sota_solutions", [])
        sota_details = format_sota_details(sota_solutions)
        memory_summary = get_memory_summary_for_planning(state)

        # Extract domain-specific patterns and format insights
        domain_patterns = extract_domain_specific_patterns(sota_solutions, domain)
        domain_insights = format_domain_insights(domain, domain_patterns)

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

        # Resolve max components
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

        # Combine domain_guidance with domain_insights
        enhanced_domain_guidance = domain_guidance + "\n\n" + domain_insights if domain_insights else domain_guidance

        if self.use_dspy:
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
            # Use direct LLM call
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

Generate a plan that leverages proven successful strategies and avoids known pitfalls.

Return a JSON array with up to {max_components} components. Each component must have:
- name: unique identifier
- component_type: one of [feature_engineering, model, preprocessing, ensemble]
- description: what this component does
- estimated_impact: float 0.0-1.0
- code_outline: brief implementation sketch"""

            # INJECT FAILURE ANALYSIS
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

            print("  🧠 Using LLM for ablation plan generation (Adopt & Improve strategy)...")
            try:
                response = self.llm.invoke(messages)
                try:
                    plan_text = get_text_content(response.content)
                except Exception:
                    plan_text = response.content if hasattr(response, "content") else response
                if not isinstance(plan_text, str):
                    try:
                        plan_text = "\n".join(map(str, plan_text))
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
        """Refine the ablation plan based on previous results."""
        return refine_ablation_plan(
            state=state,
            sota_analysis=sota_analysis,
            llm=self.llm,
            use_dspy=self.use_dspy,
            refine_ablation_plan_prompt=REFINE_ABLATION_PLAN_PROMPT,
            analyze_gaps_fn=lambda **kwargs: analyze_gaps(
                llm=self.llm,
                planner_system_prompt=PLANNER_SYSTEM_PROMPT,
                analyze_gaps_prompt=ANALYZE_GAPS_PROMPT,
                get_memory_summary_for_planning_fn=get_memory_summary_for_planning,
                **kwargs,
            ),
            create_refined_fallback_plan_fn=lambda *args: create_refined_fallback_plan(
                *args,
                extract_validation_score_fn=extract_validation_score,
            ),
            create_diversified_fallback_plan_fn=create_diversified_fallback_plan,
            get_memory_summary_for_planning_fn=get_memory_summary_for_planning,
        )

    def _validate_plan(
        self,
        plan: list[AblationComponent],
        *,
        state: KaggleState | None = None,
    ) -> list[AblationComponent]:
        """Validate and enhance the ablation plan."""
        return validate_plan(
            plan=plan,
            state=state,
            coerce_components_fn=self._coerce_components,
            is_image_competition_without_features_fn=is_image_competition_without_features,
        )

    def _create_fallback_plan(
        self,
        domain: str,
        sota_analysis: dict[str, Any],
        curriculum_insights: str = "",
        *,
        state: KaggleState | None = None,
    ) -> list[dict[str, Any]]:
        """Create domain-specific fallback plan when LLM parsing fails."""
        return create_fallback_plan(
            domain=domain,
            sota_analysis=sota_analysis,
            curriculum_insights=curriculum_insights,
            state=state,
            is_image_competition_without_features_fn=is_image_competition_without_features,
        )

    def _parse_llm_plan_response(
        self,
        response_text: str,
        sota_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Parse LLM response into a list of component dictionaries."""
        # Coerce non-string responses
        if not isinstance(response_text, str):
            try:
                response_text = "\n".join(map(str, response_text))
            except Exception:
                response_text = str(response_text)

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

        return []

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
