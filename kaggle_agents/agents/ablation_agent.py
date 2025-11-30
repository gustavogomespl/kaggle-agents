"""
Ablation Study Agent.

Implements the ADK pattern for systematic ablation studies:
1. Generate ablation code to test component contributions
2. Execute ablations and collect results
3. Summarize findings and identify improvement targets
4. Plan and implement focused improvements

This agent follows the inner/outer loop pattern from the ADK example.
"""

import re
import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.state import KaggleState
from ..core.config import get_config
from ..core.logger import get_logger
from ..tools.code_executor import CodeExecutor
from ..utils.log_parser import TrainingFeedback, parse_training_logs
from ..prompts.templates.developer_prompts import (
    ABLATION_STUDY_PROMPT,
    ABLATION_STUDY_SEQUENTIAL_PROMPT,
    SUMMARIZE_ABLATION_PROMPT,
    EXTRACT_IMPROVEMENT_PLAN_PROMPT,
    EXTRACT_IMPROVEMENT_PLAN_SEQUENTIAL_PROMPT,
    PLAN_REFINEMENT_PROMPT,
    IMPLEMENT_PLAN_PROMPT,
)

logger = get_logger(__name__)


@dataclass
class AblationResult:
    """Result of a single ablation test."""
    
    component_name: str
    baseline_score: float
    modified_score: float
    delta: float
    execution_time: float = 0.0
    details: str = ""


@dataclass
class AblationSummary:
    """Summary of all ablation results."""
    
    results: list[AblationResult] = field(default_factory=list)
    most_impactful: str = ""
    least_impactful: str = ""
    recommended_improvements: list[str] = field(default_factory=list)
    raw_output: str = ""


@dataclass
class ImprovementPlan:
    """Plan for improving a specific code block."""
    
    code_block: str
    plan: str
    expected_impact: float = 0.0
    risk_level: str = "medium"
    implemented: bool = False
    result_score: float | None = None


@dataclass
class AblationState:
    """State tracking for ablation study iterations."""
    
    # Current solution
    current_code: str = ""
    current_score: float = 0.0
    current_std: float = 0.0
    
    # Ablation tracking
    ablation_round: int = 0
    ablation_results: list[AblationSummary] = field(default_factory=list)
    tested_components: list[str] = field(default_factory=list)
    
    # Improvement tracking
    refinement_round: int = 0
    improvement_plans: list[ImprovementPlan] = field(default_factory=list)
    improved_code_blocks: list[str] = field(default_factory=list)
    
    # Score history
    score_history: list[float] = field(default_factory=list)
    best_score: float = 0.0
    best_code: str = ""


class AblationStudyAgent:
    """
    Agent for performing systematic ablation studies.
    
    Implements the ADK pattern:
    - Outer loop: Ablation + Refinement cycles
    - Inner loop: Plan refinement iterations
    """
    
    def __init__(
        self,
        model,
        max_outer_loops: int = 3,
        max_inner_loops: int = 2,
        improvement_threshold: float = 0.001,
    ):
        """
        Initialize ablation study agent.
        
        Args:
            model: LangChain chat model
            max_outer_loops: Maximum ablation + refinement cycles
            max_inner_loops: Maximum plan refinement iterations
            improvement_threshold: Minimum improvement to continue
        """
        self.model = model
        self.max_outer_loops = max_outer_loops
        self.max_inner_loops = max_inner_loops
        self.improvement_threshold = improvement_threshold
        self.config = get_config()
        self.executor = CodeExecutor(timeout=600)
        
    def run_ablation_study(
        self,
        state: KaggleState,
        current_code: str,
        cv_score: float,
        cv_std: float = 0.0,
    ) -> dict[str, Any]:
        """
        Run a complete ablation study cycle.
        
        Args:
            state: Current workflow state
            current_code: The code to analyze
            cv_score: Current CV score
            cv_std: Current CV standard deviation
            
        Returns:
            State update with ablation results and improvements
        """
        logger.info("🔬 Starting Ablation Study")
        
        ablation_state = AblationState(
            current_code=current_code,
            current_score=cv_score,
            current_std=cv_std,
            score_history=[cv_score],
            best_score=cv_score,
            best_code=current_code,
        )
        
        competition_dir = state.get("competition_dir", ".")
        
        # Outer loop: Ablation + Refinement cycles
        for outer_round in range(self.max_outer_loops):
            logger.info(f"📊 Outer Round {outer_round + 1}/{self.max_outer_loops}")
            
            # Step 1: Generate and run ablation study
            ablation_summary = self._run_ablation_round(
                ablation_state, competition_dir
            )
            
            if not ablation_summary.results:
                logger.warning("No ablation results, skipping refinement")
                continue
                
            ablation_state.ablation_results.append(ablation_summary)
            ablation_state.ablation_round += 1
            
            # Step 2: Extract improvement plan
            improvement_plan = self._extract_improvement_plan(
                ablation_state, ablation_summary
            )
            
            if not improvement_plan:
                logger.warning("Could not extract improvement plan")
                continue
            
            # Step 3: Inner loop - Plan refinement
            best_inner_score = ablation_state.current_score
            best_inner_code = ablation_state.current_code
            
            for inner_round in range(self.max_inner_loops):
                logger.info(f"  🔄 Inner Round {inner_round + 1}/{self.max_inner_loops}")
                
                # Implement the plan
                improved_code, result = self._implement_and_test_plan(
                    ablation_state, improvement_plan, competition_dir
                )
                
                if result and result.cv_mean > best_inner_score:
                    best_inner_score = result.cv_mean
                    best_inner_code = improved_code
                    improvement_plan.result_score = result.cv_mean
                    improvement_plan.implemented = True
                    logger.info(f"  ✅ Improvement: {result.cv_mean:.6f}")
                
                # Refine the plan based on results
                if inner_round < self.max_inner_loops - 1:
                    improvement_plan = self._refine_plan(
                        ablation_state, improvement_plan, result
                    )
            
            # Update state with best result from inner loop
            if best_inner_score > ablation_state.best_score:
                ablation_state.best_score = best_inner_score
                ablation_state.best_code = best_inner_code
                logger.info(f"🎯 New best score: {best_inner_score:.6f}")
            
            ablation_state.current_code = best_inner_code
            ablation_state.current_score = best_inner_score
            ablation_state.score_history.append(best_inner_score)
            ablation_state.refinement_round += 1
            
            # Check if improvement is sufficient to continue
            if len(ablation_state.score_history) >= 2:
                last_improvement = (
                    ablation_state.score_history[-1] - 
                    ablation_state.score_history[-2]
                )
                if last_improvement < self.improvement_threshold:
                    logger.info(
                        f"Stopping: improvement {last_improvement:.6f} < "
                        f"threshold {self.improvement_threshold}"
                    )
                    break
        
        # Prepare state update
        return {
            "ablation_results": [
                {
                    "round": i,
                    "results": [
                        {
                            "component": r.component_name,
                            "baseline": r.baseline_score,
                            "modified": r.modified_score,
                            "delta": r.delta,
                        }
                        for r in summary.results
                    ],
                    "most_impactful": summary.most_impactful,
                }
                for i, summary in enumerate(ablation_state.ablation_results)
            ],
            "improved_code": ablation_state.best_code,
            "score_progression": ablation_state.score_history,
            "final_score": ablation_state.best_score,
            "improvement_plans": [
                {
                    "plan": p.plan,
                    "implemented": p.implemented,
                    "result_score": p.result_score,
                }
                for p in ablation_state.improvement_plans
            ],
        }
    
    def _run_ablation_round(
        self,
        state: AblationState,
        competition_dir: str,
    ) -> AblationSummary:
        """Run a single ablation study round."""
        
        # Generate ablation code
        if state.ablation_round == 0:
            prompt = ABLATION_STUDY_PROMPT.format(
                code=state.current_code,
                cv_score=state.current_score,
                cv_std=state.current_std,
            )
        else:
            # Format previous ablations
            prev_ablations = ""
            for i, summary in enumerate(state.ablation_results):
                prev_ablations += f"## Round {i + 1}\n"
                for result in summary.results:
                    prev_ablations += (
                        f"- {result.component_name}: "
                        f"delta={result.delta:+.4f}\n"
                    )
            
            prompt = ABLATION_STUDY_SEQUENTIAL_PROMPT.format(
                code=state.current_code,
                cv_score=state.current_score,
                cv_std=state.current_std,
                prev_ablations=prev_ablations,
                tested_components=", ".join(state.tested_components),
            )
        
        # Generate ablation code
        messages = [
            SystemMessage(content="You are a Kaggle Grandmaster expert in ablation studies."),
            HumanMessage(content=prompt),
        ]
        
        response = self.model.invoke(messages)
        ablation_code = self._extract_code(response.content)
        
        if not ablation_code:
            logger.warning("Failed to generate ablation code")
            return AblationSummary()
        
        # Execute ablation code
        logger.info("  Executing ablation study...")
        exec_result = self.executor.execute(
            ablation_code,
            working_dir=competition_dir,
        )
        
        if not exec_result.success:
            logger.warning(f"Ablation execution failed: {exec_result.error}")
            return AblationSummary(raw_output=exec_result.stdout)
        
        # Parse ablation results
        results = self._parse_ablation_results(exec_result.stdout)
        
        # Update tested components
        for result in results:
            if result.component_name not in state.tested_components:
                state.tested_components.append(result.component_name)
        
        # Create summary
        summary = AblationSummary(
            results=results,
            raw_output=exec_result.stdout,
        )
        
        if results:
            # Sort by absolute delta
            sorted_results = sorted(
                results, key=lambda r: abs(r.delta), reverse=True
            )
            summary.most_impactful = sorted_results[0].component_name
            summary.least_impactful = sorted_results[-1].component_name
        
        return summary
    
    def _parse_ablation_results(self, stdout: str) -> list[AblationResult]:
        """Parse ablation results from stdout."""
        results = []
        
        # Pattern: [ABLATION:component] baseline=X modified=Y delta=Z
        pattern = r"\[ABLATION:(\w+)\]\s*baseline=([\d.]+)\s*modified=([\d.]+)\s*delta=([-\d.]+)"
        
        for match in re.finditer(pattern, stdout):
            component = match.group(1)
            baseline = float(match.group(2))
            modified = float(match.group(3))
            delta = float(match.group(4))
            
            results.append(AblationResult(
                component_name=component,
                baseline_score=baseline,
                modified_score=modified,
                delta=delta,
            ))
        
        return results
    
    def _extract_improvement_plan(
        self,
        state: AblationState,
        ablation_summary: AblationSummary,
    ) -> ImprovementPlan | None:
        """Extract improvement plan from ablation results."""
        
        # Format ablation results
        ablation_text = ""
        for result in ablation_summary.results:
            ablation_text += (
                f"- {result.component_name}: "
                f"delta={result.delta:+.4f} "
                f"(baseline={result.baseline_score:.4f})\n"
            )
        
        # Choose prompt based on whether we have previous improvements
        if state.improved_code_blocks:
            prev_blocks = "\n".join([
                f"## Block {i + 1}\n```python\n{block[:500]}...\n```"
                for i, block in enumerate(state.improved_code_blocks[-3:])
            ])
            prompt = EXTRACT_IMPROVEMENT_PLAN_SEQUENTIAL_PROMPT.format(
                code=state.current_code,
                ablation_results=ablation_text,
                prev_code_blocks=prev_blocks,
            )
        else:
            prompt = EXTRACT_IMPROVEMENT_PLAN_PROMPT.format(
                code=state.current_code,
                ablation_results=ablation_text,
            )
        
        # Generate plan
        messages = [
            SystemMessage(content="You are a Kaggle Grandmaster planning improvements."),
            HumanMessage(content=prompt),
        ]
        
        response = self.model.invoke(messages)
        
        # Parse JSON response
        try:
            json_match = re.search(r"\[.*\]", response.content, re.DOTALL)
            if json_match:
                plans = json.loads(json_match.group())
                if plans and len(plans) > 0:
                    plan_dict = plans[0]
                    plan = ImprovementPlan(
                        code_block=plan_dict.get("code_block", ""),
                        plan=plan_dict.get("plan", ""),
                        expected_impact=float(plan_dict.get("expected_impact", 0.01)),
                        risk_level=plan_dict.get("risk_level", "medium"),
                    )
                    state.improvement_plans.append(plan)
                    return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse improvement plan: {e}")
        
        return None
    
    def _implement_and_test_plan(
        self,
        state: AblationState,
        plan: ImprovementPlan,
        competition_dir: str,
    ) -> tuple[str, TrainingFeedback | None]:
        """Implement the improvement plan and test it."""
        
        prompt = IMPLEMENT_PLAN_PROMPT.format(
            code_block=plan.code_block,
            plan=plan.plan,
            full_code=state.current_code[:5000],  # Truncate for context
        )
        
        messages = [
            SystemMessage(content="You are a Kaggle Grandmaster implementing improvements."),
            HumanMessage(content=prompt),
        ]
        
        response = self.model.invoke(messages)
        improved_block = self._extract_code(response.content)
        
        if not improved_block:
            logger.warning("Failed to generate improved code block")
            return state.current_code, None
        
        # Replace the code block in the full code
        improved_code = state.current_code
        if plan.code_block in state.current_code:
            improved_code = state.current_code.replace(
                plan.code_block, improved_block
            )
        else:
            # Try fuzzy replacement
            logger.warning("Exact code block not found, trying fuzzy match")
            improved_code = state.current_code  # Keep original if can't match
        
        # Record the improved block
        state.improved_code_blocks.append(improved_block)
        
        # Execute improved code
        logger.info("  Testing improved code...")
        exec_result = self.executor.execute(
            improved_code,
            working_dir=competition_dir,
        )
        
        if not exec_result.success:
            logger.warning(f"Improved code failed: {exec_result.error}")
            return state.current_code, None
        
        # Parse training feedback
        feedback = parse_training_logs(exec_result.stdout)
        
        return improved_code, feedback
    
    def _refine_plan(
        self,
        state: AblationState,
        current_plan: ImprovementPlan,
        last_result: TrainingFeedback | None,
    ) -> ImprovementPlan:
        """Refine the improvement plan based on results."""
        
        # Format previous plan results
        prev_summary = ""
        for plan in state.improvement_plans[-3:]:
            status = "✅" if plan.implemented else "❌"
            score = f"{plan.result_score:.4f}" if plan.result_score else "N/A"
            prev_summary += f"{status} Plan: {plan.plan[:100]}... | Score: {score}\n"
        
        prompt = PLAN_REFINEMENT_PROMPT.format(
            code_block=current_plan.code_block,
            prev_plan_summary=prev_summary,
        )
        
        messages = [
            SystemMessage(content="You are a Kaggle Grandmaster refining improvement plans."),
            HumanMessage(content=prompt),
        ]
        
        response = self.model.invoke(messages)
        
        # Create refined plan
        refined_plan = ImprovementPlan(
            code_block=current_plan.code_block,
            plan=response.content.strip(),
            expected_impact=current_plan.expected_impact * 0.9,  # Slightly lower expectations
            risk_level=current_plan.risk_level,
        )
        
        state.improvement_plans.append(refined_plan)
        
        return refined_plan
    
    def _extract_code(self, content: str) -> str:
        """Extract Python code from response content."""
        # Try to find code block
        code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Try without language tag
        code_match = re.search(r"```\n(.*?)```", content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Return content as-is if it looks like code
        if "import " in content or "def " in content:
            return content.strip()
        
        return ""


def create_ablation_agent(model, **kwargs) -> AblationStudyAgent:
    """
    Create an ablation study agent.
    
    Args:
        model: LangChain chat model
        **kwargs: Additional configuration
        
    Returns:
        AblationStudyAgent instance
    """
    return AblationStudyAgent(model, **kwargs)

