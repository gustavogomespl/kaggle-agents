"""
Explainability Agent for didactic explanation of system decisions.

This agent analyzes the entire workflow history to generate a comprehensive,
didactic report explaining WHY certain decisions were made, suitable for
educational purposes and TCC documentation.
"""

from typing import Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.state import KaggleState
from ..core.config import get_config

class ExplainabilityAgent:
    """
    Agent that produces didactic explanations of the solving process.
    """

    def __init__(self):
        self.config = get_config()
        # Using the same advanced model as Meta-Evaluator for high-quality explanations
        self.llm = ChatOpenAI(
            model="gpt-5.1",
            temperature=0.7,
            max_tokens=4096,
        )

    def __call__(self, state: KaggleState) -> Dict[str, Any]:
        """
        Generate explanation report.

        Args:
            state: Current workflow state

        Returns:
            State updates with explanation report
        """
        print("\n" + "="*60)
        print("= EXPLAINABILITY AGENT: Generating Didactic Report")
        print("="*60)

        # Gather context
        competition_info = state.get("competition_info")
        domain = state.get("domain_detected", "unknown")
        ablation_plan = state.get("ablation_plan", [])
        dev_results = state.get("development_results", [])
        ensemble = state.get("best_model", {})
        meta_eval = state.get("failure_analysis", {})

        # Build context string
        context = self._build_context(
            competition_info, domain, ablation_plan, dev_results, ensemble, meta_eval
        )

        # Generate report
        report = self._generate_report(context)

        # Save report to file
        self._save_report(state.get("working_directory", "."), report)

        return {
            "explanation_report": report,
            "last_updated": datetime.now(),
        }

    def _build_context(self, competition_info, domain, ablation_plan, dev_results, ensemble, meta_eval) -> str:
        """Build context for the LLM."""
        
        # Format ablation plan
        plan_str = ""
        for comp in ablation_plan:
            plan_str += f"- {comp.name} ({comp.component_type}): Impact {comp.estimated_impact}\n"
            plan_str += f"  Reasoning: {comp.reasoning}\n"

        # Format results
        results_str = ""
        for res in dev_results:
            status = "✅ Success" if res.success else "❌ Failed"
            results_str += f"- {status}: Time {res.execution_time:.2f}s\n"

        return f"""
        Competition: {competition_info.name if competition_info else 'Unknown'}
        Domain: {domain}
        Problem Type: {competition_info.problem_type if competition_info else 'Unknown'}
        
        ## Strategy Plan
        {plan_str}
        
        ## Execution Results
        {results_str}
        
        ## Ensemble Strategy
        Model: {ensemble.get('name', 'None')}
        Score: {ensemble.get('mean_cv_score', 'N/A')}
        
        ## Meta-Evaluation (Self-Correction)
        Success Patterns: {', '.join(meta_eval.get('success_patterns', []))}
        Error Patterns: {', '.join(meta_eval.get('error_patterns', []))}
        """

    def _generate_report(self, context: str) -> str:
        """Generate the didactic report using LLM."""
        
        system_prompt = """You are an expert Data Science Professor explaining an automated solution.
        Your goal is to write a DIDACTIC REPORT explaining the decisions made by the AI system.
        
        Structure your report as follows:
        1. **Problem Analysis**: Why did the system classify this as {domain}? What are the challenges?
        2. **Strategy Selection**: Why were these specific models/techniques chosen in the plan?
        3. **Execution & Challenges**: What went wrong? How did the system correct itself? (Refer to Meta-Evaluation)
        4. **Final Solution**: Explain the final ensemble strategy. Why is it better than single models?
        5. **Key Takeaways**: What can a human learn from this automated run?
        
        Tone: Academic, insightful, encouraging. Use markdown formatting."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze this run context and generate the report:\n\n{context}"),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _save_report(self, working_dir: str, report: str):
        """Save the report to a markdown file."""
        import os
        path = os.path.join(working_dir, "DECISION_EXPLANATION.md")
        with open(path, "w") as f:
            f.write(report)
        print(f"\n📝 Report saved to: {path}")

def explainability_agent_node(state: KaggleState) -> Dict[str, Any]:
    """LangGraph node for explainability agent."""
    agent = ExplainabilityAgent()
    return agent(state)
