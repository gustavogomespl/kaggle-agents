"""
Reporting Agent for generating explainability reports.

This agent analyzes the final state of the workflow and generates a comprehensive
report (report.md) with insights, feature importance, and model performance.
"""

from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from ..core.state import KaggleState
from ..core.config import get_config, get_llm_for_role
from langchain_core.messages import HumanMessage, SystemMessage


class ReportingAgent:
    """
    Agent responsible for generating the final explainability report.
    """

    def __init__(self):
        self.config = get_config()
        self.llm = get_llm_for_role(role="developer")  # Use developer role for now

    def __call__(self, state: KaggleState) -> Dict[str, Any]:
        """
        Generate the report.
        """
        print("\n" + "=" * 60)
        print("= REPORTING AGENT: Generating Explainability Report")
        print("=" * 60)

        working_dir = Path(state["working_directory"])
        report_path = working_dir / "report.md"

        # Gather data for the report
        context = self._gather_context(state)

        # Generate report content using LLM
        report_content = self._generate_report_content(context)

        # Save report
        with open(report_path, "w") as f:
            f.write(report_content)

        print(f"\n📄 Report generated at: {report_path}")

        return {
            "report_path": str(report_path),
            "last_updated": datetime.now(),
        }

    def _gather_context(self, state: KaggleState) -> str:
        """Gather context from state for the report."""
        competition_info = state.get("competition_info")
        best_score = state.get("best_score", 0.0)
        submissions = state.get("submissions", [])
        dev_results = state.get("development_results", [])

        # Get best submission based on metric direction
        best_sub = None
        is_minimization = False
        if submissions:
            metric = (
                competition_info.evaluation_metric.lower()
                if competition_info and competition_info.evaluation_metric
                else ""
            )
            min_metrics = ["rmse", "mae", "logloss", "error", "rmsle", "loss"]
            is_minimization = any(m in metric for m in min_metrics)

            valid_subs = [s for s in submissions if s.public_score is not None]
            if valid_subs:
                if is_minimization:
                    best_sub = min(valid_subs, key=lambda x: x.public_score)
                else:
                    best_sub = max(valid_subs, key=lambda x: x.public_score)

        context = f"""
# Competition Context
Name: {competition_info.name if competition_info else "Unknown"}
Problem Type: {competition_info.problem_type if competition_info else "Unknown"}
Metric: {competition_info.evaluation_metric if competition_info else "Unknown"}
Metric Direction: {"Minimize" if is_minimization else "Maximize"}

# Performance
Best Score: {best_sub.public_score if best_sub else best_score}
Best Public LB Score: {best_sub.public_score if best_sub else "N/A"}
Best Percentile: {best_sub.percentile if best_sub else "N/A"}%

# Components Developed
"""
        for res in dev_results:
            status = "Success" if res.success else "Failed"
            context += f"- {res.code[:50]}... ({status})\n"

        return context

    def _generate_report_content(self, context: str) -> str:
        """Generate report markdown using LLM."""

        system_prompt = """You are an expert Data Science Communicator.
Your goal is to write a clear, insightful, and educational report about the machine learning solution developed.
The report should be in Markdown format.

Structure:
# 📊 Solution Report

## 1. Executive Summary
- Brief overview of the best performing approach.
- Key metrics achieved.

## 2. Methodology
- Explain the models used (e.g., XGBoost, Neural Networks, Stacking).
- Describe key feature engineering steps.

## 3. Insights & Explainability
- What features were likely most important? (Infer from the context of tabular data).
- Why did this approach work? (e.g., "Ensembling diverse models captured non-linear patterns").

## 4. Recommendations
- What could be improved further?
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Generate the report based on this context:\n{context}"
            ),
        ]

        response = self.llm.invoke(messages)
        return response.content


def reporting_agent_node(state: KaggleState) -> Dict[str, Any]:
    agent = ReportingAgent()
    return agent(state)
