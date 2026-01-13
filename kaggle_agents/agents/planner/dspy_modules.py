"""DSPy signatures and modules for ablation planning."""

import dspy


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
