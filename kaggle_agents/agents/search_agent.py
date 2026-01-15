"""
Search Agent for SOTA Solution Retrieval.

This agent implements the "Search-First Strategy" from Google ADK,
retrieving and analyzing state-of-the-art solutions before generating code.
"""

import json
from datetime import datetime
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ..core.config import get_config
from ..core.state import KaggleState, SOTASolution
from ..tools.kaggle_search import search_competition_notebooks
from ..utils.llm_utils import get_text_content


def calculate_adaptive_k(
    current_iteration: int,
    iteration_memory: list | None = None,
    base_k: int = 5,
    expanded_k: int = 10,
) -> int:
    """
    Calculate number of notebooks to search based on iteration and improvement trend.

    Strategy:
    - Iteration 1-2: Base search (top 5)
    - Iteration 3+: Evaluate improvement trend
    - If stagnating (trend < 0.01): Expand to top 10

    Args:
        current_iteration: Current iteration number (1-indexed)
        iteration_memory: List of iteration memories with score_improvement
        base_k: Base number of notebooks (default 5)
        expanded_k: Expanded number on stagnation (default 10)

    Returns:
        Number of notebooks to search
    """
    # Early iterations: use base k
    if current_iteration <= 2:
        return base_k

    # Iteration 3+: evaluate improvement trend
    if iteration_memory and len(iteration_memory) >= 2:
        # Get recent improvements (last 3 iterations)
        recent_improvements = []
        for mem in iteration_memory[-3:]:
            if hasattr(mem, "score_improvement"):
                recent_improvements.append(mem.score_improvement)
            elif isinstance(mem, dict) and "score_improvement" in mem:
                recent_improvements.append(mem["score_improvement"])

        if recent_improvements:
            trend = sum(recent_improvements) / len(recent_improvements)

            # Stagnation threshold
            if trend < 0.01:
                print(f"   📈 Low improvement trend ({trend:.4f}), expanding search to top {expanded_k}")
                return expanded_k

    return base_k


class SearchAgent:
    """
    Agent responsible for retrieving state-of-the-art solutions.

    Strategy (inspired by Google ADK MLE-STAR):
    1. Search for top-voted notebooks in the competition
    2. Download and analyze code
    3. Extract strategies, models, and techniques
    4. Rank solutions by relevance and quality
    5. Return top solutions for merging/adaptation
    """

    def __init__(self):
        """Initialize the search agent."""
        self.config = get_config()

        # Initialize LLM for analysis
        if self.config.llm.provider == "openai":
            self.llm = ChatOpenAI(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                use_responses_api=self.config.llm.use_responses_api,
            )
        elif self.config.llm.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm = ChatGoogleGenerativeAI(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
            )
        else:  # anthropic
            self.llm = ChatAnthropic(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
            )

    def __call__(self, state: KaggleState) -> dict[str, Any]:
        """
        Execute the search agent.

        Args:
            state: Current workflow state

        Returns:
            State updates with SOTA solutions
        """
        print("\n" + "=" * 60)
        print("SEARCH AGENT: Retrieving SOTA Solutions")
        print("=" * 60)

        competition_name = state["competition_info"].name
        state.get("domain_detected", "tabular")

        # Get current iteration and memory for adaptive k
        current_iteration = state.get("current_iteration", 1)
        iteration_memory = state.get("iteration_memory", [])

        # 0. Calculate adaptive top-K
        adaptive_k = calculate_adaptive_k(
            current_iteration=current_iteration,
            iteration_memory=iteration_memory,
            base_k=self.config.search.max_notebooks or 5,
            expanded_k=10,
        )
        print(f"\n= Adaptive search: top {adaptive_k} solutions (iteration {current_iteration})")

        # 1. Generate search queries
        search_queries = self._generate_search_queries(state)
        print(f"\n= Generated {len(search_queries)} search queries:")
        for i, query in enumerate(search_queries, 1):
            print(f"  {i}. {query}")

        # 2. Search for notebooks
        print(f"\n= Searching notebooks for: {competition_name}")
        sota_solutions = search_competition_notebooks(
            competition=competition_name,
            max_notebooks=adaptive_k,  # Use adaptive k instead of fixed config
            min_votes=self.config.search.min_votes,
        )

        # 2b. Fallback if no notebooks found (old competitions, niche domains)
        if not sota_solutions:
            print("   WARNING: No notebooks found for this competition")
            print("   Generating fallback SOTA guidance from domain heuristics...")
            sota_solutions = self._generate_fallback_sota(state)

        # 3. Rank and filter solutions
        ranked_solutions = self._rank_solutions(sota_solutions, state)

        # 4. Enhance with LLM analysis
        enhanced_solutions = self._enhance_with_llm_analysis(ranked_solutions, state)

        # 5. Print summary
        self._print_summary(enhanced_solutions)

        # Return state updates
        return {
            "sota_solutions": enhanced_solutions,
            "search_queries_used": search_queries,
            "sota_retrieval_k": adaptive_k,  # Track how many solutions were searched
            "last_sota_update_iteration": current_iteration,  # Track when SOTA was updated
            "last_updated": datetime.now(),
        }

    def _generate_search_queries(self, state: KaggleState) -> list[str]:
        """
        Generate search queries based on competition info and domain.

        Args:
            state: Current state

        Returns:
            List of search query strings
        """
        competition_info = state["competition_info"]
        domain = state.get("domain_detected", "tabular")

        queries = [
            f"{competition_info.name} winning solution",
            f"{competition_info.name} gold medal",
            f"{competition_info.name} top solution",
        ]

        # Domain-specific queries
        domain_queries = {
            "tabular": [
                f"{competition_info.name} xgboost",
                f"{competition_info.name} lightgbm",
                f"{competition_info.name} feature engineering",
            ],
            "computer_vision": [
                f"{competition_info.name} resnet",
                f"{competition_info.name} efficientnet",
                f"{competition_info.name} data augmentation",
            ],
            "nlp": [
                f"{competition_info.name} bert",
                f"{competition_info.name} transformer",
                f"{competition_info.name} fine-tuning",
            ],
            "time_series": [
                f"{competition_info.name} forecasting",
                f"{competition_info.name} lstm",
                f"{competition_info.name} time series",
            ],
        }

        queries.extend(domain_queries.get(domain, []))

        return queries

    def _rank_solutions(
        self,
        solutions: list[SOTASolution],
        state: KaggleState,
    ) -> list[SOTASolution]:
        """
        Rank solutions by relevance and quality.

        Ranking factors:
        - Vote count (popularity)
        - Recency (implicit in Kaggle's ordering)
        - Completeness (has strategies, models, code)
        - Domain relevance

        Args:
            solutions: List of SOTA solutions
            state: Current state

        Returns:
            Ranked list of solutions
        """

        def score_solution(sol: SOTASolution) -> float:
            score = 0.0

            # Vote count (normalized, max 100 points)
            score += min(sol.votes / 10, 100)

            # Has code snippets
            score += len(sol.code_snippets) * 5

            # Has models
            score += len(sol.models_used) * 10

            # Has feature engineering
            score += len(sol.feature_engineering) * 8

            # Has ensemble approach
            if sol.ensemble_approach:
                score += 20

            return score

        # Score and sort
        scored = [(score_solution(sol), sol) for sol in solutions]
        scored.sort(reverse=True, key=lambda x: x[0])

        # Return top solutions
        return [sol for _, sol in scored]

    def _enhance_with_llm_analysis(
        self,
        solutions: list[SOTASolution],
        state: KaggleState,
    ) -> list[SOTASolution]:
        """
        Use LLM to analyze and enhance solution descriptions.

        Args:
            solutions: List of SOTA solutions
            state: Current state

        Returns:
            Enhanced solutions with better strategy descriptions
        """
        if not solutions:
            return solutions

        print("\n   🔍 Analyzing SOTA code snippets with LLM...")

        # Analyze top 3 solutions with LLM
        for i, sol in enumerate(solutions[:3]):
            if sol.code_snippets:
                print(f"      Analyzing solution {i + 1}: {sol.title[:50]}...")
                analysis = self._analyze_code_snippets(sol)
                if analysis:
                    # Enrich solution with extracted insights
                    if analysis.get("models"):
                        # Avoid duplicates
                        new_models = [m for m in analysis["models"] if m not in sol.models_used]
                        sol.models_used.extend(new_models)
                    if analysis.get("features"):
                        new_features = [
                            f for f in analysis["features"] if f not in sol.feature_engineering
                        ]
                        sol.feature_engineering.extend(new_features)
                    if analysis.get("ensemble") and not sol.ensemble_approach:
                        sol.ensemble_approach = analysis["ensemble"]
                    if analysis.get("strategies"):
                        new_strategies = [
                            s for s in analysis["strategies"] if s not in sol.strategies
                        ]
                        sol.strategies.extend(new_strategies)

        return solutions

    def _analyze_code_snippets(self, solution: SOTASolution) -> dict[str, Any]:
        """
        Use LLM to extract insights from code snippets.

        Args:
            solution: SOTA solution with code snippets

        Returns:
            Dictionary with extracted models, features, ensemble, strategies
        """
        # Prepare code snippets (limit to first 3, truncate each to 1000 chars)
        snippets_text = "\n\n---\n\n".join(snippet[:1000] for snippet in solution.code_snippets[:3])

        prompt = f"""Analyze these Kaggle solution code snippets and extract key patterns.

Title: {solution.title}

Code Snippets:
{snippets_text}

Return a JSON object with:
- models: list of ML models/algorithms used (e.g., ["LightGBM", "XGBoost", "CatBoost"])
- features: list of feature engineering techniques (e.g., ["target encoding", "polynomial features", "lag features"])
- ensemble: ensemble strategy if any (e.g., "stacking with Ridge meta-learner", "weighted average")
- strategies: list of key strategies/tricks (e.g., ["5-fold stratified CV", "early stopping", "adversarial validation"])

Return ONLY valid JSON, no explanation or markdown."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = get_text_content(response.content).strip()

            # Parse JSON - handle markdown wrapping
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            print(f"      ⚠️ JSON parse error: {e}")
            return {}
        except Exception as e:
            print(f"      ⚠️ Code analysis failed: {e}")
            return {}

    def _generate_fallback_sota(self, state: KaggleState) -> list[SOTASolution]:
        """
        Generate synthetic SOTA solution when search returns empty.

        Used for older competitions without indexed notebooks or niche domains.

        Args:
            state: Current state with competition info

        Returns:
            List with a single fallback SOTASolution
        """
        domain = state.get("domain_detected", "tabular")

        # Normalize domain aliases
        domain_aliases = {
            "computer_vision": "image",
            "nlp": "text",
            "natural_language": "text",
        }
        domain = domain_aliases.get(domain, domain)

        # Domain-specific model defaults
        domain_models = {
            "audio": ["mel_spectrogram + CNN", "wav2vec2", "AST (Audio Spectrogram Transformer)"],
            "image": ["EfficientNet-B4", "ResNet-200D", "ConvNeXt"],
            "tabular": ["LightGBM", "XGBoost", "CatBoost"],
            "text": ["RoBERTa-large", "DeBERTa-v3", "DistilBERT"],
            "time_series": ["LightGBM", "LSTM", "N-BEATS"],
        }

        # Domain-specific feature engineering defaults
        domain_features = {
            "audio": ["Mel spectrograms", "MFCCs", "Chromagram", "Temporal features"],
            "image": ["ImageNet pretrained features", "Augmentation (flip, rotate, crop)"],
            "tabular": ["Target encoding", "Feature interactions", "Aggregations"],
            "text": ["Tokenization", "Subword embeddings", "Attention masks"],
            "time_series": ["Lag features", "Rolling statistics", "Fourier features"],
        }

        fallback = SOTASolution(
            source="fallback/domain-heuristics",
            title=f"Domain-based baseline for {domain}",
            score=0.0,
            votes=0,
            code_snippets=[],
            strategies=[
                f"Standard {domain} pipeline",
                "Start with baseline model, iterate on features",
                "Use cross-validation for robust evaluation",
            ],
            models_used=domain_models.get(domain, ["Baseline model"]),
            feature_engineering=domain_features.get(domain, ["Standard preprocessing"]),
            ensemble_approach="Weighted averaging of top models",
        )

        print(f"   Created fallback SOTA with {len(fallback.models_used)} model suggestions")
        return [fallback]

    def _print_summary(self, solutions: list[SOTASolution]) -> None:
        """
        Print a summary of found solutions.

        Args:
            solutions: List of SOTA solutions
        """
        print(f"\n= Found {len(solutions)} SOTA Solutions:")
        print("-" * 60)

        for i, sol in enumerate(solutions[:5], 1):  # Show top 5
            print(f"\n{i}. {sol.title}")
            print(f"   Source: {sol.source}")
            print(f"   Votes: {sol.votes}")
            if sol.models_used:
                print(f"   Models: {', '.join(sol.models_used)}")
            if sol.feature_engineering:
                print(f"   Features: {', '.join(sol.feature_engineering[:3])}...")
            if sol.ensemble_approach:
                print(f"   Ensemble: {sol.ensemble_approach}")
            print(f"   Code Snippets: {len(sol.code_snippets)}")

        print("\n" + "=" * 60)


# ==================== LangGraph Node Function ====================


def search_agent_node(state: KaggleState) -> dict[str, Any]:
    """
    LangGraph node function for the search agent.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = SearchAgent()
    return agent(state)
