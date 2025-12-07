"""
Search Agent for SOTA Solution Retrieval.

This agent implements the "Search-First Strategy" from Google ADK,
retrieving and analyzing state-of-the-art solutions before generating code.
"""

from datetime import datetime
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from ..core.config import get_config
from ..core.state import KaggleState, SOTASolution
from ..tools.kaggle_search import search_competition_notebooks


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
        print("\n" + "="*60)
        print("SEARCH AGENT: Retrieving SOTA Solutions")
        print("="*60)

        competition_name = state["competition_info"].name
        state.get("domain_detected", "tabular")

        # 1. Generate search queries
        search_queries = self._generate_search_queries(state)
        print(f"\n= Generated {len(search_queries)} search queries:")
        for i, query in enumerate(search_queries, 1):
            print(f"  {i}. {query}")

        # 2. Search for notebooks
        print(f"\n= Searching notebooks for: {competition_name}")
        sota_solutions = search_competition_notebooks(
            competition=competition_name,
            max_notebooks=self.config.search.max_notebooks,
            min_votes=self.config.search.min_votes,
        )

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

        # For now, return as-is
        # In future, can use LLM to summarize code snippets into strategies
        return solutions

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

        print("\n" + "="*60)


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
