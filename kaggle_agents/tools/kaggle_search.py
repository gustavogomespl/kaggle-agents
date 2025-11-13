"""
Kaggle Search Tool for retrieving notebooks and discussions.

This module provides functionality to search and retrieve state-of-the-art
solutions from Kaggle competitions via the official API and web scraping.
"""

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup
from kaggle.api.kaggle_api_extended import KaggleApi

from ..core.state import SOTASolution
from ..core.config import get_config


@dataclass
class NotebookMetadata:
    """Metadata for a Kaggle notebook."""

    ref: str  # notebook reference (username/notebook-slug)
    title: str
    author: str
    total_votes: int
    medal_type: Optional[str]  # gold, silver, bronze
    language: str  # python, r
    competition: str
    url: str


@dataclass
class DiscussionMetadata:
    """Metadata for a Kaggle discussion."""

    id: int
    title: str
    author: str
    total_votes: int
    total_comments: int
    tags: List[str]
    url: str


class KaggleSearcher:
    """
    Search and retrieve content from Kaggle competitions.

    This class provides methods to:
    - Search for top notebooks in a competition
    - Search for relevant discussions
    - Download notebook source code
    - Extract code snippets and strategies
    """

    def __init__(self):
        """Initialize the Kaggle searcher with API client."""
        self.api = KaggleApi()
        self.api.authenticate()
        self.config = get_config()

    def search_notebooks(
        self,
        competition: str,
        sort_by: str = "voteCount",
        page_size: int = 20,
        language: str = "python",
    ) -> List[NotebookMetadata]:
        """
        Search for notebooks in a competition.

        Args:
            competition: Competition name/slug
            sort_by: Sort order (voteCount, dateCreated, scoreAscending, scoreDescending)
            page_size: Number of results to return
            language: Programming language filter (python, r, all)

        Returns:
            List of notebook metadata
        """
        try:
            # Use Kaggle API to list kernels
            kernels = self.api.kernels_list(
                competition=competition,
                sort_by=sort_by,
                page_size=min(page_size, 100),  # API limit
                language=language if language != "all" else None,
            )

            notebooks = []
            for kernel in kernels:
                # Extract metadata
                metadata = NotebookMetadata(
                    ref=kernel.ref,
                    title=kernel.title,
                    author=kernel.author,
                    total_votes=kernel.totalVotes,
                    medal_type=getattr(kernel, 'medalType', None),
                    language=kernel.language,
                    competition=competition,
                    url=f"https://www.kaggle.com/code/{kernel.ref}",
                )
                notebooks.append(metadata)

            return notebooks[:page_size]

        except Exception as e:
            print(f"  Error searching notebooks: {e}")
            return []

    def download_notebook(
        self,
        notebook_ref: str,
        output_dir: Path | str,
    ) -> Optional[Path]:
        """
        Download notebook source code.

        Args:
            notebook_ref: Notebook reference (username/notebook-slug)
            output_dir: Directory to save the notebook

        Returns:
            Path to downloaded notebook file, or None if failed
        """
        output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Download kernel
            self.api.kernels_pull(
                notebook_ref,
                path=str(output_path),
            )

            # Find downloaded file
            notebook_files = list(output_path.glob("*.ipynb")) + list(output_path.glob("*.py"))
            if notebook_files:
                return notebook_files[0]

            return None

        except Exception as e:
            print(f"  Error downloading notebook {notebook_ref}: {e}")
            return None

    def extract_code_from_notebook(self, notebook_path: Path) -> List[str]:
        """
        Extract code cells from a Jupyter notebook.

        Args:
            notebook_path: Path to .ipynb file

        Returns:
            List of code snippets
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_data = json.load(f)

            code_snippets = []
            for cell in notebook_data.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        code = ''.join(source)
                    else:
                        code = source

                    # Skip empty cells and magic commands
                    if code.strip() and not code.strip().startswith('%'):
                        code_snippets.append(code)

            return code_snippets

        except Exception as e:
            print(f"  Error extracting code from {notebook_path}: {e}")
            return []

    def extract_code_from_script(self, script_path: Path) -> List[str]:
        """
        Extract code sections from a Python script.

        Args:
            script_path: Path to .py file

        Returns:
            List of code snippets (split by major sections)
        """
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by major comments (### or more #)
            sections = re.split(r'\n#{3,}.*?\n', content)

            # Filter out empty sections
            code_snippets = [s.strip() for s in sections if s.strip()]

            return code_snippets

        except Exception as e:
            print(f"  Error extracting code from {script_path}: {e}")
            return []

    def analyze_notebook_strategies(self, code_snippets: List[str]) -> Dict[str, Any]:
        """
        Analyze code to extract ML strategies and approaches.

        Args:
            code_snippets: List of code snippets from notebook

        Returns:
            Dictionary with extracted strategies
        """
        strategies = {
            "models_used": [],
            "feature_engineering": [],
            "ensemble_approach": None,
        }

        all_code = "\n".join(code_snippets)

        # Detect models
        model_patterns = {
            "XGBoost": r"xgboost|XGB|xgb\.train|XGBClassifier|XGBRegressor",
            "LightGBM": r"lightgbm|lgbm|lgb\.train|LGBMClassifier|LGBMRegressor",
            "CatBoost": r"catboost|CatBoost|CatBoostClassifier|CatBoostRegressor",
            "RandomForest": r"RandomForest|RandomForestClassifier|RandomForestRegressor",
            "Neural Network": r"keras|tensorflow|torch|nn\.Module|Sequential",
            "Linear Models": r"LinearRegression|LogisticRegression|Ridge|Lasso",
        }

        for model_name, pattern in model_patterns.items():
            if re.search(pattern, all_code, re.IGNORECASE):
                strategies["models_used"].append(model_name)

        # Detect feature engineering techniques
        feature_patterns = {
            "Target Encoding": r"TargetEncoder|target_encode",
            "One-Hot Encoding": r"OneHotEncoder|get_dummies",
            "Polynomial Features": r"PolynomialFeatures",
            "Feature Scaling": r"StandardScaler|MinMaxScaler|RobustScaler",
            "Feature Selection": r"SelectKBest|RFE|feature_importances",
            "PCA": r"PCA\(",
            "Time Features": r"dt\.year|dt\.month|dt\.day|dt\.hour",
        }

        for feat_name, pattern in feature_patterns.items():
            if re.search(pattern, all_code, re.IGNORECASE):
                strategies["feature_engineering"].append(feat_name)

        # Detect ensemble methods
        if re.search(r"VotingClassifier|VotingRegressor", all_code):
            strategies["ensemble_approach"] = "Voting"
        elif re.search(r"StackingClassifier|StackingRegressor", all_code):
            strategies["ensemble_approach"] = "Stacking"
        elif re.search(r"\.mean\(axis=|average.*predictions", all_code):
            strategies["ensemble_approach"] = "Averaging"
        elif re.search(r"weighted.*mean|weights.*predictions", all_code):
            strategies["ensemble_approach"] = "Weighted Averaging"

        return strategies

    def search_discussions(
        self,
        competition: str,
        max_results: int = 10,
    ) -> List[DiscussionMetadata]:
        """
        Search for discussions in a competition.

        Args:
            competition: Competition name/slug
            max_results: Maximum number of discussions to retrieve

        Returns:
            List of discussion metadata
        """
        discussions = []

        try:
            # Web scraping approach (Kaggle API doesn't provide discussion search)
            url = f"https://www.kaggle.com/competitions/{competition}/discussion"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find discussion elements (this may need adjustment based on Kaggle's HTML structure)
            # Note: This is a simplified example and may need to be updated
            discussion_items = soup.find_all('div', class_='topic-list-item', limit=max_results)

            for item in discussion_items[:max_results]:
                try:
                    # Extract metadata (adjust selectors as needed)
                    title_elem = item.find('a', class_='topic-title')
                    votes_elem = item.find('span', class_='vote-count')

                    if title_elem:
                        discussion = DiscussionMetadata(
                            id=hash(title_elem.get('href', '')),
                            title=title_elem.text.strip(),
                            author='',  # Would need additional parsing
                            total_votes=int(votes_elem.text) if votes_elem else 0,
                            total_comments=0,  # Would need additional parsing
                            tags=[],
                            url=f"https://www.kaggle.com{title_elem.get('href', '')}",
                        )
                        discussions.append(discussion)

                except Exception as e:
                    print(f"  Error parsing discussion item: {e}")
                    continue

        except Exception as e:
            print(f"  Error searching discussions: {e}")

        return discussions

    def create_sota_solution(
        self,
        notebook_metadata: NotebookMetadata,
        code_snippets: List[str],
        strategies: Dict[str, Any],
    ) -> SOTASolution:
        """
        Create a SOTASolution object from notebook data.

        Args:
            notebook_metadata: Notebook metadata
            code_snippets: Extracted code snippets
            strategies: Extracted strategies

        Returns:
            SOTASolution object
        """
        return SOTASolution(
            source=notebook_metadata.ref,
            title=notebook_metadata.title,
            score=0.0,  # Score not available from API
            votes=notebook_metadata.total_votes,
            code_snippets=code_snippets,
            strategies=[f"{k}: {v}" for k, v in strategies.items() if v],
            models_used=strategies.get("models_used", []),
            feature_engineering=strategies.get("feature_engineering", []),
            ensemble_approach=strategies.get("ensemble_approach"),
        )


# ==================== Convenience Functions ====================

def search_competition_notebooks(
    competition: str,
    max_notebooks: int = 10,
    min_votes: int = 5,
) -> List[SOTASolution]:
    """
    Search and analyze top notebooks for a competition.

    Args:
        competition: Competition name
        max_notebooks: Maximum number of notebooks to analyze
        min_votes: Minimum votes threshold

    Returns:
        List of SOTASolution objects
    """
    searcher = KaggleSearcher()

    # Search notebooks
    print(f"=
 Searching notebooks for {competition}...")
    notebooks = searcher.search_notebooks(competition, page_size=max_notebooks * 2)

    # Filter by votes
    notebooks = [nb for nb in notebooks if nb.total_votes >= min_votes][:max_notebooks]

    print(f"= Found {len(notebooks)} high-quality notebooks")

    # Download and analyze
    solutions = []
    config = get_config()
    download_dir = config.paths.cache_dir / "notebooks" / competition

    for nb in notebooks:
        print(f"  = Analyzing: {nb.title} ({nb.total_votes} votes)")

        # Download notebook
        nb_path = searcher.download_notebook(nb.ref, download_dir)
        if not nb_path:
            continue

        # Extract code
        if nb_path.suffix == ".ipynb":
            code_snippets = searcher.extract_code_from_notebook(nb_path)
        else:
            code_snippets = searcher.extract_code_from_script(nb_path)

        # Analyze strategies
        strategies = searcher.analyze_notebook_strategies(code_snippets)

        # Create SOTA solution
        solution = searcher.create_sota_solution(nb, code_snippets, strategies)
        solutions.append(solution)

        # Rate limiting
        time.sleep(1)

    print(f" Successfully analyzed {len(solutions)} notebooks")
    return solutions
