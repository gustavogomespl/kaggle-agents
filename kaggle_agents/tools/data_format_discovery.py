"""
Data Format Discovery Tool.

Discovers data format from competition page and generates parsing code
for non-standard Kaggle competition data formats.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from bs4 import BeautifulSoup

from .kaggle_search import KaggleSearcher

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class DataFormatDiscoverer:
    """
    Discovers data format from competition page and generates parsing code.

    This tool acts as a fallback mechanism when traditional CSV detection fails.
    It fetches information from multiple sources and uses an LLM to generate
    adaptive parsing instructions.
    """

    def __init__(self):
        """Initialize the discoverer."""
        self._searcher = None

    @property
    def searcher(self) -> KaggleSearcher:
        """Lazy-load KaggleSearcher."""
        if self._searcher is None:
            self._searcher = KaggleSearcher()
        return self._searcher

    def fetch_data_page(self, competition_slug: str) -> str:
        """
        Fetch data format info from competition's data page.

        Args:
            competition_slug: Competition URL suffix (e.g., 'mlsp-2013-birds')

        Returns:
            Extracted text content from the data page
        """
        urls_to_try = [
            f"https://www.kaggle.com/competitions/{competition_slug}/data",
            f"https://www.kaggle.com/c/{competition_slug}/data",
        ]

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        for url in urls_to_try:
            try:
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Remove script and style elements
                    for element in soup(["script", "style", "nav", "header", "footer"]):
                        element.decompose()

                    # Extract main content
                    content_parts = []

                    # Look for data description sections
                    for selector in [
                        "div[class*='data']",
                        "div[class*='description']",
                        "div[class*='content']",
                        "article",
                        "main",
                    ]:
                        elements = soup.select(selector)
                        for elem in elements:
                            text = elem.get_text(separator="\n", strip=True)
                            if len(text) > 100:  # Only meaningful content
                                content_parts.append(text)

                    # If no specific sections found, get body text
                    if not content_parts:
                        body = soup.find("body")
                        if body:
                            content_parts.append(body.get_text(separator="\n", strip=True))

                    # Deduplicate and join
                    seen = set()
                    unique_parts = []
                    for part in content_parts:
                        if part not in seen:
                            seen.add(part)
                            unique_parts.append(part)

                    return "\n\n".join(unique_parts)[:10000]  # Limit size

            except Exception as e:
                print(f"  Warning: Could not fetch {url}: {e}")
                continue

        return ""

    def list_data_files(self, data_dir: Path) -> list[dict[str, Any]]:
        """
        List all files in data directory with metadata.

        Args:
            data_dir: Path to data directory

        Returns:
            List of file info dictionaries with name, extension, size, sample content
        """
        files_info = []

        if not data_dir.exists():
            return files_info

        # Recursively find all files
        for file_path in data_dir.rglob("*"):
            if not file_path.is_file():
                continue

            # Skip hidden files and common non-data files
            if file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() in [".pyc", ".pyo", ".so", ".dll"]:
                continue

            file_info = {
                "name": str(file_path.relative_to(data_dir)),
                "extension": file_path.suffix.lower(),
                "size_bytes": file_path.stat().st_size,
                "sample_content": "",
            }

            # Read sample content for text-like files
            text_extensions = {".txt", ".csv", ".tsv", ".json", ".md", ".xml", ".yaml", ".yml"}
            if file_path.suffix.lower() in text_extensions:
                try:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        lines = []
                        for i, line in enumerate(f):
                            if i >= 10:  # First 10 lines
                                break
                            lines.append(line.rstrip()[:200])  # Limit line length
                        file_info["sample_content"] = "\n".join(lines)
                except Exception:
                    pass

            files_info.append(file_info)

        # Sort by size (larger files first, usually more important)
        files_info.sort(key=lambda x: x["size_bytes"], reverse=True)

        return files_info[:50]  # Limit to 50 files

    def analyze_sota_data_loading(self, competition: str, max_notebooks: int = 3) -> list[str]:
        """
        Analyze SOTA notebooks to extract data loading patterns.

        Args:
            competition: Competition name/slug
            max_notebooks: Maximum notebooks to analyze

        Returns:
            List of data loading code snippets from notebooks
        """
        loading_patterns = []

        try:
            # Search for top notebooks
            notebooks = self.searcher.search_notebooks(
                competition, sort_by="voteCount", page_size=max_notebooks * 2
            )

            # Filter to get top ones
            notebooks = [nb for nb in notebooks if nb.total_votes >= 1][:max_notebooks]

            if not notebooks:
                return loading_patterns

            # Download and analyze each notebook
            from ..core.config import get_config

            config = get_config()
            download_dir = config.paths.cache_dir / "notebooks" / competition

            for nb in notebooks:
                try:
                    nb_path = self.searcher.download_notebook(nb.ref, download_dir)
                    if not nb_path:
                        continue

                    # Extract code
                    if nb_path.suffix == ".ipynb":
                        code_snippets = self.searcher.extract_code_from_notebook(nb_path)
                    else:
                        code_snippets = self.searcher.extract_code_from_script(nb_path)

                    # Find data loading code (look for file reads, path definitions)
                    loading_keywords = [
                        r"read_csv",
                        r"open\(",
                        r"pd\.read",
                        r"np\.load",
                        r"Path\(",
                        r"glob\(",
                        r"os\.listdir",
                        r"train.*path",
                        r"data.*dir",
                        r"\.txt",
                        r"\.csv",
                        r"essential_data",
                        r"supplemental_data",
                    ]

                    pattern = "|".join(loading_keywords)

                    for snippet in code_snippets:
                        if re.search(pattern, snippet, re.IGNORECASE):
                            # Extract relevant lines (not the whole snippet)
                            relevant_lines = []
                            for line in snippet.split("\n"):
                                if re.search(pattern, line, re.IGNORECASE):
                                    relevant_lines.append(line.strip())
                                    # Also get a few surrounding lines for context
                                    # This is simplified - just add the matching line

                            if relevant_lines:
                                loading_patterns.append(
                                    f"# From: {nb.title}\n" + "\n".join(relevant_lines[:20])
                                )

                except Exception as e:
                    print(f"  Warning: Could not analyze notebook {nb.ref}: {e}")
                    continue

        except Exception as e:
            print(f"  Warning: Could not search notebooks: {e}")

        return loading_patterns[:10]  # Limit total snippets

    def generate_parsing_instructions(
        self,
        llm: "BaseChatModel",
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Use LLM to generate parsing instructions based on discovered context.

        Args:
            llm: Language model to use for generation
            context: Dictionary containing:
                - competition: Competition name
                - data_page_content: Content from data page
                - file_listing: List of files with metadata
                - description: Competition description
                - sota_loading_code: Code snippets from SOTA notebooks

        Returns:
            Dictionary with parsing instructions
        """
        from ..prompts.templates.data_format_prompt import DATA_FORMAT_DISCOVERY_PROMPT

        # Format file listing for prompt
        file_listing_str = ""
        for f in context.get("file_listing", [])[:20]:
            file_listing_str += f"\n- {f['name']} ({f['extension']}, {f['size_bytes']} bytes)"
            if f.get("sample_content"):
                sample = f["sample_content"][:500].replace("\n", "\n    ")
                file_listing_str += f"\n    Sample:\n    {sample}"

        # Format SOTA code
        sota_code_str = "\n\n".join(context.get("sota_loading_code", [])[:5])

        # Build prompt
        prompt = DATA_FORMAT_DISCOVERY_PROMPT.format(
            competition=context.get("competition", "unknown"),
            description=context.get("description", "")[:2000],
            data_page_content=context.get("data_page_content", "")[:3000],
            file_listing=file_listing_str,
            sota_loading_code=sota_code_str or "No SOTA notebooks found",
        )

        try:
            # Call LLM
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from response
            parsing_info = self._extract_json_from_response(content)

            # Validate required fields
            required_fields = ["format_type", "id_column", "target_column"]
            for field in required_fields:
                if field not in parsing_info:
                    parsing_info[field] = "unknown"

            # Ensure loading_code exists
            if "loading_code" not in parsing_info:
                parsing_info["loading_code"] = ""

            # Ensure can_generate_csv exists
            if "can_generate_csv" not in parsing_info:
                parsing_info["can_generate_csv"] = False

            return parsing_info

        except Exception as e:
            print(f"  Warning: LLM parsing failed: {e}")
            return {
                "format_type": "unknown",
                "id_column": "unknown",
                "target_column": "unknown",
                "loading_code": "",
                "can_generate_csv": False,
                "error": str(e),
            }

    def _extract_json_from_response(self, response: str) -> dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try to find JSON block
        json_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    json_str = match.group(1) if match.lastindex else match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        # If no JSON found, return empty dict
        return {}


def detect_traditional_format(working_dir: Path) -> dict[str, str] | None:
    """
    Check if traditional CSV format exists.

    Args:
        working_dir: Working directory to check

    Returns:
        Dictionary with train/test paths if found, None otherwise
    """
    # Standard CSV patterns
    train_patterns = [
        "train.csv",
        "train_labels.csv",
        "training.csv",
        "train_data.csv",
    ]
    test_patterns = [
        "test.csv",
        "test_data.csv",
        "testing.csv",
    ]

    # Check root directory
    train_path = None
    test_path = None

    for pattern in train_patterns:
        path = working_dir / pattern
        if path.exists():
            train_path = str(path)
            break

    for pattern in test_patterns:
        path = working_dir / pattern
        if path.exists():
            test_path = str(path)
            break

    # Check if we found both
    if train_path and test_path:
        return {"train": train_path, "test": test_path}

    # Check for train/test directories with common structures
    train_dir = working_dir / "train"
    test_dir = working_dir / "test"

    if train_dir.exists() and train_dir.is_dir() and test_dir.exists() and test_dir.is_dir():
        # Check for image/audio files
        train_files = list(train_dir.rglob("*.*"))
        if train_files:
            extensions = {f.suffix.lower() for f in train_files[:100]}
            media_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".wav", ".mp3", ".flac"}
            if extensions & media_extensions:
                return {"train": str(train_dir), "test": str(test_dir)}

    return None


def get_loading_code_for_developer(parsing_info: dict[str, Any]) -> str:
    """
    Format the loading code from parsing_info for inclusion in developer prompts.

    This function does NOT execute the code - it only formats it for the developer
    agent to incorporate into component code, where it will be executed through
    the sandboxed CodeExecutor.

    Args:
        parsing_info: Parsing instructions from LLM

    Returns:
        Formatted loading code string, or empty string if not available
    """
    loading_code = parsing_info.get("loading_code", "")

    if not loading_code:
        return ""

    # Return the code as-is - it will be executed by the developer agent
    # through the sandboxed CodeExecutor, not here
    return loading_code
