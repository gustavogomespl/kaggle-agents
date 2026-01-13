"""Data format discovery node for the Kaggle Agents workflow."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ...core.state import KaggleState
from ...tools.data_format_discovery import DataFormatDiscoverer, detect_traditional_format


def data_format_discovery_node(state: KaggleState) -> dict[str, Any]:
    """
    Intelligent data format discovery with fallback mechanism.

    This node acts as a fallback when traditional CSV detection fails.
    It fetches information from the competition's Kaggle page and uses
    an LLM to generate adaptive parsing instructions.

    Args:
        state: Current state

    Returns:
        State updates with data format information and parsing instructions
    """
    print("\n" + "=" * 60)
    print("= DATA FORMAT DISCOVERY")
    print("=" * 60)

    working_dir = Path(state["working_directory"])
    competition_info = state["competition_info"]
    competition = competition_info.name

    # Step 1: Try traditional detection first
    print("\n   Checking for standard CSV format...")
    traditional_files = detect_traditional_format(working_dir)

    if traditional_files:
        print("   ✓ Traditional CSV format detected")
        print(f"     Train: {traditional_files.get('train', 'N/A')}")
        print(f"     Test: {traditional_files.get('test', 'N/A')}")
        return {
            "data_format_type": "traditional",
            "last_updated": datetime.now(),
        }

    # Step 2: Fallback - discover format from multiple sources
    print("\n   ⚠️  Non-standard format detected, initiating discovery...")

    discoverer = DataFormatDiscoverer()

    # Gather information from multiple sources
    print("   📄 Fetching competition data page...")
    data_page_content = discoverer.fetch_data_page(competition)

    print("   📁 Listing data files...")
    file_listing = discoverer.list_data_files(working_dir)

    print("   🔍 Analyzing SOTA notebooks for data loading patterns...")
    sota_loading_code = discoverer.analyze_sota_data_loading(competition, max_notebooks=3)

    context = {
        "competition": competition,
        "data_page_content": data_page_content,
        "file_listing": file_listing,
        "description": competition_info.description or "",
        "sota_loading_code": sota_loading_code,
    }

    # Step 3: Use LLM to generate parsing instructions
    print("   🤖 Generating parsing instructions with LLM...")

    from ...core.config import get_llm_for_role

    try:
        llm = get_llm_for_role(role="planner", temperature=0.0)
        parsing_info = discoverer.generate_parsing_instructions(llm, context)

        print(f"   ✓ Format type: {parsing_info.get('format_type', 'unknown')}")
        print(f"   ✓ ID column: {parsing_info.get('id_column', 'unknown')}")
        print(f"   ✓ Target column: {parsing_info.get('target_column', 'unknown')}")

        if parsing_info.get("notes"):
            print(f"   📝 Notes: {parsing_info.get('notes')}")

    except Exception as e:
        print(f"   ⚠️  LLM parsing failed: {e}")
        parsing_info = {
            "format_type": "unknown",
            "id_column": "unknown",
            "target_column": "unknown",
            "loading_code": "",
            "can_generate_csv": False,
            "error": str(e),
        }

    # Step 4: Pass parsing instructions to developer agent
    # NOTE: We intentionally do NOT execute the LLM-generated loading code here.
    # The code will be incorporated into component code by the developer agent
    # and executed through the sandboxed CodeExecutor for security.
    updates: dict[str, Any] = {
        "data_format_type": parsing_info.get("format_type", "custom"),
        "parsing_info": parsing_info,
        "last_updated": datetime.now(),
    }

    if parsing_info.get("loading_code"):
        print("\n   📝 Passing loading code to developer agent (will run in sandbox)")
        updates["data_loading_code"] = parsing_info.get("loading_code", "")
    else:
        print("\n   ⚠️  No loading code generated - developer will need to infer format")

    return updates
