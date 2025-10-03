"""Prompt templates for agents."""

from .prompt_base import *
from .prompt_planner import *
from .prompt_developer import *
from .prompt_reviewer import *
from .prompt_reader import *

__all__ = [
    # Base prompts
    "AGENT_ROLE_TEMPLATE",
    "PROMPT_DATA_PREVIEW",
    "PROMPT_FEATURE_INFO",
    "PROMPT_EACH_EXPERIENCE_WITH_SUGGESTION",
    "PROMPT_REORGANIZE_JSON",
    "PROMPT_REORGANIZE_EXTRACT_TOOLS",

    # Planner prompts
    "PROMPT_PLANNER",
    "PROMPT_PLANNER_TASK",
    "PROMPT_PLANNER_TOOLS",
    "PROMPT_PLANNER_REORGANIZE_IN_MARKDOWN",
    "PROMPT_PLANNER_REORGANIZE_IN_JSON",

    # Developer prompts
    "PROMPT_DEVELOPER",
    "PROMPT_DEVELOPER_TASK",
    "PROMPT_EXTRACT_TOOLS",
    "PROMPT_FIX_CODE",

    # Reviewer prompts
    "PROMPT_REVIEWER",
    "PROMPT_REVIEWER_TASK",

    # Reader prompts
    "PROMPT_READER",
    "PROMPT_READER_TASK",
]
