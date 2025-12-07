"""LLM utility functions for handling response content."""

from typing import Any


def get_text_content(content: Any) -> str:
    """Extract text from LLM response content.

    Handles both string responses and list responses from OpenAI Responses API.
    When use_responses_api=True, ChatOpenAI may return response.content as a list
    of content blocks instead of a plain string.

    Args:
        content: The response.content from an LLM call (string or list)

    Returns:
        Extracted text as a string
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif hasattr(block, "text"):
                texts.append(block.text)
            elif isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
        return "\n".join(texts)
    return str(content)
