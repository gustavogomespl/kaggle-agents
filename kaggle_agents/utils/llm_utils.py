"""LLM utility functions for handling response content."""

import logging
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if the exception looks like a rate-limit / quota error."""
    msg = str(exc).lower()
    rate_limit_markers = [
        "429",
        "rate limit",
        "rate_limit",
        "resource_exhausted",
        "quota exceeded",
        "too many requests",
    ]
    return any(marker in msg for marker in rate_limit_markers)


@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=5, max=120),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        "Rate limited. Retrying in %.1fs (attempt %d/5)...",
        retry_state.next_action.sleep,  # type: ignore[union-attr]
        retry_state.attempt_number,
    ),
)
def invoke_with_retry(llm: Any, messages: list) -> Any:
    """Invoke an LLM with automatic retry on rate-limit errors.

    Uses exponential backoff: 5s min, 120s max, up to 5 attempts.
    Non-rate-limit errors are re-raised immediately.
    """
    return llm.invoke(messages)


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
