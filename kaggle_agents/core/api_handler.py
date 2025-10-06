"""API Handler with robust retry logic and error handling."""

import os
import time
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import openai
import httpx

# LangSmith tracing
try:
    from langsmith import traceable
except ImportError:
    # Fallback if langsmith not installed
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Constants
MAX_ATTEMPTS = 5
RETRY_DELAY = 30
DEFAULT_MAX_LENGTH = 100000

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class APISettings:
    """Settings for API calls with sensible defaults."""

    max_completion_tokens: int
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None

    @property
    def timeout(self) -> int:
        """Calculate timeout based on expected tokens."""
        return (self.max_completion_tokens // 1000 + 1) * 30


def load_api_config() -> Tuple[str, Optional[str]]:
    """Load API configuration from environment or file.

    Returns:
        Tuple of (api_key, base_url)

    Raises:
        ValueError: If API key is not found
    """
    # Try environment first
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if api_key:
        return api_key, base_url

    # Fallback to api_key.txt for backward compatibility
    api_key_file = Path(__file__).parent.parent.parent / "api_key.txt"

    if api_key_file.exists():
        with open(api_key_file, 'r') as f:
            api_config = f.readlines()
        api_key = api_config[0].strip()
        base_url = api_config[1].strip() if len(api_config) > 1 else None
        return api_key, base_url

    raise ValueError(
        "API key not found. Set OPENAI_API_KEY environment variable "
        "or create api_key.txt file."
    )


@traceable(name="OpenAI_API_Call", run_type="llm")
def generate_response(
    client: openai.OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    settings: APISettings,
    response_type: str = 'text'
) -> Any:
    """Generate response from OpenAI API.

    Args:
        client: OpenAI client instance
        model: Model name (e.g., 'gpt-5-mini', 'o1-mini')
        messages: List of message dictionaries
        settings: API settings
        response_type: 'text' or 'image'

    Returns:
        API response object

    Raises:
        Exception: If API call fails
    """
    logger.info(f"Generating response for model: {model}")
    start_time = time.time()

    # Special handling for models that only support default temperature
    if model in ['o1-mini', 'gpt-5-mini']:
        settings.temperature = 1.0

    try:
        if response_type == 'text':
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=settings.temperature,
                max_completion_tokens=settings.max_completion_tokens,
                top_p=settings.top_p,
                frequency_penalty=settings.frequency_penalty,
                presence_penalty=settings.presence_penalty,
                stop=settings.stop,
                timeout=settings.timeout,
            )
        elif response_type == 'image':
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=settings.temperature,
                timeout=settings.timeout,
            )
        else:
            raise ValueError(f"Unsupported response type: {response_type}")
    except Exception as e:
        logging.error(f"Error during API call: {e}")
        raise

    elapsed_time = time.time() - start_time

    # Log metrics
    if hasattr(response, 'usage'):
        usage = response.usage
        logger.info(
            f"API Call Metrics - Model: {model}, "
            f"Time: {elapsed_time:.2f}s, "
            f"Prompt Tokens: {getattr(usage, 'prompt_tokens', 0)}, "
            f"Completion Tokens: {getattr(usage, 'completion_tokens', 0)}, "
            f"Total Tokens: {getattr(usage, 'total_tokens', 0)}"
        )
    else:
        logger.info(f"Response generated in {elapsed_time:.2f} seconds")

    return response


class APIHandler:
    """Handle OpenAI API calls with retry logic and error handling."""

    def __init__(self, model: str, verify_ssl: bool = True):
        """Initialize API handler.

        Args:
            model: OpenAI model name
            verify_ssl: Whether to verify SSL certificates
        """
        self.model = model
        self.api_key, self.base_url = load_api_config()

        # Configure HTTP client
        http_client = httpx.Client(verify=verify_ssl) if not verify_ssl else None

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client
        )

    def _save_long_message(
        self,
        messages: List[Dict[str, str]],
        save_dir: Optional[Path] = None
    ):
        """Save long messages to file for debugging.

        Args:
            messages: List of messages
            save_dir: Directory to save to (default: current directory)
        """
        if save_dir is None:
            save_dir = Path.cwd()
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = save_dir / f"long_message_{timestamp}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            for message in messages:
                f.write(f"Role: {message['role']}\n")
                f.write(f"Content: {message['content']}\n\n")

        logger.info(f"Long message saved to {filename}")

    def _truncate_messages(
        self,
        messages: List[Dict[str, str]],
        max_length: int = DEFAULT_MAX_LENGTH
    ) -> List[Dict[str, str]]:
        """Truncate messages to fit within maximum length.

        Strategy: Keep all messages except the last one, then truncate
        the last message to fit within the remaining space.

        Args:
            messages: List of messages
            max_length: Maximum total character length

        Returns:
            Truncated messages
        """
        total_length = sum(len(message['content']) for message in messages)

        if total_length <= max_length:
            return messages

        # Keep all messages except the last one
        truncated = messages[:-1]
        last_message = messages[-1]

        # Calculate available space for last message
        available_length = max_length - sum(
            len(message['content']) for message in truncated
        )

        if available_length > 100:  # Ensure meaningful truncation
            truncated_content = last_message['content'][:available_length-3] + "..."
            truncated.append({
                "role": last_message['role'],
                "content": truncated_content
            })
            logger.warning(
                f"Truncated last message from {len(last_message['content'])} "
                f"to {len(truncated_content)} characters"
            )
        else:
            logger.warning("Not enough space to include truncated message")

        return truncated

    def get_output(
        self,
        messages: List[Dict[str, str]],
        settings: APISettings,
        response_type: str = 'text',
        save_dir: Optional[Path] = None
    ) -> str:
        """Get output from OpenAI API with retry logic.

        Args:
            messages: List of message dictionaries
            settings: API settings
            response_type: 'text' or 'image'
            save_dir: Directory to save long messages

        Returns:
            Generated text response
        """
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = generate_response(
                    self.client,
                    self.model,
                    messages,
                    settings,
                    response_type
                )

                if (response.choices and
                    response.choices[0].message and
                    hasattr(response.choices[0].message, 'content')):
                    return response.choices[0].message.content
                else:
                    return "Error: Wrong response format."

            except openai.BadRequestError as error:
                error_message = str(error)

                if "string too long" in error_message or \
                   "maximum context length" in error_message:
                    logging.error("Message too long. Attempting to truncate.")
                    self._save_long_message(messages, save_dir)
                    messages = self._truncate_messages(messages)
                    continue
                else:
                    logging.error(
                        f'Attempt {attempt + 1} of {MAX_ATTEMPTS} '
                        f'failed with error: {error}'
                    )

            except (TimeoutError,
                    openai.APIError,
                    openai.APIConnectionError,
                    openai.RateLimitError) as error:
                logging.error(
                    f'Attempt {attempt + 1} of {MAX_ATTEMPTS} '
                    f'failed with error: {error}'
                )

            if attempt < MAX_ATTEMPTS - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                return f"Error: Max attempts reached. Last error: {error}"

        return "Error: All retry attempts failed"


if __name__ == '__main__':
    # Test the API handler
    handler = APIHandler('gpt-5-mini')
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How are you today?"}
    ]
    settings = APISettings(max_completion_tokens=50)
    output_text = handler.get_output(messages=messages, settings=settings)
    print(output_text)
