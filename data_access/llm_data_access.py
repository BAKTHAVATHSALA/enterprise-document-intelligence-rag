"""OpenAI LLM Data Access Layer.

Handles OpenAI API client initialization, chat completion generation using gpt-4o-mini,
temperature control for factual grounding, and mock mode support for isolated testing.
"""

import os
from typing import Optional
from openai import OpenAI
from utils.config import get_config
from utils.logger import logger

FUNC_GENERATE_COMPLETION: str = "generate_llm_completion"

DEFAULT_LLM_MODEL: str = "gpt-4o-mini"
_OPENAI_CLIENT_INSTANCE: Optional[OpenAI] = None
_USE_MOCK_LLM: bool = False


def set_mock_llm_mode(enabled: bool) -> None:
    """Toggle mock LLM completion mode for isolated unit testing.

    @param enabled: True to return mock answer text, False for OpenAI API execution.
    """
    global _USE_MOCK_LLM
    _USE_MOCK_LLM = enabled


def get_openai_client() -> Optional[OpenAI]:
    """Retrieve or initialize singleton OpenAI API client instance.

    @returns: OpenAI client instance or None if API key missing or mock mode active.
    """
    global _OPENAI_CLIENT_INSTANCE
    if _OPENAI_CLIENT_INSTANCE is not None:
        return _OPENAI_CLIENT_INSTANCE

    if _USE_MOCK_LLM:
        return None

    config = get_config()
    api_key: str = config.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning(FUNC_GENERATE_COMPLETION, "OPENAI_API_KEY unavailable. Defaulting to mock LLM mode.")
        return None

    try:
        _OPENAI_CLIENT_INSTANCE = OpenAI(api_key=api_key)
        logger.info(FUNC_GENERATE_COMPLETION, "Successfully initialized OpenAI API client.")
        return _OPENAI_CLIENT_INSTANCE
    except Exception as exc:
        logger.error(FUNC_GENERATE_COMPLETION, "Failed to initialize OpenAI API client", exc=exc)
        return None


def generate_llm_completion(
    system_prompt: str,
    user_prompt: str,
    model_id: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.0,
) -> str:
    """Generate chat completion text from OpenAI API using gpt-4o-mini or mock fallback.

    @param system_prompt: Instructions defining assistant role and grounding rules.
    @param user_prompt: Input payload containing evidence context and question.
    @param model_id: OpenAI model identifier string (default 'gpt-4o-mini').
    @param temperature: Sampling temperature (default 0.0 for deterministic factual answer).
    @returns: Generated response text string.
    """
    client = get_openai_client()

    if client is not None and not _USE_MOCK_LLM:
        try:
            logger.info(FUNC_GENERATE_COMPLETION, f"Invoking OpenAI model '{model_id}' (temp={temperature})...")
            response = client.chat.completions.create(
                model=model_id,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content: str = response.choices[0].message.content or ""
            logger.info(FUNC_GENERATE_COMPLETION, f"Received response from '{model_id}' ({len(content)} chars).")
            return content.strip()
        except Exception as exc:
            logger.error(FUNC_GENERATE_COMPLETION, f"Error calling OpenAI API model '{model_id}'", exc=exc)
            # Fall through to mock response on API failure

    # Mock response for testing or offline environment
    logger.info(FUNC_GENERATE_COMPLETION, "Returning mock LLM completion response.")
    return "Based on the provided context, the requested policy details are specified in the ingested documentation."
