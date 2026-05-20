"""
LLM client wrapper using LiteLLM for multi-provider routing with fallback.

Features:
- Multi-provider routing via LiteLLM (OpenAI-compatible interface)
- Automatic fallback to secondary model on failure
- Structured JSON output support
- Retry with exponential backoff for rate limits
- Token usage and latency logging
"""

from __future__ import annotations

import json
import time
from typing import Any, cast

from loguru import logger

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]


class LLMClientError(Exception):
    """Raised when LLM calls fail after all retries and fallbacks."""


class LLMClient:
    """LiteLLM-based LLM client with fallback, retry, and structured output."""

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        temperature: float = 0.7,
        api_keys: dict[str, str] | None = None,
        fallback_model: str | None = None,
        max_retries: int = 3,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            model: Primary LiteLLM model identifier (e.g. "anthropic/claude-sonnet-4").
            temperature: Sampling temperature (0.0-1.5).
            api_keys: Optional dict mapping provider names to API keys.
                      Keys are set as environment variables at call time.
            fallback_model: Secondary model to try if primary fails.
            max_retries: Number of retry attempts per model.
            max_tokens: Maximum tokens in the response.
            timeout: Timeout in seconds per LLM call.
        """
        if litellm is None:
            raise ImportError(
                "litellm is required. Install with: pip install litellm"
            )

        self.model = model
        self.temperature = temperature
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._api_keys = api_keys or {}

        # Set API keys from env if provided
        for provider, key in self._api_keys.items():
            litellm.api_key = key  # Default key
            logger.debug(f"Set API key for provider: {provider}")

        # Cumulative stats
        self.total_tokens_prompt = 0
        self.total_tokens_completion = 0
        self.total_calls = 0
        self.total_latency = 0.0

    def call(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Call the LLM with messages and optional structured output format.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_format: Optional dict specifying JSON schema for structured output.
            temperature: Override temperature for this call.

        Returns:
            The LLM's response as a string (JSON if response_format provided).

        Raises:
            LLMClientError: If all models and retries are exhausted.
        """
        temp = temperature if temperature is not None else self.temperature
        models_to_try = [self.model]
        if self.fallback_model:
            models_to_try.append(self.fallback_model)

        last_error: Exception | None = None

        for model_id in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    start = time.monotonic()
                    kwargs: dict[str, Any] = {
                        "model": model_id,
                        "messages": messages,
                        "temperature": temp,
                        "max_tokens": self.max_tokens,
                        "timeout": self.timeout,
                    }

                    # Structured output via response_format
                    if response_format is not None:
                        kwargs["response_format"] = {"type": "json_object"}

                    response = litellm.completion(**kwargs)
                    latency = time.monotonic() - start

                    # Extract content
                    content = response.choices[0].message.content
                    if content is None:
                        content = ""

                    # Track usage
                    usage = response.get("usage", {})
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

                    self.total_tokens_prompt += prompt_tokens
                    self.total_tokens_completion += completion_tokens
                    self.total_calls += 1
                    self.total_latency += latency

                    logger.info(
                        f"LLM call ({model_id}): "
                        f"prompt_tokens={prompt_tokens}, "
                        f"completion_tokens={completion_tokens}, "
                        f"latency={latency:.2f}s, "
                        f"attempt={attempt + 1}"
                    )

                    return str(content)

                except Exception as exc:
                    last_error = exc
                    wait = min(2**attempt * 1.0, 30.0)
                    logger.warning(
                        f"LLM call failed ({model_id}, attempt {attempt + 1}): "
                        f"{exc}. Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)

        raise LLMClientError(
            f"All models and retries exhausted. Last error: {last_error}"
        )

    def call_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Call the LLM and parse the response as JSON.

        Args:
            messages: List of message dicts.
            temperature: Override temperature.

        Returns:
            Parsed JSON dict from the LLM response.

        Raises:
            LLMClientError: If the response cannot be parsed as JSON.
        """
        raw = self.call(
            messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )

        # Strip markdown code blocks if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening ```json or ```
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline:]
            # Remove closing ```
            cleaned = cleaned.rstrip().removesuffix("```").strip()

        try:
            return cast(dict[str, Any], json.loads(cleaned))
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse LLM response as JSON: {cleaned[:500]}")
            raise LLMClientError(f"JSON parse error: {exc}") from exc

    def get_stats(self) -> dict[str, Any]:
        """Return cumulative usage statistics."""
        avg_latency = (
            self.total_latency / self.total_calls if self.total_calls > 0 else 0.0
        )
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_tokens_prompt,
            "total_completion_tokens": self.total_tokens_completion,
            "total_tokens": self.total_tokens_prompt + self.total_tokens_completion,
            "total_latency_seconds": round(self.total_latency, 2),
            "avg_latency_seconds": round(avg_latency, 3),
        }

    def reset_stats(self) -> None:
        """Reset cumulative usage statistics."""
        self.total_tokens_prompt = 0
        self.total_tokens_completion = 0
        self.total_calls = 0
        self.total_latency = 0.0
