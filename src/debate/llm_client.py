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

import hashlib
import json
import time
from typing import Any

from loguru import logger

from src.config import (
    get_default_debate_max_tokens,
    get_default_debate_temperature,
    get_default_debate_timeout_seconds,
    get_default_litellm_model,
    get_default_llm_cache_max_entries,
    get_default_llm_cache_ttl_seconds,
    get_default_llm_circuit_breaker_reset_seconds,
    get_default_llm_circuit_breaker_threshold,
    get_default_llm_max_retries,
)
from src.shared_utils import trim_mapping_size

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
        model: str | None = None,
        temperature: float | None = None,
        api_keys: dict[str, str] | None = None,
        fallback_model: str | None = None,
        max_retries: int | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        cache_ttl_seconds: float | None = None,
        cache_max_entries: int | None = None,
        circuit_breaker_threshold: int | None = None,
        circuit_breaker_reset_seconds: float | None = None,
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
            cache_ttl_seconds: TTL for exact-request response caching.
            cache_max_entries: Maximum number of cached exact requests.
            circuit_breaker_threshold: Consecutive failed requests before opening the breaker.
            circuit_breaker_reset_seconds: How long to keep the circuit open.
        """
        if litellm is None:
            raise ImportError(
                "litellm is required. Install with: pip install litellm"
            )

        self.model = model if model is not None else get_default_litellm_model()
        self.temperature = (
            temperature if temperature is not None else get_default_debate_temperature()
        )
        self.fallback_model = fallback_model
        self.max_retries = (
            max_retries if max_retries is not None else get_default_llm_max_retries()
        )
        self.max_tokens = (
            max_tokens if max_tokens is not None else get_default_debate_max_tokens()
        )
        self.timeout = (
            timeout if timeout is not None else get_default_debate_timeout_seconds()
        )
        self.cache_ttl_seconds = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else get_default_llm_cache_ttl_seconds()
        )
        self.cache_max_entries = (
            cache_max_entries
            if cache_max_entries is not None
            else get_default_llm_cache_max_entries()
        )
        self.circuit_breaker_threshold = (
            circuit_breaker_threshold
            if circuit_breaker_threshold is not None
            else get_default_llm_circuit_breaker_threshold()
        )
        self.circuit_breaker_reset_seconds = (
            circuit_breaker_reset_seconds
            if circuit_breaker_reset_seconds is not None
            else get_default_llm_circuit_breaker_reset_seconds()
        )
        self._api_keys = api_keys or {}
        self._response_cache: dict[str, tuple[float, str]] = {}
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

        for provider in self._api_keys:
            logger.debug(f"Configured API key for provider: {provider}")

        # Cumulative stats
        self.total_tokens_prompt = 0
        self.total_tokens_completion = 0
        self.total_calls = 0
        self.total_latency = 0.0
        self.cache_hits = 0
        self.circuit_open_rejections = 0

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

        now = time.monotonic()
        if self._circuit_open_until > now:
            self.circuit_open_rejections += 1
            raise LLMClientError(
                "LLM circuit breaker is open due to repeated upstream failures"
            )

        cache_key = self._build_cache_key(
            models_to_try=models_to_try,
            messages=messages,
            response_format=response_format,
            temperature=temp,
        )
        cached = self._get_cached_response(cache_key, now)
        if cached is not None:
            self.cache_hits += 1
            return cached

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

                    api_key = self._resolve_api_key(model_id)
                    if api_key:
                        kwargs["api_key"] = api_key

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

                    self._record_success()
                    self._cache_response(cache_key, content)

                    return content

                except Exception as exc:
                    last_error = exc
                    wait = min(2**attempt * 1.0, 30.0)
                    logger.warning(
                        f"LLM call failed ({model_id}, attempt {attempt + 1}): "
                        f"{exc}. Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)

        self._record_failure(last_error)

        raise LLMClientError(
            f"All models and retries exhausted. Last error: {last_error}"
        )

    def _build_cache_key(
        self,
        *,
        models_to_try: list[str],
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
        temperature: float,
    ) -> str:
        """Build a deterministic cache key for an exact LLM request."""
        payload = json.dumps(
            {
                "models_to_try": models_to_try,
                "messages": messages,
                "response_format": response_format,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_cached_response(self, cache_key: str, now: float) -> str | None:
        """Return a cached response when the exact-request TTL is still valid."""
        cached = self._response_cache.get(cache_key)
        if cached is None:
            return None

        expires_at, content = cached
        if expires_at <= now:
            self._response_cache.pop(cache_key, None)
            return None

        return content

    def _cache_response(self, cache_key: str, content: str) -> None:
        """Store a successful exact-request response in the short-lived cache."""
        if self.cache_ttl_seconds <= 0:
            return

        expires_at = time.monotonic() + self.cache_ttl_seconds
        self._response_cache[cache_key] = (expires_at, content)

        trim_mapping_size(self._response_cache, self.cache_max_entries)

    def _record_success(self) -> None:
        """Reset circuit-breaker state after a successful upstream response."""
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_failure(self, exc: Exception | None) -> None:
        """Advance circuit-breaker state after a full request failure."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self.circuit_breaker_threshold:
            return

        self._circuit_open_until = time.monotonic() + self.circuit_breaker_reset_seconds
        logger.error(
            "LLM circuit breaker opened for "
            f"{self.circuit_breaker_reset_seconds:.1f}s after "
            f"{self._consecutive_failures} consecutive failures: {exc}"
        )

    def _resolve_api_key(self, model_id: str) -> str | None:
        """Return the API key configured for the model's provider."""
        provider = model_id.split("/", 1)[0].lower()
        return self._api_keys.get(provider)

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
            return json.loads(cleaned)
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
            "cache_hits": self.cache_hits,
            "cached_requests": len(self._response_cache),
            "circuit_open_until": round(self._circuit_open_until, 3),
            "circuit_open_rejections": self.circuit_open_rejections,
        }

    def reset_stats(self) -> None:
        """Reset cumulative usage statistics."""
        self.total_tokens_prompt = 0
        self.total_tokens_completion = 0
        self.total_calls = 0
        self.total_latency = 0.0
        self.cache_hits = 0
        self.circuit_open_rejections = 0
        self._response_cache.clear()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
