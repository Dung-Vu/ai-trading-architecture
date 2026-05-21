from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.debate.llm_client import LLMClient, LLMClientError


@patch("src.debate.llm_client.time.sleep", return_value=None)
@patch("src.debate.llm_client.litellm.completion")
def test_llm_client_uses_provider_specific_api_keys_for_fallbacks(
    mock_completion,
    _mock_sleep,
):
    response = MagicMock()
    response.choices = [SimpleNamespace(message=SimpleNamespace(content="ok"))]
    response.get.return_value = {}

    mock_completion.side_effect = [RuntimeError("primary failed"), response]

    client = LLMClient(
        model="openai/gpt-4o-mini",
        fallback_model="anthropic/claude-sonnet-4",
        api_keys={
            "openai": "openai-key",
            "anthropic": "anthropic-key",
        },
        max_retries=1,
    )

    result = client.call([
        {"role": "user", "content": "hello"},
    ])

    assert result == "ok"
    assert mock_completion.call_args_list[0].kwargs["api_key"] == "openai-key"
    assert mock_completion.call_args_list[1].kwargs["api_key"] == "anthropic-key"


@patch("src.debate.llm_client.time.sleep", return_value=None)
@patch("src.debate.llm_client.litellm.completion")
def test_llm_client_caches_identical_requests(
    mock_completion,
    _mock_sleep,
):
    response = MagicMock()
    response.choices = [SimpleNamespace(message=SimpleNamespace(content="cached"))]
    response.get.return_value = {}
    mock_completion.return_value = response

    client = LLMClient(
        model="openai/gpt-4o-mini",
        api_keys={"openai": "openai-key"},
        max_retries=1,
        cache_ttl_seconds=60.0,
        cache_max_entries=8,
    )

    messages = [{"role": "user", "content": "hello"}]

    assert client.call(messages) == "cached"
    assert client.call(messages) == "cached"
    assert mock_completion.call_count == 1
    assert client.get_stats()["cache_hits"] == 1


@patch("src.debate.llm_client.time.sleep", return_value=None)
@patch("src.debate.llm_client.litellm.completion", side_effect=RuntimeError("boom"))
def test_llm_client_opens_circuit_breaker_after_repeated_failures(
    mock_completion,
    _mock_sleep,
):
    client = LLMClient(
        model="openai/gpt-4o-mini",
        api_keys={"openai": "openai-key"},
        max_retries=1,
        circuit_breaker_threshold=2,
        circuit_breaker_reset_seconds=30.0,
    )

    messages = [{"role": "user", "content": "hello"}]

    with pytest.raises(LLMClientError, match="All models and retries exhausted"):
        client.call(messages)

    with pytest.raises(LLMClientError, match="All models and retries exhausted"):
        client.call(messages)

    with pytest.raises(LLMClientError, match="circuit breaker is open"):
        client.call(messages)

    assert mock_completion.call_count == 2