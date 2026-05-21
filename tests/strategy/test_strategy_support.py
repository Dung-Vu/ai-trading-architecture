import pandas as pd
import pytest

from src.strategy.indicator_utils import calculate_indicator
from src.strategy.mock_runtime import MockBroker, ensure_strategy_runtime_kwargs


def test_ensure_strategy_runtime_kwargs_injects_mock_runtime_objects():
    runtime_kwargs = ensure_strategy_runtime_kwargs({})

    assert isinstance(runtime_kwargs["broker"], MockBroker)
    assert runtime_kwargs["data_source"] is runtime_kwargs["broker"].data_source


def test_ensure_strategy_runtime_kwargs_preserves_explicit_runtime_objects():
    broker = MockBroker()
    data_source = object()

    runtime_kwargs = ensure_strategy_runtime_kwargs(
        {"broker": broker, "data_source": data_source}
    )

    assert runtime_kwargs["broker"] is broker
    assert runtime_kwargs["data_source"] is data_source


def test_calculate_indicator_supports_indicator_aliases():
    pytest.importorskip("ta")
    bars = pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [101, 102, 103, 104, 105, 106, 107, 108],
            "low": [99, 100, 101, 102, 103, 104, 105, 106],
            "volume": [10, 11, 12, 13, 14, 15, 16, 17],
        }
    )

    series = calculate_indicator(bars, "stochastic", period=3, smooth_window=2)

    assert len(series) == len(bars)


def test_calculate_indicator_returns_bollinger_tuple():
    pytest.importorskip("ta")
    bars = pd.DataFrame(
        {
            "close": [100 + index for index in range(30)],
            "high": [101 + index for index in range(30)],
            "low": [99 + index for index in range(30)],
            "volume": [10 + index for index in range(30)],
        }
    )

    upper, middle, lower = calculate_indicator(bars, "bb", window=20)

    assert len(upper) == len(bars)
    assert len(middle) == len(bars)
    assert len(lower) == len(bars)
    assert upper.iloc[-1] >= middle.iloc[-1] >= lower.iloc[-1]