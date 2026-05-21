from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.execution.exchange_client import ExchangeClient


def test_exchange_property_requires_connection():
    client = ExchangeClient()

    with pytest.raises(RuntimeError, match="Call connect\(\) first"):
        _ = client.exchange


def test_connect_builds_exchange_and_enables_sandbox_in_testnet_mode():
    exchange = MagicMock()
    exchange_factory = MagicMock(return_value=exchange)

    with patch(
        "src.execution.exchange_client.ccxt",
        SimpleNamespace(binance=exchange_factory),
    ):
        client = ExchangeClient(
            api_key="key",
            api_secret="secret",
            testnet=True,
            rate_limit_ms=123,
            exchange_name="binance",
        )
        client.connect()

    assert client.is_connected is True
    assert client.exchange is exchange
    exchange_factory.assert_called_once_with({
        "apiKey": "key",
        "secret": "secret",
        "enableRateLimit": True,
        "rateLimit": 123,
        "options": {"defaultType": "spot"},
    })
    exchange.set_sandbox_mode.assert_called_once_with(True)


def test_connect_rejects_unknown_exchange_name():
    with patch("src.execution.exchange_client.ccxt", SimpleNamespace()):
        client = ExchangeClient(exchange_name="unknown")

        with pytest.raises(ValueError, match="Unsupported CCXT exchange"):
            client.connect()


@pytest.mark.parametrize(
    ("method_name", "args", "expected", "expected_kwargs"),
    [
        ("fetch_ticker", ("BTC/USDT",), {"last": 100.0}, {}),
        ("fetch_balance", tuple(), {"free": {"USDT": 1000.0}}, {}),
        (
            "fetch_open_orders",
            ("BTC/USDT",),
            [{"id": "open-1", "symbol": "BTC/USDT"}],
            {},
        ),
        ("fetch_order", ("order-1", "BTC/USDT"), {"id": "order-1"}, {}),
    ],
)
def test_exchange_client_proxies_exchange_calls(
    method_name,
    args,
    expected,
    expected_kwargs,
):
    client = ExchangeClient()
    exchange = MagicMock()
    getattr(exchange, method_name).return_value = expected
    client._exchange = exchange

    result = getattr(client, method_name)(*args)

    assert result == expected
    getattr(exchange, method_name).assert_called_once_with(*args, **expected_kwargs)


def test_fetch_ohlcv_passes_limit_as_keyword_argument():
    client = ExchangeClient()
    exchange = MagicMock()
    exchange.fetch_ohlcv.return_value = [[1, 2, 3, 4, 5, 6]]
    client._exchange = exchange

    result = client.fetch_ohlcv("BTC/USDT", "5m", 50)

    assert result == [[1, 2, 3, 4, 5, 6]]
    exchange.fetch_ohlcv.assert_called_once_with("BTC/USDT", "5m", limit=50)


def test_close_clears_connected_exchange():
    client = ExchangeClient()
    client._exchange = MagicMock()

    client.close()

    assert client.is_connected is False