from types import SimpleNamespace
from unittest.mock import MagicMock, call

from src.execution.order_manager import OrderManager


def test_create_market_order_uses_exchange_precision_in_live_mode():
    exchange = MagicMock()
    exchange.amount_to_precision.return_value = "0.1234"
    exchange.create_market_order.return_value = {"id": "mkt-1", "status": "open"}

    manager = OrderManager(SimpleNamespace(exchange=exchange), dry_run=False)

    result = manager.create_market_order("BTC/USDT", "buy", 0.123456)

    assert result == {"id": "mkt-1", "status": "open"}
    exchange.amount_to_precision.assert_called_once_with("BTC/USDT", 0.123456)
    exchange.create_market_order.assert_called_once_with("BTC/USDT", "buy", "0.1234")


def test_create_bracket_order_formats_prices_and_places_live_orders():
    exchange = MagicMock()
    exchange.amount_to_precision.return_value = "0.2500"
    exchange.price_to_precision.side_effect = lambda symbol, price: f"{symbol}:{float(price):.2f}"
    exchange.create_order.side_effect = [
        {"id": "entry-1"},
        {"id": "sl-1"},
        {"id": "tp-1"},
    ]

    manager = OrderManager(SimpleNamespace(exchange=exchange), dry_run=False)

    result = manager.create_bracket_order(
        symbol="ETH/USDT",
        side="buy",
        amount=0.25,
        entry_price=3000.0,
        stop_loss=2900.0,
        take_profit=3300.0,
    )

    assert result == {
        "entry_order_id": "entry-1",
        "sl_order_id": "sl-1",
        "tp_order_id": "tp-1",
        "symbol": "ETH/USDT",
        "side": "buy",
        "amount": 0.25,
        "entry_price": 3000.0,
        "stop_loss": 2900.0,
        "take_profit": 3300.0,
    }
    assert exchange.amount_to_precision.call_count == 3
    assert exchange.price_to_precision.call_args_list == [
        call("ETH/USDT", 3000.0),
        call("ETH/USDT", 2900.0),
        call("ETH/USDT", 2900.0),
        call("ETH/USDT", 3300.0),
        call("ETH/USDT", 3300.0),
    ]
    assert exchange.create_order.call_args_list == [
        call("ETH/USDT", "limit", "buy", "0.2500", "ETH/USDT:3000.00"),
        call(
            "ETH/USDT",
            "stop_loss_limit",
            "sell",
            "0.2500",
            "ETH/USDT:2900.00",
            {"stopPrice": "ETH/USDT:2900.00"},
        ),
        call(
            "ETH/USDT",
            "take_profit_limit",
            "sell",
            "0.2500",
            "ETH/USDT:3300.00",
            {"stopPrice": "ETH/USDT:3300.00"},
        ),
    ]