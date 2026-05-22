from unittest.mock import patch

from src.data.binance_connector import BinanceConnector, CANDLES, TICKER, TRADES
from src.data.config import DataConfig


class _FakeFeedHandler:
    def __init__(self):
        self.feeds = []

    def add_feed(self, feed):
        self.feeds.append(feed)


class _FakeBinance:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_create_feedhandler_registers_single_binance_feed():
    handler = _FakeFeedHandler()
    config = DataConfig(
        symbols=["BTC-USDT", "ETH-USDT"],
        channels=["TRADES", "CANDLES", "TICKER"],
        candle_interval="5m",
    )
    connector = BinanceConnector(config=config)

    with patch("src.data.binance_connector.FeedHandler", return_value=handler), patch(
        "src.data.binance_connector.Binance",
        side_effect=lambda **kwargs: _FakeBinance(**kwargs),
    ), patch(
        "src.data.binance_connector.TradeCallback",
        side_effect=lambda callback: ("trade", callback),
    ), patch(
        "src.data.binance_connector.CandleCallback",
        side_effect=lambda callback, candle_interval: (
            "candle",
            callback,
            candle_interval,
        ),
    ), patch(
        "src.data.binance_connector.TickerCallback",
        side_effect=lambda callback: ("ticker", callback),
    ):
        created = connector._create_feedhandler()

    assert created is handler
    assert connector._feedhandler is handler
    assert len(handler.feeds) == 1

    feed = handler.feeds[0]
    assert feed.kwargs["symbols"] == ["BTC-USDT", "ETH-USDT"]
    assert feed.kwargs["channels"] == [TRADES, CANDLES, TICKER]
    assert feed.kwargs["callbacks"][TRADES][0] == "trade"
    assert feed.kwargs["callbacks"][CANDLES] == (
        "candle",
        connector._candle_callback,
        "5m",
    )
    assert feed.kwargs["callbacks"][TICKER] == ("ticker", connector._ticker_callback)