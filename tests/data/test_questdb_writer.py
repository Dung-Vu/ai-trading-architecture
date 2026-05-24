from datetime import datetime, timezone
from unittest.mock import Mock

from src.data.questdb_writer import QuestDBWriter


def test_write_trade_uses_sender_row_api():
    writer = QuestDBWriter(addr="localhost:9000")
    writer._sender = Mock()

    result = writer.write_trade(
        symbol="BTC-USDT",
        side="buy",
        price=100000.0,
        amount=0.25,
        trade_id="trade-1",
        exchange="binance",
        ts_ns=123456789,
    )

    assert result is True
    writer._sender.row.assert_called_once_with(
        "trades",
        symbols={
            "symbol": "BTC-USDT",
            "side": "buy",
            "trade_id": "trade-1",
            "exchange": "binance",
        },
        columns={
            "price": 100000.0,
            "amount": 0.25,
            "quantity": 0.25,
            "pnl": 0.0,
        },
        at=123456789,
    )


def test_write_news_saves_expected_fields():
    writer = QuestDBWriter(addr="localhost:9000")
    writer._sender = Mock()

    result = writer.write_news(
        {
            "title": "BTC ETF inflows rise",
            "url": "https://news.example/etf",
            "source": "CoinDesk",
            "symbol": "BTC",
            "content_snippet": "Institutional adoption keeps climbing.",
            "sentiment_score": 0.75,
            "timestamp": "2026-05-20T00:00:00Z",
        }
    )

    assert result is True
    writer._sender.row.assert_called_once_with(
        "news",
        symbols={
            "symbol": "BTC",
            "source": "CoinDesk",
        },
        columns={
            "title": "BTC ETF inflows rise",
            "url": "https://news.example/etf",
            "content_snippet": "Institutional adoption keeps climbing.",
            "sentiment_score": 0.75,
        },
        at=datetime(2026, 5, 20, 0, 0, tzinfo=timezone.utc),
    )


def test_write_trade_returns_false_on_sender_error():
    writer = QuestDBWriter(addr="localhost:9000")
    writer._sender = Mock()
    writer._sender.row.side_effect = RuntimeError("sender failed")

    result = writer.write_trade(
        symbol="BTC-USDT",
        side="buy",
        price=100000.0,
        amount=0.25,
        trade_id="trade-1",
        exchange="binance",
        ts_ns=123456789,
    )

    assert result is False


def test_save_news_to_db_uses_writer_method():
    from src.data.news_pipeline import NewsPipeline

    pipeline = NewsPipeline(["BTC"])
    writer = Mock()
    writer.write_news.return_value = True

    saved = pipeline.save_news_to_db(
        [
            {
                "title": "BTC breakout",
                "url": "https://news.example/btc-breakout",
                "source": "CoinDesk",
                "symbol": "BTC",
                "content_snippet": "Bullish rally after ETF approval.",
                "timestamp": "2026-05-20T00:00:00Z",
            }
        ],
        writer,
    )

    assert saved == 1
    writer.write_news.assert_called_once()
    record = writer.write_news.call_args.args[0]
    assert record["symbol"] == "BTC"
    assert record["source"] == "CoinDesk"
    assert record["timestamp"] == "2026-05-20T00:00:00Z"
    assert isinstance(record["sentiment_score"], float)


def test_save_news_to_db_skips_failed_writes():
    from src.data.news_pipeline import NewsPipeline

    pipeline = NewsPipeline(["BTC"])
    writer = Mock()
    writer.write_news.return_value = False

    saved = pipeline.save_news_to_db(
        [
            {
                "title": "BTC breakout",
                "url": "https://news.example/btc-breakout",
                "source": "CoinDesk",
                "symbol": "BTC",
                "content_snippet": "Bullish rally after ETF approval.",
                "timestamp": "2026-05-20T00:00:00Z",
            }
        ],
        writer,
    )

    assert saved == 0
    writer.write_news.assert_called_once()
