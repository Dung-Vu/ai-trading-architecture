import time
from types import SimpleNamespace
from unittest.mock import patch

from src.data.news_pipeline import NewsPipeline


class _FakeEntry(dict):
    pass


def test_fetch_rss_filters_symbol_mentions_and_parses_entries():
    matching = _FakeEntry(
        title="Bitcoin rally continues",
        summary="BTC adoption drives another breakout.",
        link="https://news.example/btc-rally",
        published="2026-05-20T00:00:00Z",
    )
    matching.published_parsed = time.gmtime(1716163200)

    ignored = _FakeEntry(
        title="Gold market update",
        summary="Macro commentary only.",
        link="https://news.example/gold",
    )
    ignored.published_parsed = time.gmtime(1716163200)

    pipeline = NewsPipeline(["BTC"])

    fake_feedparser = SimpleNamespace(
        parse=lambda _url: type("Feed", (), {"entries": [matching, ignored]})()
    )

    with patch("src.data.news_pipeline.FEEDPARSER_AVAILABLE", True), patch(
        "src.data.news_pipeline.feedparser",
        fake_feedparser,
    ):
        items = pipeline._fetch_rss("https://rss.example/feed", "BTC", source="Feed")

    assert items == [{
        "title": "Bitcoin rally continues",
        "url": "https://news.example/btc-rally",
        "source": "Feed",
        "timestamp": "2026-05-20T00:00:00Z",
        "symbol": "BTC",
        "content_snippet": "BTC adoption drives another breakout.",
        "_ts": 1716163200,
        "_raw_summary": "BTC adoption drives another breakout.",
    }]


def test_fetch_crypto_news_filters_old_items_and_deduplicates_urls():
    pipeline = NewsPipeline(["BTC"], cryptopanic_api_key="key")
    now_ts = time.time()

    with patch.object(
        pipeline,
        "_fetch_rss",
        side_effect=[
            [
                {
                    "title": "Recent BTC rally",
                    "url": "https://news.example/shared",
                    "source": "CoinDesk",
                    "timestamp": "recent",
                    "symbol": "BTC",
                    "content_snippet": "bullish breakout",
                    "_ts": now_ts,
                },
                {
                    "title": "Old BTC story",
                    "url": "https://news.example/old",
                    "source": "CoinDesk",
                    "timestamp": "old",
                    "symbol": "BTC",
                    "content_snippet": "old news",
                    "_ts": now_ts - 60 * 60 * 30,
                },
            ],
            [
                {
                    "title": "Duplicate URL from second feed",
                    "url": "https://news.example/shared",
                    "source": "CoinTelegraph",
                    "timestamp": "recent",
                    "symbol": "BTC",
                    "content_snippet": "same url",
                    "_ts": now_ts,
                }
            ],
            [],
        ],
    ), patch.object(
        pipeline,
        "_fetch_cryptopanic",
        return_value=[
            {
                "title": "Fresh API item",
                "url": "https://news.example/api",
                "source": "CryptoPanic",
                "timestamp": "recent",
                "symbol": "BTC",
                "content_snippet": "institutional adoption",
                "_ts": now_ts,
            }
        ],
    ):
        items = pipeline.fetch_crypto_news("BTC", hours=24)

    assert items == [
        {
            "title": "Recent BTC rally",
            "url": "https://news.example/shared",
            "source": "CoinDesk",
            "timestamp": "recent",
            "symbol": "BTC",
            "content_snippet": "bullish breakout",
        },
        {
            "title": "Fresh API item",
            "url": "https://news.example/api",
            "source": "CryptoPanic",
            "timestamp": "recent",
            "symbol": "BTC",
            "content_snippet": "institutional adoption",
        },
    ]


def test_fetch_cryptopanic_parses_created_at_without_name_error():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "title": "BTC ETF inflows rise",
                        "created_at": "2026-05-20T00:00:00Z",
                        "url": "https://news.example/api",
                    }
                ]
            }

    class Client:
        def get(self, url, params):
            assert url == "https://panic.example/api"
            assert params["currencies"] == "BTC"
            return Response()

    pipeline = NewsPipeline(
        ["BTC"],
        cryptopanic_api_key="key",
        cryptopanic_api_url="https://panic.example/api",
    )

    with patch.object(pipeline, "_get_http_client", return_value=Client()):
        items = pipeline._fetch_cryptopanic("BTC")

    assert items == [
        {
            "title": "BTC ETF inflows rise",
            "url": "https://news.example/api",
            "source": "CryptoPanic",
            "timestamp": "2026-05-20T00:00:00Z",
            "symbol": "BTC",
            "content_snippet": "BTC ETF inflows rise",
            "_ts": 1779235200.0,
        }
    ]


def test_get_market_sentiment_aggregates_symbol_scores():
    pipeline = NewsPipeline(["BTC", "ETH"])

    with patch.object(
        pipeline,
        "fetch_crypto_news",
        side_effect=[
            [
                {
                    "title": "BTC bullish breakout",
                    "content_snippet": "adoption rally and breakout",
                    "source": "CoinDesk",
                    "symbol": "BTC",
                }
            ],
            [
                {
                    "title": "ETH hack investigation",
                    "content_snippet": "bearish crash after exploit",
                    "source": "CoinTelegraph",
                    "symbol": "ETH",
                }
            ],
        ],
    ):
        sentiment = pipeline.get_market_sentiment(hours=12)

    assert sentiment["BTC"]["positive"] == 1
    assert sentiment["BTC"]["negative"] == 0
    assert sentiment["ETH"]["positive"] == 0
    assert sentiment["ETH"]["negative"] == 1
    assert sentiment["aggregate"] == 0.0
    assert sentiment["total_news"] == 2


def test_close_releases_http_client_once():
    client = SimpleNamespace(close=patch)
    pipeline = NewsPipeline(["BTC"])
    close_mock = patch("src.data.news_pipeline.logger.debug")

    from unittest.mock import MagicMock

    managed_client = MagicMock()
    pipeline._http_client = managed_client

    pipeline.close()
    pipeline.close()

    managed_client.close.assert_called_once_with()
    assert pipeline._http_client is None
