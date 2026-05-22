"""
News & Sentiment Pipeline — Fetches crypto news from multiple sources,
analyzes sentiment, and integrates with the trading decision pipeline.

Sources:
    - CoinDesk RSS feed
    - CoinTelegraph RSS feed
    - CryptoPanic API (if API key provided)
    - Twitter/X sentiment via keyword analysis

Sentiment Analysis:
    - Keyword-based scoring with positive/negative word lists
    - Per-symbol and aggregate sentiment scores
    - Historical persistence via QuestDB

Usage:
    >>> pipeline = NewsPipeline(["BTC", "ETH"])
    >>> news = pipeline.fetch_crypto_news("BTC", hours=24)
    >>> sentiment = pipeline.analyze_sentiment(news)
    >>> market = pipeline.get_market_sentiment(["BTC", "ETH"])
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from src.config import (
    get_default_cryptopanic_api_key,
    get_default_cryptopanic_api_url,
    get_default_news_rss_feeds,
)
from src.shared_utils import parse_iso_timestamp

# ─── Optional Dependencies ─────────────────────────────────────────────

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    feedparser = None  # type: ignore

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore


# ─── Sentiment Keyword Lists ───────────────────────────────────────────

POSITIVE_KEYWORDS = {
    # Price action
    "bull", "bullish", "rally", "surge", "surging", "breakthrough",
    "breakout", "moon", "pump", "ATH", "all-time high", "record high",
    "uptrend", "recovery", "rebound", "recovered",
    # Adoption
    "adoption", "adopted", "partnership", "partner", "integrate",
    "integration", "launch", "launched", "mainnet", "upgrade",
    "halving", "institutional", "etf", "approved", "approval",
    # General positive
    "growth", "profit", "profitable", "gains", "milestone",
    "innovative", "revolutionary", "bullrun", "accumulation",
    "whale buying", "whale accumulation",
}

NEGATIVE_KEYWORDS = {
    # Price action
    "crash", "dump", "dumping", "plunge", "plummet", "collapse",
    "bearish", "bear", "downtrend", "recession", "correction",
    "sell-off", "selloff", "liquidation", "liquidated",
    # Security
    "hack", "hacked", "exploit", "exploited", "breach", "stolen",
    "scam", "fraud", "rug pull", "rugpull", "ponzi",
    # Regulation
    "regulation", "regulated", "ban", "banned", "SEC", "lawsuit",
    "sued", "investigation", "investigating", "enforcement",
    "sanction", "sanctions", "illegal", "prohibited",
    # General negative
    "loss", "losses", "bankrupt", "bankruptcy", "shutdown",
    "delisted", "delisting", "vulnerability", "critical bug",
    "contagion", "crisis", "panic", "fear", "FTX", "Celsius",
}

# Neutral but noteworthy keywords
NOTABLE_KEYWORDS = {
    "announcement", "update", "release", "fork", "airdrop",
    "whale", "transfer", "migration", "v2", "v3", "roadmap",
    "governance", "vote", "proposal", "tokenomics",
}


# ─── NewsPipeline ──────────────────────────────────────────────────────

class NewsPipeline:
    """
    Multi-source crypto news fetcher and sentiment analyzer.

    Fetches news from RSS feeds and APIs, analyzes sentiment using
    keyword scoring, and provides per-symbol and aggregate sentiment.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        languages: list[str] | None = None,
        cryptopanic_api_key: str | None = None,
        twitter_api_key: str | None = None,
        rss_feeds: dict[str, str] | None = None,
        cryptopanic_api_url: str | None = None,
    ) -> None:
        """
        Initialize the news pipeline.

        Args:
            symbols: Crypto symbols to track (e.g. ["BTC", "ETH", "SOL"]).
            languages: Languages to filter news (default: ["en"]).
            cryptopanic_api_key: Optional CryptoPanic API key.
            twitter_api_key: Optional Twitter/X API key for sentiment.
        """
        self.symbols = symbols or ["BTC", "ETH", "SOL", "XRP", "ADA"]
        self.languages = languages or ["en"]
        self.cryptopanic_api_key = (
            cryptopanic_api_key
            if cryptopanic_api_key is not None
            else get_default_cryptopanic_api_key()
        )
        self.twitter_api_key = twitter_api_key
        self.rss_feeds = (
            dict(rss_feeds) if rss_feeds else get_default_news_rss_feeds()
        )
        self.cryptopanic_api_url = (
            cryptopanic_api_url or get_default_cryptopanic_api_url()
        )

        # Internal HTTP client
        self._http_client: Any = None

        logger.info(
            f"✅ NewsPipeline initialized for symbols: {self.symbols}"
        )

    def close(self) -> None:
        """Release any owned HTTP client resources."""
        client = self._http_client
        self._http_client = None
        if client is None:
            return

        try:
            client.close()
        except Exception as exc:
            logger.debug(f"[NewsPipeline] HTTP client close failed: {exc}")

    def __enter__(self) -> "NewsPipeline":
        """Support context-manager usage for explicit client cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close owned resources when leaving a context-manager scope."""
        del exc_type, exc, tb
        self.close()

    def _get_http_client(self) -> Any:
        """Get or create an HTTP client."""
        if HTTPX_AVAILABLE and httpx:
            if self._http_client is None:
                self._http_client = httpx.Client(timeout=15.0, follow_redirects=True)
            return self._http_client
        return None

    # ─── News Fetching ─────────────────────────────────────────────────

    def fetch_crypto_news(
        self, symbol: str, hours: int = 24
    ) -> list[dict[str, Any]]:
        """
        Fetch news for a specific symbol from multiple sources.

        Args:
            symbol: Crypto symbol (e.g. "BTC", "ETH").
            hours: How many hours back to fetch (default: 24).

        Returns:
            List of news item dicts:
                {title, url, source, timestamp, symbol, content_snippet}
        """
        all_news: list[dict[str, Any]] = []
        cutoff = time.time() - hours * 3600

        for source, url in self.rss_feeds.items():
            try:
                rss_news = self._fetch_rss(url, symbol, source=source)
                all_news.extend(rss_news)
                logger.debug(
                    f"[NewsPipeline] {source}: {len(rss_news)} items for {symbol}"
                )
            except Exception as exc:
                logger.warning(f"[NewsPipeline] {source} fetch failed: {exc}")

        # 3. CryptoPanic API (if key provided)
        if self.cryptopanic_api_key:
            try:
                cryptopanic_news = self._fetch_cryptopanic(symbol, hours=hours)
                all_news.extend(cryptopanic_news)
                logger.debug(
                    f"[NewsPipeline] CryptoPanic: {len(cryptopanic_news)} items for {symbol}"
                )
            except Exception as exc:
                logger.warning(f"[NewsPipeline] CryptoPanic fetch failed: {exc}")

        # 4. Filter by time cutoff and deduplicate
        filtered = []
        seen_urls: set[str] = set()
        for item in all_news:
            ts = item.get("_ts", 0)
            if ts >= cutoff:
                url = item.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    # Remove internal fields
                    item.pop("_ts", None)
                    filtered.append(item)

        logger.info(
            f"[NewsPipeline] Fetched {len(filtered)} unique news items "
            f"for {symbol} (last {hours}h)"
        )
        return filtered

    def _fetch_rss(
        self, url: str, symbol: str, source: str
    ) -> list[dict[str, Any]]:
        """Fetch and parse an RSS feed, filtering for symbol mentions."""
        if not FEEDPARSER_AVAILABLE or not feedparser:
            return []

        feed = feedparser.parse(url)
        results: list[dict[str, Any]] = []

        for entry in feed.entries:
            # Check if entry mentions our symbol
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            combined = f"{title} {summary}".upper()

            # Look for symbol or full name
            symbol_variants = self._get_symbol_variants(symbol)
            if not any(v in combined for v in symbol_variants):
                continue

            # Parse timestamp
            ts = 0.0
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                import calendar
                ts = calendar.timegm(entry.published_parsed)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                import calendar
                ts = calendar.timegm(entry.updated_parsed)

            # Build content snippet
            snippet = summary[:300] if summary else title[:300]

            results.append({
                "title": title,
                "url": entry.get("link", ""),
                "source": source,
                "timestamp": entry.get("published", ""),
                "symbol": symbol,
                "content_snippet": snippet,
                "_ts": ts,
                "_raw_summary": summary,
            })

        return results

    def _fetch_cryptopanic(
        self, symbol: str, hours: int = 24
    ) -> list[dict[str, Any]]:
        """Fetch news from CryptoPanic API."""
        if not self.cryptopanic_api_key:
            return []

        client = self._get_http_client()
        if not client:
            return []

        params = {
            "auth_token": self.cryptopanic_api_key,
            "currencies": symbol,
            "public": "true",
            "filter": "important",
        }

        response = client.get(self.cryptopanic_api_url, params=params)
        response.raise_for_status()

        data = response.json()
        results: list[dict[str, Any]] = []

        for post in data.get("results", []):
            title = post.get("title", "")
            created = post.get("created_at", "")

            # Parse ISO timestamp
            ts = parse_iso_timestamp(created)

            results.append({
                "title": title,
                "url": post.get("url", ""),
                "source": "CryptoPanic",
                "timestamp": created,
                "symbol": symbol,
                "content_snippet": title[:300],
                "_ts": ts,
            })

        return results

    # ─── Sentiment Analysis ────────────────────────────────────────────

    def analyze_sentiment(
        self, news_items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analyze sentiment of news items using keyword scoring.

        Positive keywords (bull, rally, surge, ATH, adoption...) → +1
        Negative keywords (crash, dump, hack, SEC, lawsuit...) → -1

        Args:
            news_items: List of news item dicts from fetch_crypto_news.

        Returns:
            Dict with:
                overall_score: -1.0 to 1.0
                positive_count: number of positive items
                negative_count: number of negative items
                neutral_count: number of neutral items
                key_topics: list of notable topics found
                item_scores: per-item sentiment scores
        """
        if not news_items:
            return {
                "overall_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "key_topics": [],
                "item_scores": [],
            }

        positive_count = 0
        negative_count = 0
        neutral_count = 0
        item_scores: list[dict[str, Any]] = []
        all_topics: set[str] = set()

        for item in news_items:
            text = self._get_item_text(item)
            score = self._score_text(text)

            # Categorize
            if score > 0.1:
                positive_count += 1
            elif score < -0.1:
                negative_count += 1
            else:
                neutral_count += 1

            # Extract topics
            topics = self._extract_topics(text)
            all_topics.update(topics)

            item_scores.append({
                "title": item.get("title", "")[:100],
                "score": round(score, 4),
                "source": item.get("source", ""),
                "symbol": item.get("symbol", ""),
            })

        # Calculate overall score
        total = len(news_items)
        if total > 0:
            overall_score = (positive_count - negative_count) / total
        else:
            overall_score = 0.0

        # Clamp to [-1, 1]
        overall_score = max(-1.0, min(1.0, overall_score))

        return {
            "overall_score": round(overall_score, 4),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "key_topics": sorted(all_topics)[:10],
            "item_scores": item_scores,
        }

    def get_market_sentiment(
        self, symbols: list[str] | None = None, hours: int = 24
    ) -> dict[str, Any]:
        """
        Fetch and analyze news sentiment for all tracked symbols.

        Args:
            symbols: Override symbols list (default: self.symbols).
            hours: How many hours back to fetch.

        Returns:
            Dict with per-symbol sentiment and aggregate:
                {
                    "BTC": {"score": 0.6, "positive": 8, "negative": 2, ...},
                    "ETH": {"score": 0.3, "positive": 5, "negative": 3, ...},
                    "aggregate": 0.4,
                    "overall_positive": 13,
                    "overall_negative": 5,
                }
        """
        symbols = symbols or self.symbols
        result: dict[str, Any] = {}
        total_positive = 0
        total_negative = 0
        total_neutral = 0

        for symbol in symbols:
            # Fetch news
            news = self.fetch_crypto_news(symbol, hours=hours)

            # Analyze sentiment
            sentiment = self.analyze_sentiment(news)

            result[symbol] = {
                "score": sentiment["overall_score"],
                "positive": sentiment["positive_count"],
                "negative": sentiment["negative_count"],
                "neutral": sentiment["neutral_count"],
                "key_topics": sentiment["key_topics"],
                "news_count": len(news),
            }

            total_positive += sentiment["positive_count"]
            total_negative += sentiment["negative_count"]
            total_neutral += sentiment["neutral_count"]

        # Calculate aggregate
        total_news = total_positive + total_negative + total_neutral
        if total_news > 0:
            aggregate = (total_positive - total_negative) / total_news
        else:
            aggregate = 0.0

        result["aggregate"] = round(aggregate, 4)
        result["overall_positive"] = total_positive
        result["overall_negative"] = total_negative
        result["overall_neutral"] = total_neutral
        result["total_news"] = total_news

        logger.info(
            f"[NewsPipeline] Market sentiment: aggregate={aggregate:.2f}, "
            f"+{total_positive}/-{total_negative}/={total_neutral}"
        )
        return result

    # ─── Database Persistence ──────────────────────────────────────────

    def save_news_to_db(
        self, news_items: list[dict[str, Any]], db_writer: Any
    ) -> int:
        """
        Save news items to QuestDB for historical analysis.

        Args:
            news_items: News item dicts to save.
            db_writer: QuestDBWriter instance.

        Returns:
            Number of items saved.
        """
        if not news_items or db_writer is None:
            return 0

        saved = 0
        for item in news_items:
            try:
                # Compute sentiment score for storage
                text = self._get_item_text(item)
                sentiment_score = self._score_text(text)

                # Build record
                record = {
                    "title": item.get("title", "")[:500],
                    "url": item.get("url", ""),
                    "source": item.get("source", ""),
                    "symbol": item.get("symbol", ""),
                    "content_snippet": item.get("content_snippet", "")[:1000],
                    "sentiment_score": sentiment_score,
                    "timestamp": item.get("timestamp", ""),
                }

                if db_writer.write_news(record):
                    saved += 1
                else:
                    logger.warning(
                        f"[NewsPipeline] News persistence rejected for {record['symbol']} {record['url']}"
                    )
            except Exception as exc:
                logger.warning(f"[NewsPipeline] Failed to save news item: {exc}")

        logger.info(f"[NewsPipeline] Saved {saved} news items to database")
        return saved

    # ─── Internal Helpers ──────────────────────────────────────────────

    def _get_symbol_variants(self, symbol: str) -> set[str]:
        """Get common variants of a symbol for matching."""
        variants = {symbol.upper()}

        # Add common full names
        name_map = {
            "BTC": {"BITCOIN"},
            "ETH": {"ETHEREUM"},
            "SOL": {"SOLANA"},
            "XRP": {"RIPPLE"},
            "ADA": {"CARDANO"},
            "DOGE": {"DOGECOIN"},
            "AVAX": {"AVALANCHE"},
            "MATIC": {"POLYGON"},
            "DOT": {"POLKADOT"},
            "BNB": {"BINANCE"},
            "LINK": {"CHAINLINK"},
        }

        variants.update(name_map.get(symbol.upper(), set()))

        # Add ticker with USDT
        variants.add(f"{symbol}USDT")
        variants.add(f"{symbol}/USDT")
        variants.add(f"{symbol}-USDT")

        return variants

    def _get_item_text(self, item: dict[str, Any]) -> str:
        """Extract all text from a news item for analysis."""
        parts = [
            item.get("title", ""),
            item.get("content_snippet", ""),
            item.get("_raw_summary", ""),
        ]
        return " ".join(p for p in parts if p).lower()

    def _score_text(self, text: str) -> float:
        """
        Score text sentiment using keyword matching.

        Returns a value between -1.0 (very negative) and 1.0 (very positive).
        """
        if not text:
            return 0.0

        # Tokenize
        import re
        words = set(re.findall(r'\b\w+\b', text.lower()))

        positive_hits = words & {k.lower() for k in POSITIVE_KEYWORDS}
        negative_hits = words & {k.lower() for k in NEGATIVE_KEYWORDS}

        pos_count = len(positive_hits)
        neg_count = len(negative_hits)
        total = pos_count + neg_count

        if total == 0:
            return 0.0

        # Weighted score
        score = (pos_count - neg_count) / total

        # Boost if many keywords found (higher confidence)
        confidence = min(total / 5, 1.0)
        return score * confidence

    def _extract_topics(self, text: str) -> set[str]:
        """Extract notable topics from text."""
        import re
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return words & {k.lower() for k in NOTABLE_KEYWORDS}


# ─── Helpers ───────────────────────────────────────────────────────────
