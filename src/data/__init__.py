"""Data pipeline module for crypto trading bot.

Provides market data collection (Binance/Cryptofeed), persistence (QuestDB),
caching (Redis), and quality validation gates.

Usage
-----
>>> from src.data import BinanceConnector, QuestDBWriter, RedisCache, QualityGates, DataConfig
>>> config = DataConfig.load_from_env()
>>> qdb = QuestDBWriter(addr=config.questdb_addr)
>>> qdb.connect()
>>> cache = RedisCache(url=config.redis_url)
>>> await cache.connect()
>>> gates = QualityGates(...)
>>> connector = BinanceConnector(...)
>>> connector.start()  # blocks
"""


def __getattr__(name):
    """Lazy imports to avoid requiring all dependencies at package level."""
    if name == "BinanceConnector":
        from .binance_connector import BinanceConnector
        return BinanceConnector
    if name == "DataConfig":
        from .config import DataConfig
        return DataConfig
    if name == "QualityGates":
        from .quality_gates import QualityGates
        return QualityGates
    if name == "QuestDBWriter":
        from .questdb_writer import QuestDBWriter
        return QuestDBWriter
    if name == "RedisCache":
        from .redis_cache import RedisCache
        return RedisCache
    if name == "NewsPipeline":
        from .news_pipeline import NewsPipeline
        return NewsPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BinanceConnector",
    "DataConfig",
    "NewsPipeline",
    "QualityGates",
    "QuestDBWriter",
    "RedisCache",
]
