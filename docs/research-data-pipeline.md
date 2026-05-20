# Cryptofeed + QuestDB + Redis: Crypto Data Pipeline Research

> **Date:** 2026-05-20
> **Purpose:** Deep research on building a real-time crypto trading data pipeline.

## Key Findings

### Cryptofeed (v2.4.1)
- Asyncio-based WebSocket feed handler for 40+ exchanges
- Callbacks receive typed objects: `Trade`, `Candle`, `Ticker` — use `.to_dict(numeric_type=float)` for serialization
- `candle.closed` flag: only store completed candles to avoid duplicates
- Binance WS: 24h forced disconnect, auto-reconnect built-in
- IP rate limits — 429 errors after extended runtime, use exponential backoff

### QuestDB (v3.0.0)
- Python client: `Sender.from_conf('http::addr=localhost:9000;auto_flush_rows=5000;auto_flush_interval=1000;')`
- Use `SYMBOL` for categorical columns, `DOUBLE` for prices
- Partition: `HOUR` for trades (high volume), `DAY` for OHLCV
- `DEDUP UPSERT KEYS` for idempotent ticker writes
- ILP over HTTP recommended over TCP (auto-retry, TLS)

### Redis
- `redis.asyncio` for full async compatibility
- HSET/HGETALL for latest price cache
- Pub/Sub for real-time price broadcasting
- Streams for durability + replay capability
- Set `maxmemory-policy` appropriately (allkeys-lru for cache)

### Integration Pattern
```
Binance WS → Cryptofeed → Quality Gates → Redis (hot) + QuestDB (persistent)
```
- Callbacks must be lightweight — offload heavy work to queues
- Store both exchange timestamp AND receipt timestamp
- Use auto-flush with 5000 rows + 1000ms interval for QuestDB

## Common Pitfalls
- Blocking callbacks freeze the event loop → always use `async/await`
- Decimal types can't be JSON-encoded → convert to float first
- QuestDB auto-flush only triggers on new row → manual flush needed for quiet periods
- Pub/Sub messages lost for late subscribers → use Streams alongside

## Docker Quick Start
```bash
docker compose up -d  # Redis + QuestDB + PostgreSQL + Qdrant
pip install -r requirements.txt
python -m src.main --data-pipeline
```
