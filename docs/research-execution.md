# Lumibot + CCXT + Telegram Bot: Execution Research

> **Date:** 2026-05-20
> **Purpose:** Deep research on trading execution, backtesting, and monitoring.

## Lumibot (v3.6+)
- Event-based backtesting engine — loops bar-by-bar
- **Strategy class:** `initialize()` + `on_trading_iteration()`
- Use `self.vars` for state — NEVER assign directly to `self`
- Use `self.get_datetime()` — NEVER use `datetime.now()`
- **Must set** `self.set_market("24/7")` for crypto (stops at 4PM EST otherwise)
- `get_last_price()` can return `None` — always validate
- `__future__ import annotations` crashes Lumibot — remove it
- CCXT backtesting: `CcxtBacktesting` with `ccxt_exchange="binance"`

## CCXT (v4.2+)
- `enableRateLimit=True` is MANDATORY — OFF by default
- Testnet: set `exchange.urls['api']` to `https://testnet.binance.vision/api`
- Use `exchange.amount_to_precision()` and `exchange.price_to_precision()` before ALL orders
- Order types: `market`, `limit`, `stop_loss_limit`, `take_profit_limit`
- Binance spot testnet only — no futures on testnet
- Always call `exchange.load_markets()` before trading
- Async: must `await exchange.close()` to prevent resource leaks

## python-telegram-bot (v21+)
- `ApplicationBuilder().token("...").build()` — replaces old `Updater`
- All handlers are `async def` — use `await` for all bot methods
- `ParseMode.HTML` for formatted messages
- `<b>bold</b>`, `<i>italic</i>`, `<code>code</code>` supported
- `<br>` NOT supported — use `\n` for newlines
- JobQueue for scheduled reports (requires `[job-queue]` extra)

## Common Pitfalls
- CCXT rate limiting not configured → banned by exchange
- Symbol format: Lumibot expects `BTC/USDT` (CCXT), not `BTCUSDT` (Binance)
- Minimum order sizes: Binance ~$10 minimum notional — Lumibot won't catch this
- Position updates aren't immediate after `submit_order()` — check on next iteration
- Deprecated `take_profit_price`/`stop_loss_price` — use bracket orders instead
