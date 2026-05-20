#!/usr/bin/env python3
"""
AI-Powered Trading — Main Entry Point with Debate Engine Integration.

Usage:
    python -m src.main_ai --mode dryrun --strategy ai_debate    # AI debate trading
    python -m src.main_ai --mode dryrun --strategy sma_cross    # SMA with AI confirmation
    python -m src.main_ai --debate-only                         # Run debate only, no trades
    python -m src.main_ai --backtest                            # Backtest with AI

Features:
    - Integrates DebateEngine for AI-driven trading decisions
    - TradeMemory for persistent trade logging
    - WeeklyReviewer for automated performance reviews
    - DSPyOptimizer for prompt optimization
    - Risk Engine pre-trade validation
    - Telegram alerts on trade execution
    - Graceful shutdown with state save
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, cast

from loguru import logger


# ─── Logging Setup ─────────────────────────────────────────────────────
def setup_logging(log_level: str = "INFO") -> None:
    """Configure loguru logging for the AI trading bot."""
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.remove()

    # Console output with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    # File output (daily rotation)
    logger.add(
        log_dir / "ai_trading_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level=log_level,
        enqueue=True,
    )

    # Error-only file
    logger.add(
        log_dir / "ai_error_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="90 days",
        level="ERROR",
        enqueue=True,
    )


# ─── Argument Parsing ─────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Trading with Multi-Agent Debate Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main_ai --mode dryrun --strategy ai_debate
      Run AI debate trading in dry-run mode

  python -m src.main_ai --mode dryrun --strategy sma_cross
      Run SMA Cross strategy with AI debate confirmation

  python -m src.main_ai --debate-only --symbol BTC/USDT
      Run a single debate without executing trades

  python -m src.main_ai --backtest --backtest-days 90
      Backtest the AI strategy over 90 days

  python -m src.main_ai --mode dryrun --strategy ai_debate --optimize
      Run trading with DSPy prompt optimization enabled
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["dryrun", "testnet", "live"],
        default="dryrun",
        help="Trading mode (default: dryrun)",
    )

    parser.add_argument(
        "--strategy",
        choices=["sma_cross", "bbands", "ai_debate"],
        default="ai_debate",
        help="Trading strategy (default: ai_debate)",
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT", "ETH/USDT"],
        help="Trading symbols (default: BTC/USDT ETH/USDT)",
    )

    parser.add_argument(
        "--debate-only",
        action="store_true",
        help="Run debate engine only, no trade execution",
    )

    parser.add_argument(
        "--debate-symbol",
        type=str,
        default=None,
        help="Single symbol for debate-only mode",
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest instead of live trading",
    )

    parser.add_argument(
        "--backtest-days",
        type=int,
        default=90,
        help="Number of days for backtest (default: 90)",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable DSPy prompt optimization",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to settings.yaml config file",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000,
        help="Initial capital for dry-run/backtest (default: 10000)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Trading loop interval in seconds (default: 60)",
    )

    return parser.parse_args()


# ─── AITradingBot ─────────────────────────────────────────────────────
class AITradingBot:
    """
    Main AI trading bot that integrates all components.

    Orchestrates:
    - Data pipeline (Redis cache + historical data)
    - Strategy execution (SMA, BBands, or AI Debate)
    - Risk management (pre-trade validation)
    - Trade memory (logging + analytics)
    - Monitoring (Telegram alerts)
    """

    def __init__(
        self,
        config: Any,
        mode: str = "dryrun",
        strategy: str = "ai_debate",
        symbols: list[str] | None = None,
        interval: int = 60,
    ) -> None:
        """
        Initialize the AI trading bot.

        Args:
            config: AppConfig from src.config.
            mode: Trading mode (dryrun, testnet, live).
            strategy: Strategy name (sma_cross, bbands, ai_debate).
            symbols: List of trading symbols.
            interval: Loop interval in seconds.
        """
        self.config = config
        self.mode = mode
        self.strategy = strategy
        self.symbols = symbols or ["BTC/USDT"]
        self.interval = interval

        # Component references (initialized in setup)
        self._trade_memory: Any = None
        self._debate_engine: Any = None
        self._risk_engine: Any = None
        self._dry_run_executor: Any = None
        self._order_manager: Any = None
        self._telegram_bot: Any = None
        self._redis_cache: Any = None
        self._dspy_optimizer: Any = None
        self._weekly_reviewer: Any = None
        self._public_client: Any = None

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._trade_count = 0
        self._loop_count = 0
        self._last_weekly_review: datetime | None = None
        self._positions: dict[str, dict[str, Any]] = {}
        self._pending_sl_tp: list[dict[str, Any]] = []

        # Register signal handlers
        self._register_signal_handlers()

    def _register_signal_handlers(self) -> None:
        """Register OS signal handlers for graceful shutdown."""
        def _handle_signal(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._running = False
            self._shutdown_event.set()

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, OSError):
            # Signal handlers only work in main thread
            logger.debug("Signal handlers not available (not in main thread)")

    async def setup(self) -> None:
        """Initialize all bot components."""
        logger.info("🚀 Setting up AI Trading Bot...")

        # 1. Trade Memory
        from src.memory import TradeMemory

        db_url = getattr(self.config, "database_url", "postgresql://postgres:postgres@localhost:5432/trading_db")
        redis_url = getattr(self.config, "redis_url", "redis://localhost:6379")

        self._trade_memory = TradeMemory(db_url=db_url, redis_url=redis_url)
        try:
            await self._trade_memory.connect()
            logger.info("✅ TradeMemory connected")
        except Exception as exc:
            logger.warning(f"⚠️ TradeMemory connection failed, running without DB: {exc}")
            self._trade_memory = None

        # 2. Risk Engine
        from src.execution.position_sizer import PositionSizer
        from src.risk.risk_engine import RiskEngine

        self._risk_engine = RiskEngine(
            max_daily_loss_pct=self.config.risk.max_daily_loss_pct / 100,
            max_drawdown_pct=self.config.risk.max_drawdown_pct / 100,
            max_position_pct=self.config.risk.max_position_pct / 100,
            max_leverage=self.config.risk.max_leverage,
        )
        self._risk_engine.update_peak_equity(self.config.trading.initial_capital)

        self._position_sizer = PositionSizer(
            max_position_pct=self.config.risk.max_position_pct / 100,
            max_leverage=self.config.risk.max_leverage,
            daily_loss_limit_pct=self.config.risk.max_daily_loss_pct / 100,
        )
        logger.info("✅ RiskEngine and PositionSizer initialized")

        # 3. Dry-run executor or OrderManager (for trading)
        from src.execution.order_manager import OrderManager
        if self.mode == "dryrun":
            from src.execution.dry_run import DryRunExecutor

            self._dry_run_executor = DryRunExecutor(
                initial_balance=self.config.trading.initial_capital
            )
            self._order_manager = OrderManager(None, dry_run=True)
            logger.info("✅ DryRunExecutor initialized")
        else:
            from src.execution.exchange_client import ExchangeClient
            api_key = self.config.binance_testnet_api_key if self.mode == "testnet" else self.config.binance_api_key
            api_secret = self.config.binance_testnet_api_secret if self.mode == "testnet" else self.config.binance_api_secret
            if not api_key or not api_secret:
                raise ValueError(f"Missing Binance API credentials for {self.mode} mode")

            exchange_client = ExchangeClient(
                api_key=api_key,
                api_secret=api_secret,
                testnet=(self.mode == "testnet"),
            )
            try:
                exchange_client.connect()
            except Exception as e:
                logger.error(f"Failed to connect exchange client for {self.mode}: {e}")
                raise RuntimeError(
                    f"Exchange connection failed for {self.mode}"
                ) from e
            self._order_manager = OrderManager(exchange_client, dry_run=False)
            logger.info(f"✅ {self.mode} OrderManager initialized")

        # 4. Redis cache (for latest prices)
        from src.data.redis_cache import RedisCache

        self._redis_cache = RedisCache(url=redis_url)
        try:
            await self._redis_cache.connect()
            logger.info("✅ RedisCache connected")
        except Exception as exc:
            logger.warning(f"⚠️ Redis connection failed: {exc}")
            self._redis_cache = None

        # 5. Debate Engine (if using AI strategy)
        if self.strategy == "ai_debate" or self.strategy in ("sma_cross", "bbands"):
            await self._setup_debate_engine()

        # 6. Weekly Reviewer
        if self._trade_memory:
            from src.memory import WeeklyReviewer

            self._weekly_reviewer = WeeklyReviewer(self._trade_memory)
            logger.info("✅ WeeklyReviewer initialized")

        # 7. DSPy Optimizer (if enabled)
        # Optimizer setup deferred to avoid requiring dspy-ai always

        logger.info("✅ All components initialized")

    async def _setup_debate_engine(self) -> None:
        """Initialize the debate engine."""
        try:
            from src.debate import DebateConfig, DebateEngine
            from src.debate.llm_client import LLMClient

            llm_model = getattr(self.config, "litellm_model", "anthropic/claude-sonnet-4")

            llm_client = LLMClient(
                model=llm_model,
                temperature=0.7,
                max_retries=3,
            )

            debate_config = DebateConfig(
                max_rounds=3,
                llm_model=llm_model,
                symbols=self.symbols,
            )

            self._debate_engine = DebateEngine(debate_config, llm_client)
            logger.info(f"✅ DebateEngine initialized with {llm_model}")

        except ImportError as exc:
            logger.warning(f"⚠️ DebateEngine dependencies not available: {exc}")
            self._debate_engine = None
        except Exception as exc:
            logger.error(f"❌ DebateEngine setup failed: {exc}")
            self._debate_engine = None

    async def shutdown(self) -> None:
        """Graceful shutdown: save state, close connections."""
        logger.info("🛑 Shutting down AI Trading Bot...")
        self._running = False

        # Save final portfolio state
        if self._dry_run_executor:
            portfolio = self._dry_run_executor.get_portfolio()
            logger.info(
                f"📊 Final portfolio: ${portfolio['total_value']:,.2f} "
                f"(P&L: ${portfolio['total_pnl']:+,.2f})"
            )

        # Save trade state
        if self._trade_memory:
            with suppress(Exception):
                await self._trade_memory.close()

        # Close Redis
        if self._redis_cache:
            with suppress(Exception):
                await self._redis_cache.close()

        logger.info("👋 AI Trading Bot shut down complete")

    # ─── Trading Loop ──────────────────────────────────────────────────

    async def run(self) -> None:
        """Main trading loop."""
        self._running = True
        logger.info(f"🤖 AI Trading Bot started (mode={self.mode}, strategy={self.strategy})")
        logger.info(f"📊 Trading symbols: {self.symbols}")
        logger.info(f"⏱️  Loop interval: {self.interval}s")
        logger.info("Press Ctrl+C to stop...")

        try:
            while self._running:
                self._loop_count += 1

                for symbol in self.symbols:
                    if not self._running:
                        break

                    try:
                        await self._process_symbol(symbol)
                    except Exception as exc:
                        logger.error(f"Error processing {symbol}: {exc}")

                # Check for weekly review
                await self._check_weekly_review()

                # Wait for next iteration
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.interval,
                    )
                    # Event was set — we're shutting down
                    break
                except TimeoutError:
                    pass  # Normal timeout, continue loop

        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        finally:
            await self.shutdown()

    async def _process_symbol(self, symbol: str) -> None:
        """
        Process a single trading symbol through the full pipeline.

        Pipeline:
        1. Get latest price from Redis
        2. Get historical data for indicators
        3. Run strategy (SMA/BBands/AI Debate)
        4. Run debate engine (if AI strategy or confirmation)
        5. Risk Manager pre-trade check
        6. Execute order (dry-run or live)
        7. Log trade + debate to memory
        8. Send Telegram alert
        """
        # Step 1: Get latest price from Redis cache
        current_price = await self._get_latest_price(symbol)
        if current_price is None:
            if self.mode == "dryrun":
                base_prices = {
                    "BTC/USDT": 67500.0,
                    "ETH/USDT": 3450.0,
                    "SOL/USDT": 145.0,
                }
                current_price = base_prices.get(symbol, 100.0)
                import random
                current_price = round(current_price * (1 + random.uniform(-0.01, 0.01)), 2)
            else:
                try:
                    loop = asyncio.get_running_loop()
                    ticker = await loop.run_in_executor(
                        None, lambda: self._order_manager._client.fetch_ticker(symbol)
                    )
                    current_price = float(ticker.get("last", 0.0))
                except Exception as exc:
                    logger.error(f"Failed to fetch fallback ticker for {symbol}: {exc}")

        if current_price is None or current_price <= 0:
            logger.error(f"Failed to get price for {symbol}, skipping")
            return

        # Update peak equity in risk engine
        if self._risk_engine:
            portfolio = await self._get_portfolio_state()
            self._risk_engine.update_peak_equity(portfolio["total_value"])

        # Step 2: Get market data for indicators
        market_data = await self._build_market_data(symbol, current_price)

        # Step 2.5: Evaluate Stop-Loss and Take-Profit bounds
        if self._pending_sl_tp:
            symbol_orders = [o for o in self._pending_sl_tp if o["symbol"] == symbol]
            if symbol_orders:
                if self.mode == "dryrun" and self._dry_run_executor:
                    try:
                        triggered = self._dry_run_executor.simulate_sl_tp(
                            symbol_orders, {symbol: current_price}
                        )
                        if triggered:
                            for trg in triggered:
                                trg_type = trg.get("triggered_by", "sl/tp")
                                logger.info(
                                    f"🚨 [{symbol}] {trg_type.upper()} triggered! "
                                    f"Closed position @ {current_price}"
                                )
                                if symbol in self._positions:
                                    del self._positions[symbol]
                                self._pending_sl_tp = [
                                    o for o in self._pending_sl_tp if o["symbol"] != symbol
                                ]
                                await self._send_trade_alert(symbol, trg, {"action": "SELL", "confidence": 100, "rounds": 0})
                    except Exception as e:
                        logger.error(f"Error evaluating dry-run SL/TP for {symbol}: {e}")
                else:
                    # Live / testnet mode: manual price-crossing check and CCXT execution
                    triggered_orders = []
                    for order in symbol_orders:
                        stop_price = order["stop_price"]
                        direction = order.get("direction", "")
                        triggered_price = False
                        if direction == "below" and current_price <= stop_price or direction == "above" and current_price >= stop_price:
                            triggered_price = True

                        if triggered_price:
                            triggered_orders.append(order)

                    if triggered_orders:
                        for trg_order in triggered_orders:
                            logger.info(f"🚨 [LIVE] {trg_order['type'].upper()} trigger price reached: ${current_price:.2f} (stop was ${trg_order['stop_price']:.2f})")
                            try:
                                side_to_execute = trg_order["side"]
                                amount_to_execute = trg_order["quantity"]
                                order_symbol = symbol
                                portfolio = await self._get_portfolio_state()

                                logger.info(f"🚨 Sending real market {side_to_execute.upper()} SL/TP close order to exchange for {symbol}...")
                                loop = asyncio.get_running_loop()
                                order = await loop.run_in_executor(
                                    None,
                                    partial(
                                        self._order_manager.create_market_order,
                                        symbol=order_symbol,
                                        side=side_to_execute,
                                        amount=amount_to_execute,
                                    ),
                                )

                                avg_fill_price = float(order.get("average", current_price) or current_price)
                                filled_qty = float(order.get("filled", amount_to_execute))

                                pnl = 0.0
                                pnl_pct = 0.0
                                if symbol in self._positions:
                                    entry_price = self._positions[symbol]["entry_price"]
                                    if side_to_execute == "sell":
                                        pnl = (avg_fill_price - entry_price) * filled_qty
                                        pnl_pct = ((avg_fill_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
                                    else:
                                        pnl = (entry_price - avg_fill_price) * filled_qty
                                        pnl_pct = ((entry_price - avg_fill_price) / entry_price * 100) if entry_price > 0 else 0.0

                                trg_result = {
                                    "trade_id": order.get("id"),
                                    "symbol": symbol,
                                    "side": side_to_execute,
                                    "quantity": filled_qty,
                                    "price": avg_fill_price,
                                    "revenue": filled_qty * avg_fill_price,
                                    "pnl": pnl,
                                    "pnl_pct": pnl_pct,
                                    "cash_total": portfolio["cash"] + (filled_qty * avg_fill_price if side_to_execute == "sell" else -filled_qty * avg_fill_price),
                                    "timestamp": datetime.now(UTC).isoformat(),
                                    "triggered_by": trg_order.get("type"),
                                    "order_info": order,
                                }

                                if symbol in self._positions:
                                    del self._positions[symbol]

                                self._pending_sl_tp = [
                                    o for o in self._pending_sl_tp if o["symbol"] != symbol
                                ]

                                await self._send_trade_alert(symbol, trg_result, {"action": side_to_execute.upper(), "confidence": 100, "rounds": 0})
                                break
                            except Exception as exc:
                                logger.error(f"Failed to execute real SL/TP close order on exchange: {exc}")

        # Step 3: Run strategy
        if self.strategy == "ai_debate":
            signal_action = "BUY"
        else:
            signal_action = await self._run_strategy(symbol, market_data)

        if signal_action == "HOLD":
            logger.debug(f"[{symbol}] Strategy says HOLD")
            return

        # Step 4: Run debate engine for confirmation
        debate_result = await self._run_debate(symbol, market_data)
        if debate_result is None:
            logger.warning(f"[{symbol}] Debate engine unavailable, skipping trade")
            return

        # Step 5: Risk Manager pre-trade check
        approved, reason = await self._run_risk_check(
            symbol,
            debate_result.get("action", "HOLD"),
            current_price,
            stop_loss=debate_result.get("stop_loss", 0.0)
        )
        await self._log_debate_decision(symbol, debate_result)

        if not approved:
            logger.info(f"[{symbol}] Trade rejected by Risk Engine: {reason}")
            return

        # Step 6: Execute order
        trade_result = await self._execute_trade(
            symbol,
            debate_result.get("action", "HOLD"),
            current_price,
            debate_result,
        )

        if trade_result is None:
            return

        # Step 7: Log trade + debate to memory
        await self._log_trade(symbol, trade_result, debate_result)

        # Step 8: Send Telegram alert
        await self._send_trade_alert(symbol, trade_result, debate_result)

    async def _get_latest_price(self, symbol: str) -> float | None:
        """Get latest price from Redis cache."""
        if self._redis_cache is None:
            # Fallback: return a simulated price for dry-run
            return None

        try:
            # Convert symbol format: BTC/USDT -> BTC-USDT
            redis_symbol = symbol.replace("/", "-")
            data = await self._redis_cache.get_latest_price(redis_symbol)
            if data and "price" in data:
                return float(data["price"])
        except Exception as exc:
            logger.debug(f"Failed to get price from Redis for {symbol}: {exc}")

        return None

    async def _fetch_real_indicators(self, symbol: str) -> dict[str, Any]:
        """Fetch historical candles from Binance and compute technical indicators."""
        try:
            import ccxt
            import pandas as pd
            import ta

            if not hasattr(self, "_public_client") or self._public_client is None:
                self._public_client = ccxt.binance({"enableRateLimit": True})

            # Fetch last 100 candles (1m timeframe)
            loop = asyncio.get_running_loop()
            ohlcv = await loop.run_in_executor(
                None,
                partial(self._public_client.fetch_ohlcv, symbol, timeframe="1m", limit=100),
            )

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

            # Compute indicators
            rsi_series = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
            sma_fast_series = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
            sma_slow_series = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()

            macd_obj = ta.trend.MACD(df["close"], window_slow=26, window_fast=12)
            macd_series = macd_obj.macd()

            bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2.0)
            bb_upper_series = bb.bollinger_hband()
            bb_lower_series = bb.bollinger_lband()

            volume_sma = ta.trend.SMAIndicator(df["volume"], window=20).sma_indicator()

            latest_price = float(df["close"].iloc[-1])
            latest_volume = float(df["volume"].iloc[-1])
            avg_volume = float(volume_sma.iloc[-1]) if not pd.isna(volume_sma.iloc[-1]) else latest_volume

            return {
                "price": latest_price,
                "rsi": float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0,
                "sma_fast": float(sma_fast_series.iloc[-1]) if not pd.isna(sma_fast_series.iloc[-1]) else latest_price,
                "sma_slow": float(sma_slow_series.iloc[-1]) if not pd.isna(sma_slow_series.iloc[-1]) else latest_price,
                "macd": float(macd_series.iloc[-1]) if not pd.isna(macd_series.iloc[-1]) else 0.0,
                "bb_upper": float(bb_upper_series.iloc[-1]) if not pd.isna(bb_upper_series.iloc[-1]) else latest_price * 1.02,
                "bb_lower": float(bb_lower_series.iloc[-1]) if not pd.isna(bb_lower_series.iloc[-1]) else latest_price * 0.98,
                "volume": latest_volume,
                "volume_high": latest_volume > avg_volume * 1.5,
            }
        except Exception as exc:
            logger.warning(f"Failed to fetch/compute real indicators for {symbol}: {exc}")
            import random
            price = getattr(self, "_last_prices", {}).get(symbol, 67000.0)
            return {
                "price": price,
                "rsi": round(random.uniform(35, 65), 1),
                "sma_fast": price,
                "sma_slow": price,
                "macd": 0.0,
                "bb_upper": price * 1.02,
                "bb_lower": price * 0.98,
                "volume": 1.0,
                "volume_high": False,
            }

    async def _build_market_data(
        self, symbol: str, current_price: float
    ) -> dict[str, Any]:
        """Build market data dict with technical indicators."""
        market_data: dict[str, Any] = {
            "symbol": symbol,
            "price": current_price,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        real_ind = await self._fetch_real_indicators(symbol)
        if "price" in real_ind and real_ind["price"] > 0:
            market_data["price"] = real_ind["price"]
        market_data["indicators"] = real_ind

        return market_data

    async def _run_strategy(
        self, symbol: str, market_data: dict[str, Any]
    ) -> str:
        """
        Run the configured trading strategy.

        Returns: BUY, SELL, or HOLD
        """
        if self.strategy == "ai_debate":
            # AI Debate strategy: defer decision to debate engine
            return "BUY"  # Debate engine will confirm/reject

        elif self.strategy == "sma_cross":
            # SMA Cross: use traditional signal, AI confirms
            return self._sma_cross_signal(market_data)

        elif self.strategy == "bbands":
            # Bollinger Bands: use traditional signal, AI confirms
            return self._bbands_signal(market_data)

        return "HOLD"

    def _sma_cross_signal(self, market_data: dict[str, Any]) -> str:
        """SMA crossover signal."""
        price = market_data.get("price", 0)
        indicators = market_data.get("indicators", {})

        # Simple simulation: use price position relative to BB as proxy
        bb_upper = indicators.get("bb_upper", price * 1.02)
        bb_lower = indicators.get("bb_lower", price * 0.98)

        if price < bb_lower:
            return "BUY"
        elif price > bb_upper:
            return "SELL"
        return "HOLD"

    def _bbands_signal(self, market_data: dict[str, Any]) -> str:
        """Bollinger Bands signal."""
        return self._sma_cross_signal(market_data)  # Same logic for now

    async def _run_debate(
        self, symbol: str, market_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Run the debate engine for a trading decision.

        Returns: Debate result dict or None if debate engine unavailable.
        """
        if self._debate_engine is None:
            logger.warning(f"[{symbol}] Debate engine not available")
            return None

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._debate_engine.run_debate(
                    market_data=market_data,
                    symbol=symbol,
                ),
            )

            # Convert DebateResult to dict
            if hasattr(result, "model_dump"):
                return cast(dict[str, Any], result.model_dump())
            elif hasattr(result, "dict"):
                return cast(dict[str, Any], result.dict())
            else:
                return {
                    "action": getattr(result, "action", "HOLD"),
                    "confidence": getattr(result, "confidence", 50.0),
                    "reason": getattr(result, "reason", ""),
                    "stop_loss": getattr(result, "stop_loss", 0),
                    "take_profit": getattr(result, "take_profit", 0),
                    "bull_argument": getattr(result, "bull_argument", ""),
                    "bear_argument": getattr(result, "bear_argument", ""),
                    "devil_argument": getattr(result, "devil_argument", ""),
                    "risk_decision": getattr(result, "risk_decision", "APPROVE"),
                }

        except Exception as exc:
            logger.error(f"[{symbol}] Debate engine error: {exc}")
            return None

    async def _get_portfolio_state(self) -> dict[str, Any]:
        """Get the current portfolio state (cash, positions, total equity)."""
        if self.mode == "dryrun":
            if self._dry_run_executor:
                return cast(dict[str, Any], self._dry_run_executor.get_portfolio())
            return {
                "cash": self.config.trading.initial_capital,
                "positions": {},
                "total_value": self.config.trading.initial_capital,
                "initial_balance": self.config.trading.initial_capital,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
            }

        # For testnet or live modes, query Binance via CCXT
        try:
            loop = asyncio.get_running_loop()
            raw_balance = await loop.run_in_executor(
                None, self._order_manager._client.fetch_balance
            )

            quote_asset = "USDT"
            if self.symbols:
                parts = self.symbols[0].split("/")
                if len(parts) > 1:
                    quote_asset = parts[1]

            cash = float(raw_balance.get(quote_asset, {}).get("free", 0.0))

            positions_data = {}
            total_value = cash

            for asset, bal in raw_balance.items():
                if asset in ("free", "used", "total", "info", quote_asset):
                    continue

                total_qty = float(bal.get("total", 0.0))
                if total_qty > 0.000001:
                    symbol = f"{asset}/{quote_asset}"
                    current_price = await self._get_latest_price(symbol)
                    if current_price is None:
                        try:
                            ticker_symbol = symbol
                            ticker = await loop.run_in_executor(
                                None,
                                partial(self._order_manager._client.fetch_ticker, ticker_symbol)
                            )
                            current_price = float(ticker.get("last", 0.0))
                        except Exception:
                            current_price = 0.0

                    market_value = total_qty * current_price
                    total_value += market_value

                    positions_data[symbol] = {
                        "quantity": total_qty,
                        "avg_price": current_price,
                        "cost_basis": market_value,
                        "market_value": market_value,
                        "unrealized_pnl": 0.0,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }

            initial_balance = getattr(self.config.trading, "initial_capital", 10000.0)
            total_pnl = total_value - initial_balance
            total_pnl_pct = (total_pnl / initial_balance * 100) if initial_balance > 0 else 0.0

            return {
                "cash": cash,
                "positions": positions_data,
                "total_value": total_value,
                "initial_balance": initial_balance,
                "total_pnl": total_pnl,
                "total_pnl_pct": total_pnl_pct,
            }
        except Exception as e:
            logger.error(f"Failed to fetch portfolio state from exchange: {e}")
            return {
                "cash": self.config.trading.initial_capital,
                "positions": {},
                "total_value": self.config.trading.initial_capital,
                "initial_balance": self.config.trading.initial_capital,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
            }

    async def _run_risk_check(
        self,
        symbol: str,
        action: str,
        price: float,
        stop_loss: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Run risk manager pre-trade check.

        Returns: (approved, reason)
        """
        if self._risk_engine is None:
            return False, "Risk engine not available"

        # Get current portfolio state
        portfolio = await self._get_portfolio_state()
        positions = portfolio.get("positions", {})

        # Use PositionSizer to calculate quantity
        sl_price = stop_loss
        if not sl_price or sl_price <= 0:
            sl_price = price * 0.98 if action == "BUY" else price * 1.02

        # Get historical performance stats if available
        win_rate = 0.5
        if self._trade_memory:
            try:
                stats = await self._trade_memory.get_performance_summary()
                if stats and stats.total_trades > 5:
                    win_rate = stats.win_rate / 100
            except Exception:
                pass

        if not hasattr(self, "_position_sizer") or self._position_sizer is None:
            from src.execution.position_sizer import PositionSizer
            self._position_sizer = PositionSizer(
                max_position_pct=self.config.risk.max_position_pct / 100,
                max_leverage=self.config.risk.max_leverage,
                daily_loss_limit_pct=self.config.risk.max_daily_loss_pct / 100,
            )

        size_res = self._position_sizer.calc_position_size(
            strategy=self.strategy,
            symbol=symbol,
            entry_price=price,
            stop_loss_price=sl_price,
            equity=portfolio["total_value"],
            win_rate=win_rate,
        )
        quantity = size_res.size_base

        if quantity <= 0:
            return False, "Insufficient cash for minimum quantity"

        approved, reason = self._risk_engine.pre_trade_checks(
            symbol=symbol,
            side=action.lower(),
            quantity=quantity,
            price=price,
            current_equity=portfolio["total_value"],
            start_equity=self.config.trading.initial_capital,
            positions=positions,
        )

        return approved, reason

    async def _execute_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        debate_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Execute a trade (dry-run or live).

        Returns: Trade result dict or None if execution failed.
        """
        portfolio = await self._get_portfolio_state()
        sl_price = debate_result.get("stop_loss", 0.0)
        if not sl_price or sl_price <= 0:
            sl_price = price * 0.98 if action == "BUY" else price * 1.02

        # Get historical performance stats if available
        win_rate = 0.5
        if self._trade_memory:
            try:
                stats = await self._trade_memory.get_performance_summary()
                if stats and stats.total_trades > 5:
                    win_rate = stats.win_rate / 100
            except Exception:
                pass

        if not hasattr(self, "_position_sizer") or self._position_sizer is None:
            from src.execution.position_sizer import PositionSizer
            self._position_sizer = PositionSizer(
                max_position_pct=self.config.risk.max_position_pct / 100,
                max_leverage=self.config.risk.max_leverage,
                daily_loss_limit_pct=self.config.risk.max_daily_loss_pct / 100,
            )

        size_res = self._position_sizer.calc_position_size(
            strategy=self.strategy,
            symbol=symbol,
            entry_price=price,
            stop_loss_price=sl_price,
            equity=portfolio["total_value"],
            win_rate=win_rate,
        )
        quantity = size_res.size_base

        if self.mode != "dryrun":
            with suppress(Exception):
                quantity = self._order_manager._precision_amount(symbol, quantity)

        if quantity <= 0:
            logger.warning(f"[{symbol}] Insufficient funds for {action} (quantity={quantity:.6f})")
            return None

        try:
            ts = datetime.now(UTC).isoformat()

            if self.mode == "dryrun":
                if self._dry_run_executor is None:
                    logger.error(f"[{symbol}] No dry-run executor available")
                    return None

                if action == "BUY":
                    result = self._dry_run_executor.simulate_buy(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        timestamp=ts,
                    )
                    result["strategy"] = self.strategy
                    result["ai_confidence"] = debate_result.get("confidence", 50)
                    result["stop_loss"] = debate_result.get("stop_loss")
                    result["take_profit"] = debate_result.get("take_profit")

                    self._positions[symbol] = {
                        "side": "LONG",
                        "quantity": quantity,
                        "entry_price": price,
                        "entry_time": ts,
                    }

                    # Register SL/TP orders if provided
                    sl = debate_result.get("stop_loss", 0)
                    tp = debate_result.get("take_profit", 0)
                    if sl and sl > 0:
                        self._pending_sl_tp.append({
                            "id": f"sl_{symbol.replace('/', '_')}_{int(time.time())}",
                            "symbol": symbol,
                            "side": "sell",
                            "type": "stop_loss",
                            "stop_price": float(sl),
                            "quantity": quantity,
                            "direction": "below",
                        })
                        logger.info(f"Registered Stop Loss at ${sl:.2f} for {symbol}")
                    if tp and tp > 0:
                        self._pending_sl_tp.append({
                            "id": f"tp_{symbol.replace('/', '_')}_{int(time.time())}",
                            "symbol": symbol,
                            "side": "sell",
                            "type": "take_profit",
                            "stop_price": float(tp),
                            "quantity": quantity,
                            "direction": "above",
                        })
                        logger.info(f"Registered Take Profit at ${tp:.2f} for {symbol}")

                    self._trade_count += 1
                    return cast(dict[str, Any], result)

                elif action == "SELL":
                    # Close position if exists
                    if symbol in self._positions:
                        pos = self._positions[symbol]
                        result = self._dry_run_executor.simulate_sell(
                            symbol=symbol,
                            quantity=pos["quantity"],
                            price=price,
                            timestamp=ts,
                        )
                        del self._positions[symbol]
                        # Clear pending SL/TP for this symbol since position is closed
                        self._pending_sl_tp = [o for o in self._pending_sl_tp if o["symbol"] != symbol]
                    else:
                        logger.warning(
                            f"[{symbol}] SELL ignored: no dry-run long position to close"
                        )
                        return None

                    result["strategy"] = self.strategy
                    result["ai_confidence"] = debate_result.get("confidence", 50)
                    self._trade_count += 1
                    return cast(dict[str, Any], result)

                else:
                    logger.debug(f"[{symbol}] No execution needed for HOLD")
                    return None
            else:
                # Real exchange orders routing
                loop = asyncio.get_running_loop()

                if action == "BUY":
                    logger.info(f"🚨 Sending real market BUY order to exchange for {symbol}...")
                    order = await loop.run_in_executor(
                        None,
                        lambda: self._order_manager.create_market_order(
                            symbol=symbol,
                            side="buy",
                            amount=quantity
                        )
                    )

                    filled_qty = float(order.get("filled", quantity))
                    avg_fill_price = float(order.get("average", price) or price)

                    result = {
                        "trade_id": order.get("id"),
                        "symbol": symbol,
                        "side": "buy",
                        "quantity": filled_qty,
                        "price": avg_fill_price,
                        "cost": filled_qty * avg_fill_price,
                        "cash_remaining": portfolio["cash"] - (filled_qty * avg_fill_price),
                        "timestamp": ts,
                        "order_info": order,
                    }

                    self._positions[symbol] = {
                        "side": "LONG",
                        "quantity": filled_qty,
                        "entry_price": avg_fill_price,
                        "entry_time": ts,
                    }

                    sl = debate_result.get("stop_loss", 0)
                    tp = debate_result.get("take_profit", 0)
                    if sl and sl > 0:
                        self._pending_sl_tp.append({
                            "id": f"sl_{symbol.replace('/', '_')}_{int(time.time())}",
                            "symbol": symbol,
                            "side": "sell",
                            "type": "stop_loss",
                            "stop_price": float(sl),
                            "quantity": filled_qty,
                            "direction": "below",
                        })
                    if tp and tp > 0:
                        self._pending_sl_tp.append({
                            "id": f"tp_{symbol.replace('/', '_')}_{int(time.time())}",
                            "symbol": symbol,
                            "side": "sell",
                            "type": "take_profit",
                            "stop_price": float(tp),
                            "quantity": filled_qty,
                            "direction": "above",
                        })

                    self._trade_count += 1
                    result["strategy"] = self.strategy
                    result["ai_confidence"] = debate_result.get("confidence", 50)
                    result["stop_loss"] = debate_result.get("stop_loss")
                    result["take_profit"] = debate_result.get("take_profit")
                    return result

                elif action == "SELL":
                    logger.info(f"🚨 Sending real market SELL order to exchange for {symbol}...")

                    sell_quantity = 0.0
                    if symbol in self._positions:
                        sell_quantity = self._positions[symbol]["quantity"]
                    elif symbol in portfolio.get("positions", {}):
                        sell_quantity = float(
                            portfolio["positions"][symbol].get("quantity", 0.0)
                        )
                    else:
                        logger.warning(f"[{symbol}] SELL rejected: no position to close")
                        return None

                    order = await loop.run_in_executor(
                        None,
                        lambda: self._order_manager.create_market_order(
                            symbol=symbol,
                            side="sell",
                            amount=sell_quantity
                        )
                    )

                    filled_qty = float(order.get("filled", sell_quantity))
                    avg_fill_price = float(order.get("average", price) or price)

                    pnl = 0.0
                    pnl_pct = 0.0
                    if symbol in self._positions:
                        entry_price = self._positions[symbol]["entry_price"]
                        pnl = (avg_fill_price - entry_price) * filled_qty
                        pnl_pct = ((avg_fill_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
                        del self._positions[symbol]
                        self._pending_sl_tp = [o for o in self._pending_sl_tp if o["symbol"] != symbol]

                    result = {
                        "trade_id": order.get("id"),
                        "symbol": symbol,
                        "side": "sell",
                        "quantity": filled_qty,
                        "price": avg_fill_price,
                        "revenue": filled_qty * avg_fill_price,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "cash_total": portfolio["cash"] + (filled_qty * avg_fill_price),
                        "timestamp": ts,
                        "order_info": order,
                    }

                    self._trade_count += 1
                    result["strategy"] = self.strategy
                    result["ai_confidence"] = debate_result.get("confidence", 50)
                    return result

                return None

        except Exception as exc:
            logger.error(f"[{symbol}] Trade execution failed: {exc}")
            return None

    async def _log_trade(
        self,
        symbol: str,
        trade_result: dict[str, Any],
        debate_result: dict[str, Any],
    ) -> None:
        """Log an executed trade to memory."""
        if self._trade_memory is None:
            return

        try:
            ts = datetime.now(UTC).isoformat()

            # Log trade
            trade_data = {
                "timestamp": ts,
                "symbol": symbol,
                "side": trade_result.get("side", "BUY").upper(),
                "quantity": trade_result.get("quantity", 0),
                "price": trade_result.get("price", 0),
                "pnl": trade_result.get("pnl", 0),
                "pnl_pct": trade_result.get("pnl_pct", 0),
                "strategy": self.strategy,
                "mode": self.mode,
                "ai_confidence": trade_result.get("ai_confidence"),
                "debate_result": debate_result,
                "stop_loss": trade_result.get("stop_loss"),
                "take_profit": trade_result.get("take_profit"),
            }

            await self._trade_memory.log_trade(trade_data)

        except Exception as exc:
            logger.error(f"Failed to log trade: {exc}")

    async def _log_debate_decision(
        self,
        symbol: str,
        debate_result: dict[str, Any],
    ) -> None:
        """Log every AI debate decision before execution outcomes."""
        if self._trade_memory is None:
            return

        try:
            rounds = debate_result.get("rounds", [])
            debate_data = {
                "timestamp": datetime.now(UTC).isoformat(),
                "symbol": symbol,
                "bull_arg": debate_result.get("bull_argument", ""),
                "bear_arg": debate_result.get("bear_argument", ""),
                "devil_arg": debate_result.get("devil_argument", ""),
                "judge_action": debate_result.get("action", "HOLD"),
                "judge_confidence": debate_result.get("confidence", 50),
                "risk_action": debate_result.get("risk_decision", "APPROVE"),
                "risk_reasoning": debate_result.get("risk_reasoning", ""),
                "rounds": len(rounds) if isinstance(rounds, list) else int(rounds or 0),
                "latency_seconds": debate_result.get("metadata", {}).get("total_time_seconds", 0.0),
            }

            await self._trade_memory.log_debate(debate_data)
        except Exception as exc:
            logger.error(f"Failed to log debate decision: {exc}")

    async def _send_trade_alert(
        self,
        symbol: str,
        trade_result: dict[str, Any],
        debate_result: dict[str, Any],
    ) -> None:
        """Send Telegram alert for executed trade."""
        if self._telegram_bot is None:
            return

        try:
            side = trade_result.get("side", "BUY").upper()
            price = trade_result.get("price", 0)
            quantity = trade_result.get("quantity", 0)
            pnl = trade_result.get("pnl")
            confidence = debate_result.get("confidence", 0)

            emoji = "🟢" if side == "BUY" else "🔴"
            message = (
                f"{emoji} <b>TRADE EXECUTED</b>\n"
                f"Symbol: {symbol}\n"
                f"Action: {side} {quantity}\n"
                f"Price: ${price:,.2f}\n"
                f"AI Confidence: {confidence:.1f}%\n"
                f"Strategy: {self.strategy}\n"
                f"Mode: {self.mode}"
            )

            if pnl is not None:
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                message += f"\n{pnl_emoji} P&L: ${pnl:+,.2f}"

            await self._telegram_bot.send_alert(message)

        except Exception as exc:
            logger.debug(f"Failed to send Telegram alert: {exc}")

    async def _check_weekly_review(self) -> None:
        """Check if it's time for a weekly review."""
        if self._weekly_reviewer is None:
            return

        now = datetime.now(UTC)

        # Run review every 7 days
        if (
            self._last_weekly_review is None
            or (now - self._last_weekly_review).days >= 7
        ):
            try:
                logger.info("📊 Running weekly review...")
                report = await self._weekly_reviewer.generate_report()
                self._weekly_reviewer.save_report(report)
                self._last_weekly_review = now

                # Extract insights
                insights = await self._weekly_reviewer.extract_insights()
                logger.info(f"📝 Weekly insights: {len(insights)} found")
                for insight in insights[:3]:
                    logger.info(f"  - {insight[:100]}...")

            except Exception as exc:
                logger.error(f"Weekly review failed: {exc}")


# ─── Debate-Only Mode ─────────────────────────────────────────────────
async def run_debate_only(
    config: Any,
    symbol: str = "BTC/USDT",
) -> None:
    """Run a single debate without trade execution."""
    logger.info(f"🧠 Running debate-only for {symbol}...")

    try:
        from src.debate import DebateConfig, DebateEngine
        from src.debate.llm_client import LLMClient

        llm_model = getattr(config, "litellm_model", "anthropic/claude-sonnet-4")

        llm_client = LLMClient(model=llm_model)
        debate_config = DebateConfig(max_rounds=3, symbols=[symbol])
        engine = DebateEngine(debate_config, llm_client)

        # Build sample market data
        market_data = {
            "price": 67500.0,
            "rsi": 45,
            "macd": 0.0,
            "volume": 1000,
            "bb_upper": 68000,
            "bb_lower": 67000,
        }

        result = engine.run_debate(market_data=market_data, symbol=symbol)

        logger.info("=" * 60)
        logger.info("🧠 DEBATE RESULT")
        logger.info("=" * 60)
        logger.info(f"  Action:     {result.action}")
        logger.info(f"  Confidence: {result.confidence:.1f}%")
        logger.info(f"  Reason:     {result.reason[:200]}...")
        logger.info(f"  Stop Loss:  ${result.stop_loss:,.2f}")
        logger.info(f"  Take Profit: ${result.take_profit:,.2f}")
        logger.info(f"  Risk:       {result.risk_decision}")
        logger.info("=" * 60)

    except ImportError as exc:
        logger.error(f"Debate engine dependencies missing: {exc}")
    except Exception as exc:
        logger.error(f"Debate failed: {exc}")


# ─── Backtest Mode ─────────────────────────────────────────────────────
def run_backtest(config: Any, days: int = 90) -> None:
    """Run backtest with AI strategy."""
    logger.info(f"📈 Starting AI Backtest ({days} days)...")

    from src.strategy.backtest import BacktestRunner
    from src.strategy.metrics import MetricsCalculator

    # Use SMA Cross as proxy for backtest (AI debate is too slow for backtest)
    from src.strategy.sma_cross import SMACrossStrategy

    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=days)

    symbol = config.trading.symbols[0] if config.trading.symbols else "BTC/USDT"

    logger.info(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"📊 Symbol: {symbol}")
    logger.info(f"💰 Capital: ${config.trading.initial_capital:,.2f}")

    params = {
        "symbol": symbol.replace("/", ""),
        "quote_asset": "USDT",
        "initial_capital": config.trading.initial_capital,
        "sma_fast": config.strategy.sma_fast,
        "sma_slow": config.strategy.sma_slow,
        "rsi_period": config.strategy.rsi_period,
    }

    try:
        runner = BacktestRunner(
            strategy_class=SMACrossStrategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            parameters=params,
            initial_capital=config.trading.initial_capital,
        )

        results = runner.run()

        if results:
            metrics = MetricsCalculator.summarize(
                results.get("trades", []),
                results.get("equity_curve", None),
            )

            logger.info("=" * 60)
            logger.info("📊 BACKTEST RESULTS")
            logger.info("=" * 60)
            logger.info(f"  Total Return:     {metrics.get('total_return_pct', 0):.2f}%")
            logger.info(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"  Max Drawdown:     {metrics.get('max_drawdown_pct', 0):.2f}%")
            logger.info(f"  Win Rate:         {metrics.get('win_rate_pct', 0):.1f}%")
            logger.info(f"  Total Trades:     {metrics.get('total_trades', 0)}")
            logger.info(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
            logger.info("=" * 60)

            output_path = Path(__file__).parent.parent / "logs" / "ai_backtest_results.html"
            runner.plot_results(output_path=str(output_path))
            logger.info(f"📊 Chart saved to {output_path}")

    except Exception as exc:
        logger.error(f"Backtest failed: {exc}")


# ─── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    """Entry point for AI trading bot."""
    args = parse_args()

    # Load config
    from src.config import load_config

    config = load_config(config_path=args.config)

    # Override with CLI args
    config.trading.mode = args.mode
    config.trading.initial_capital = args.initial_capital
    config.monitoring.log_level = args.log_level

    if args.symbols:
        config.trading.symbols = args.symbols
    if args.strategy:
        config.strategy.name = args.strategy

    # Setup logging
    setup_logging(log_level=config.monitoring.log_level)

    logger.info("=" * 60)
    logger.info("🤖 AI-Powered Trading Architecture")
    logger.info("=" * 60)
    logger.info(f"  Mode:     {config.trading.mode}")
    logger.info(f"  Strategy: {config.strategy.name}")
    logger.info(f"  Symbols:  {config.trading.symbols}")
    logger.info(f"  Capital:  ${config.trading.initial_capital:,.2f}")
    if args.optimize:
        logger.info("  DSPy:     ✅ Optimization enabled")
    logger.info("=" * 60)

    # Route to appropriate mode
    if args.debate_only:
        symbol = args.debate_symbol or config.trading.symbols[0]
        asyncio.run(run_debate_only(config, symbol=symbol))

    elif args.backtest:
        run_backtest(config, days=args.backtest_days)

    else:
        # Main trading loop
        bot = AITradingBot(
            config=config,
            mode=args.mode,
            strategy=args.strategy,
            symbols=args.symbols,
            interval=args.interval,
        )

        async def _run():
            await bot.setup()
            await bot.run()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as exc:
            logger.error(f"Fatal error: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()
