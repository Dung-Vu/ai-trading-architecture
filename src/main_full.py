#!/usr/bin/env python3
"""
Full Unified AI Trading Bot — Combines Phase 1 + Phase 2 + Phase 4.

Integrates all modules into a single, production-ready trading system:
    - Data Pipeline (Redis cache + historical data)
    - Strategies (SMA Cross, Bollinger Bands, AI Debate)
    - Debate Engine (LangGraph multi-agent)
    - Risk Management (pre-trade validation + kill switch)
    - Execution Layer (dry-run / testnet / live)
    - Mem0 Self-Learning Memory (semantic trade recall)
    - Knowledge Graph (pattern storage)
    - News & Sentiment Pipeline (multi-source)
    - Auto-Tuner (weekly optimization)
    - Monitoring (Telegram alerts)

Usage:
    python -m src.main_full --mode dryrun --strategy sma_cross
    python -m src.main_full --mode dryrun --strategy ai_debate
    python -m src.main_full --mode testnet --strategy bbands
    python -m src.main_full --mode dryrun --strategy ai_debate --symbols BTC/USDT ETH/USDT SOL/USDT
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger


# ─── Logging Setup ─────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO") -> None:
    """Configure loguru logging for the unified trading bot."""
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
        log_dir / "full_trading_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level=log_level,
        enqueue=True,
    )

    # Error-only file
    logger.add(
        log_dir / "full_error_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="90 days",
        level="ERROR",
        enqueue=True,
    )


# ─── Argument Parsing ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Full Unified AI Trading Bot (Phase 1 + 2 + 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main_full --mode dryrun --strategy sma_cross
      Dry-run with SMA Cross strategy (classic)

  python -m src.main_full --mode dryrun --strategy ai_debate
      Dry-run with AI debate engine (Phase 2)

  python -m src.main_full --mode testnet --strategy ai_debate
      Testnet trading with full AI stack

  python -m src.main_full --mode dryrun --strategy bbands
      Dry-run with Bollinger Bands strategy

  python -m src.main_full --mode dryrun --strategy ai_debate \\
      --symbols BTC/USDT ETH/USDT --interval 30
      Fast-cycle AI trading on BTC + ETH
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
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        help="Trading symbols (default: BTC/USDT ETH/USDT SOL/USDT)",
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
        help="Initial capital (default: 10000)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Trading loop interval in seconds (default: 60)",
    )

    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable Mem0 self-learning memory",
    )

    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Disable news/sentiment pipeline",
    )

    parser.add_argument(
        "--no-autotune",
        action="store_true",
        help="Disable auto-tuner",
    )

    parser.add_argument(
        "--debate-only",
        action="store_true",
        help="Run debate engine only, no trade execution",
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest instead of live trading",
    )

    return parser.parse_args()


# ─── FullTradingBot ───────────────────────────────────────────────────

class FullTradingBot:
    """
    Full unified trading bot combining all phases.

    Orchestrates the complete trading pipeline:
    1. Fetch market data (prices, indicators)
    2. Fetch news sentiment
    3. Run debate (with news sentiment as additional context)
    4. Risk check
    5. Execute order
    6. Log to memory (Mem0 + PostgreSQL)
    7. Auto-tune check (weekly)
    8. Send alerts (Telegram)
    """

    def __init__(
        self,
        config: Any,
        mode: str = "dryrun",
        strategy: str = "ai_debate",
        symbols: list[str] | None = None,
        interval: int = 60,
        enable_memory: bool = True,
        enable_news: bool = True,
        enable_autotune: bool = True,
    ) -> None:
        """Initialize the full trading bot."""
        self.config = config
        self.mode = mode
        self.strategy_name = strategy
        self.symbols = symbols or ["BTC/USDT"]
        self.interval = interval
        self.enable_memory = enable_memory
        self.enable_news = enable_news
        self.enable_autotune = enable_autotune

        # ─── Core Components ───────────────────────────────────────────
        self._trade_memory: Any = None          # PostgreSQL memory
        self._mem0_memory: Any = None            # Mem0 semantic memory
        self._knowledge_graph: Any = None        # Pattern knowledge graph
        self._debate_engine: Any = None          # LangGraph debate
        self._risk_engine: Any = None            # Risk management
        self._kill_switch: Any = None            # Kill switch
        self._executor: Any = None               # Trade executor
        self._news_pipeline: Any = None          # News + sentiment
        self._auto_tuner: Any = None             # Auto-optimizer
        self._telegram_bot: Any = None           # Telegram alerts
        self._redis_cache: Any = None            # Redis cache
        self._strategies: dict[str, Any] = {}    # Strategy instances

        # ─── State ─────────────────────────────────────────────────────
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._trade_count = 0
        self._loop_count = 0
        self._last_autotune: datetime | None = None
        self._last_memory_cleanup: datetime | None = None
        self._positions: dict[str, dict[str, Any]] = {}  # symbol → position info
        self._portfolio = {
            "cash": config.trading.initial_capital,
            "total_value": config.trading.initial_capital,
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
        }

        # Register signal handlers
        self._register_signal_handlers()

        logger.info(f"🤖 FullTradingBot created (mode={mode}, strategy={strategy_name})")

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
            logger.debug("Signal handlers not available (not in main thread)")

    # ─── Setup ─────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Initialize all bot components."""
        logger.info("🚀 Setting up Full Unified Trading Bot...")

        # 1. PostgreSQL Trade Memory
        await self._setup_trade_memory()

        # 2. Mem0 Self-Learning Memory
        if self.enable_memory:
            await self._setup_mem0_memory()

        # 3. Knowledge Graph
        await self._setup_knowledge_graph()

        # 4. Risk Engine + Kill Switch
        self._setup_risk()

        # 5. Trade Executor
        self._setup_executor()

        # 6. Strategies
        self._setup_strategies()

        # 7. Debate Engine
        if self.strategy_name == "ai_debate" or self.strategy_name in ("sma_cross", "bbands"):
            await self._setup_debate_engine()

        # 8. News Pipeline
        if self.enable_news:
            self._setup_news_pipeline()

        # 9. Auto-Tuner
        if self.enable_autotune:
            self._setup_auto_tuner()

        # 10. Redis Cache
        await self._setup_redis()

        # 11. Load previous state
        await self._load_state()

        logger.info("✅ All components initialized")
        self._print_startup_summary()

    async def _setup_trade_memory(self) -> None:
        """Initialize PostgreSQL trade memory."""
        from src.memory import TradeMemory

        db_url = getattr(self.config, "database_url", "postgresql://postgres:***@localhost:5432/trading_db")
        redis_url = getattr(self.config, "redis_url", "redis://localhost:6379")

        self._trade_memory = TradeMemory(db_url=db_url, redis_url=redis_url)
        try:
            await self._trade_memory.connect()
            logger.info("✅ TradeMemory (PostgreSQL) connected")
        except Exception as exc:
            logger.warning(f"⚠️ TradeMemory connection failed: {exc}")
            self._trade_memory = None

    async def _setup_mem0_memory(self) -> None:
        """Initialize Mem0 semantic memory."""
        try:
            from src.memory.mem0_memory import Mem0Memory

            qdrant_url = getattr(self.config, "qdrant_url", "http://localhost:6333")
            self._mem0_memory = Mem0Memory(qdrant_url=qdrant_url)

            stats = self._mem0_memory.get_stats()
            logger.info(f"✅ Mem0Memory initialized: {stats['store_type']}")
        except Exception as exc:
            logger.warning(f"⚠️ Mem0Memory setup failed: {exc}")
            self._mem0_memory = None

    async def _setup_knowledge_graph(self) -> None:
        """Initialize knowledge graph."""
        try:
            from src.memory.knowledge_graph import KnowledgeGraph

            self._knowledge_graph = KnowledgeGraph()

            # Try to load previous graph
            graph_path = Path(__file__).parent.parent / "config" / "knowledge_graph.json"
            if graph_path.exists():
                self._knowledge_graph.load_from_file(str(graph_path))
                logger.info(f"✅ KnowledgeGraph loaded from file ({len(self._knowledge_graph)} patterns)")
            else:
                logger.info("✅ KnowledgeGraph initialized (empty)")
        except Exception as exc:
            logger.warning(f"⚠️ KnowledgeGraph setup failed: {exc}")
            self._knowledge_graph = None

    def _setup_risk(self) -> None:
        """Initialize risk engine and kill switch."""
        from src.risk.risk_engine import RiskEngine
        from src.risk.kill_switch import KillSwitch

        self._risk_engine = RiskEngine(
            max_daily_loss_pct=self.config.risk.max_daily_loss_pct / 100,
            max_drawdown_pct=self.config.risk.max_drawdown_pct / 100,
            max_position_pct=self.config.risk.max_position_pct / 100,
            max_leverage=self.config.risk.max_leverage,
        )

        self._kill_switch = KillSwitch()
        logger.info("✅ RiskEngine + KillSwitch initialized")

    def _setup_executor(self) -> None:
        """Initialize trade executor."""
        from src.execution.dry_run import DryRunExecutor
        from src.execution.order_manager import OrderManager

        self._executor = DryRunExecutor(
            initial_balance=self.config.trading.initial_capital
        )
        self._order_manager = OrderManager()
        logger.info(f"✅ {self.mode} executor initialized (${self.config.trading.initial_capital:,.2f})")

    def _setup_strategies(self) -> None:
        """Initialize trading strategies."""
        from src.strategy.sma_cross import SMACrossStrategy
        from src.strategy.bbands import BBandsStrategy

        self._strategies["sma_cross"] = SMACrossStrategy(
            symbol=self.symbols[0],
            sma_fast=self.config.strategy.sma_fast,
            sma_slow=self.config.strategy.sma_slow,
            rsi_period=self.config.strategy.rsi_period,
        )

        self._strategies["bbands"] = BBandsStrategy(
            symbol=self.symbols[0],
            rsi_period=self.config.strategy.rsi_period,
            rsi_overbought=self.config.strategy.rsi_overbought,
            rsi_oversold=self.config.strategy.rsi_oversold,
        )

        logger.info(f"✅ Strategies initialized: {list(self._strategies.keys())}")

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
            logger.warning(f"⚠️ DebateEngine not available: {exc}")
            self._debate_engine = None
        except Exception as exc:
            logger.error(f"❌ DebateEngine setup failed: {exc}")
            self._debate_engine = None

    def _setup_news_pipeline(self) -> None:
        """Initialize news and sentiment pipeline."""
        try:
            from src.data.news_pipeline import NewsPipeline

            # Extract ticker symbols from trading pairs
            tickers = [s.split("/")[0] for s in self.symbols]

            self._news_pipeline = NewsPipeline(
                symbols=tickers,
                languages=["en"],
                cryptopanic_api_key=None,  # From env if available
            )
            logger.info(f"✅ NewsPipeline initialized for {tickers}")
        except Exception as exc:
            logger.warning(f"⚠️ NewsPipeline setup failed: {exc}")
            self._news_pipeline = None

    def _setup_auto_tuner(self) -> None:
        """Initialize auto-tuner."""
        try:
            from src.autotune import AutoTuner

            debate_cfg = {}
            if self._debate_engine:
                debate_cfg = {
                    "model": self._debate_engine.config.llm_model,
                    "max_rounds": self._debate_engine.config.max_rounds,
                }

            self._auto_tuner = AutoTuner(
                trade_memory=self._trade_memory,
                debate_config=debate_cfg,
                strategy_name=self.strategy_name,
            )
            logger.info("✅ AutoTuner initialized")
        except Exception as exc:
            logger.warning(f"⚠️ AutoTuner setup failed: {exc}")
            self._auto_tuner = None

    async def _setup_redis(self) -> None:
        """Initialize Redis cache."""
        try:
            from src.data.redis_cache import RedisCache

            redis_url = getattr(self.config, "redis_url", "redis://localhost:6379")
            self._redis_cache = RedisCache(url=redis_url)
            try:
                await self._redis_cache.connect()
                logger.info("✅ RedisCache connected")
            except Exception as exc:
                logger.warning(f"⚠️ Redis connection failed: {exc}")
                self._redis_cache = None
        except Exception as exc:
            logger.warning(f"⚠️ RedisCache setup failed: {exc}")
            self._redis_cache = None

    async def _load_state(self) -> None:
        """Load previous bot state if available."""
        state_path = Path(__file__).parent.parent / "config" / "bot_state.json"
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)

                self._portfolio = state.get("portfolio", self._portfolio)
                self._positions = state.get("positions", {})
                self._trade_count = state.get("trade_count", 0)
                self._last_autotune = (
                    datetime.fromisoformat(state["last_autotune"])
                    if state.get("last_autotune") else None
                )

                logger.info(
                    f"📂 Loaded previous state: {self._trade_count} trades, "
                    f"portfolio=${self._portfolio['total_value']:,.2f}"
                )
            except Exception as exc:
                logger.warning(f"Failed to load bot state: {exc}")

    # ─── Shutdown ──────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Graceful shutdown: save state, close connections."""
        logger.info("🛑 Shutting down Full Trading Bot...")
        self._running = False

        # Save state
        await self._save_state()

        # Save knowledge graph
        if self._knowledge_graph:
            graph_path = Path(__file__).parent.parent / "config" / "knowledge_graph.json"
            self._knowledge_graph.save_to_file(str(graph_path))
            logger.info("💾 KnowledgeGraph saved")

        # Summary
        portfolio = self._executor.get_portfolio() if self._executor else {}
        trade_log = self._executor.get_trade_log() if self._executor else []

        logger.info("=" * 60)
        logger.info("📊 FINAL SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total Value:    ${portfolio.get('total_value', 0):,.2f}")
        logger.info(f"  Cash:           ${portfolio.get('cash', 0):,.2f}")
        logger.info(f"  Total P&L:      ${portfolio.get('total_pnl', 0):+,2f}")
        logger.info(f"  Total Trades:   {len(trade_log)}")
        logger.info(f"  Loops Executed: {self._loop_count}")
        logger.info("=" * 60)

        # Close connections
        if self._trade_memory:
            try:
                await self._trade_memory.close()
            except Exception:
                pass

        if self._redis_cache:
            try:
                await self._redis_cache.close()
            except Exception:
                pass

        logger.info("👋 Full Trading Bot shut down complete")

    async def _save_state(self) -> None:
        """Save current bot state to disk."""
        state_path = Path(__file__).parent.parent / "config" / "bot_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        portfolio = self._executor.get_portfolio() if self._executor else self._portfolio

        state = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "portfolio": portfolio,
            "positions": self._positions,
            "trade_count": self._trade_count,
            "loop_count": self._loop_count,
            "last_autotune": self._last_autotune.isoformat() if self._last_autotune else None,
            "strategy": self.strategy_name,
            "mode": self.mode,
        }

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    # ─── Startup Summary ───────────────────────────────────────────────

    def _print_startup_summary(self) -> None:
        """Print a comprehensive startup summary."""
        logger.info("")
        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║" + "  🚀 FULL UNIFIED AI TRADING BOT  ".ljust(58) + "║")
        logger.info("╠" + "═" * 58 + "╣")
        logger.info(f"║  Mode:        {self.mode:<44} ║")
        logger.info(f"║  Strategy:    {self.strategy_name:<44} ║")
        logger.info(f"║  Symbols:     {', '.join(self.symbols):<44} ║")
        logger.info(f"║  Capital:     ${self.config.trading.initial_capital:,.2f}{' ' * (44 - len(f'${self.config.trading.initial_capital:,.2f}'))} ║")
        logger.info(f"║  Interval:    {self.interval}s{' ' * 41} ║")
        logger.info("╟" + "─" * 58 + "╢")
        logger.info(f"║  Mem0 Memory: {'✅ Enabled' if self._mem0_memory else '❌ Disabled':<44} ║")
        logger.info(f"║  Knowledge G: {'✅ Enabled' if self._knowledge_graph else '❌ Disabled':<44} ║")
        logger.info(f"║  News/Sentim: {'✅ Enabled' if self._news_pipeline else '❌ Disabled':<44} ║")
        logger.info(f"║  Auto-Tuner:  {'✅ Enabled' if self._auto_tuner else '❌ Disabled':<44} ║")
        logger.info(f"║  Debate Eng:  {'✅ Enabled' if self._debate_engine else '❌ Disabled':<44} ║")
        logger.info(f"║  Risk Engine: ✅ Enabled{' ' * 39} ║")
        logger.info(f"║  Kill Switch: ✅ Enabled{' ' * 39} ║")
        logger.info("╚" + "═" * 58 + "╝")
        logger.info("")

    # ─── Main Trading Loop ─────────────────────────────────────────────

    async def run(self) -> None:
        """Main trading loop."""
        self._running = True
        logger.info(f"🤖 Trading loop started (mode={self.mode}, strategy={self.strategy_name})")
        logger.info(f"📊 Symbols: {self.symbols}")
        logger.info(f"⏱️  Loop interval: {self.interval}s")
        logger.info("Press Ctrl+C to stop...")

        try:
            while self._running:
                self._loop_count += 1
                loop_start = time.time()

                # Check kill switch
                if self._kill_switch and self._kill_switch.is_active():
                    logger.error("🚨 Kill switch activated! Stopping immediately.")
                    break

                for symbol in self.symbols:
                    if not self._running:
                        break

                    try:
                        await self._process_symbol(symbol)
                    except Exception as exc:
                        logger.error(f"Error processing {symbol}: {exc}")

                # Periodic maintenance tasks
                await self._check_weekly_review()
                await self._check_autotune()
                await self._check_memory_cleanup()

                # Wait for next iteration
                loop_duration = time.time() - loop_start
                wait_time = max(0.1, self.interval - loop_duration)

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=wait_time,
                    )
                    break  # Event was set — shutting down
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue

        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        finally:
            await self.shutdown()

    async def _process_symbol(self, symbol: str) -> None:
        """
        Process a single trading symbol through the full unified pipeline.

        Pipeline:
        1. Fetch market data (prices, indicators)
        2. Fetch news sentiment
        3. Run strategy → generate signal
        4. Query knowledge graph for similar patterns
        5. Query Mem0 for similar past trades
        6. Run debate (with news sentiment + memory context)
        7. Risk Manager pre-trade check
        8. Execute order
        9. Log to memory (PostgreSQL + Mem0)
        10. Update knowledge graph
        11. Send Telegram alert
        """
        logger.debug(f"🔄 Processing {symbol} (loop #{self._loop_count})")

        # Step 1: Get market data
        market_data = await self._build_market_data(symbol)
        if market_data is None:
            logger.debug(f"  No market data for {symbol}, skipping")
            return

        current_price = market_data.get("price", 0)
        indicators = market_data.get("indicators", {})

        # Step 2: Fetch news sentiment
        sentiment = await self._fetch_sentiment(symbol)

        # Step 3: Run strategy
        strategy = self._strategies.get(self.strategy_name)
        if strategy is None:
            logger.warning(f"Strategy {self.strategy_name} not found")
            return

        signal_action = strategy.generate_signal(
            price=current_price,
            indicators=indicators,
            market_data=market_data,
        )

        if signal_action == "HOLD":
            logger.debug(f"  [{symbol}] Strategy says HOLD")
            return

        logger.info(
            f"  [{symbol}] Strategy signal: {signal_action} "
            f"(price=${current_price:,.2f})"
        )

        # Step 4: Query knowledge graph for similar patterns
        kg_context = self._query_knowledge_graph(indicators)

        # Step 5: Query Mem0 for similar past trades
        mem0_context = self._query_mem0_memory(symbol, indicators)

        # Step 6: Run debate with enriched context
        debate_result = await self._run_debate(
            symbol, market_data, sentiment, kg_context, mem0_context
        )
        if debate_result is None:
            logger.warning(f"  [{symbol}] Debate engine unavailable")
            return

        final_action = debate_result.get("action", "HOLD")
        if final_action == "HOLD":
            logger.info(f"  [{symbol}] Debate says HOLD")
            return

        # Step 7: Risk check
        approved, reason = self._run_risk_check(
            symbol, final_action, current_price
        )
        if not approved:
            logger.info(f"  [{symbol}] Risk rejected: {reason}")
            return

        # Step 8: Execute trade
        trade_result = await self._execute_trade(
            symbol, final_action, current_price, debate_result
        )
        if trade_result is None:
            return

        # Step 9: Log to memory
        await self._log_trade(symbol, trade_result, debate_result, market_data)

        # Step 10: Update knowledge graph
        self._update_knowledge_graph(indicators, final_action, trade_result)

        # Step 11: Send alert
        await self._send_alert(symbol, trade_result, debate_result)

        self._trade_count += 1

    # ─── Pipeline Steps ────────────────────────────────────────────────

    async def _build_market_data(self, symbol: str) -> dict[str, Any] | None:
        """Build market data dict with price and indicators."""
        # Try Redis cache first
        if self._redis_cache:
            try:
                redis_symbol = symbol.replace("/", "-")
                price_data = await self._redis_cache.get_latest_price(redis_symbol)
                if price_data and "price" in price_data:
                    price = float(price_data["price"])
                else:
                    price = None
            except Exception:
                price = None
        else:
            price = None

        # Fallback: simulate price for dry-run
        if price is None:
            price = self._simulate_price(symbol)
            if price is None:
                return None

        # Calculate indicators (simplified for demo)
        import random
        random.seed(hash(symbol + str(int(time.time()) // 60)) % (2**32))

        indicators = {
            "price": price,
            "rsi": round(random.uniform(20, 80), 1),
            "sma_fast": round(price * (1 + random.uniform(-0.02, 0.02)), 2),
            "sma_slow": round(price * (1 + random.uniform(-0.05, 0.05)), 2),
            "volume": round(random.uniform(0.5, 3.0), 2),
            "volume_high": random.choice([True, False]),
        }

        market_conditions = {
            "trend": random.choice(["uptrend", "downtrend", "sideways"]),
            "volatility": random.choice(["low", "medium", "high"]),
            "volume_high": indicators["volume_high"],
        }

        return {
            "symbol": symbol,
            "price": price,
            "indicators": indicators,
            "market_conditions": market_conditions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _simulate_price(self, symbol: str) -> float | None:
        """Simulate a realistic price for dry-run mode."""
        base_prices = {
            "BTC/USDT": 67500,
            "ETH/USDT": 3450,
            "SOL/USDT": 145,
            "XRP/USDT": 0.52,
            "ADA/USDT": 0.45,
        }

        base = base_prices.get(symbol)
        if base is None:
            return None

        import random
        random.seed(int(time.time()) % 1000)
        return round(base * (1 + random.uniform(-0.03, 0.03)), 2)

    async def _fetch_sentiment(self, symbol: str) -> dict[str, Any]:
        """Fetch news sentiment for a symbol."""
        if not self._news_pipeline:
            return {"score": 0.0, "positive": 0, "negative": 0}

        try:
            ticker = symbol.split("/")[0]
            # News fetching is synchronous — run in executor
            loop = asyncio.get_event_loop()
            news_items = await loop.run_in_executor(
                None, self._news_pipeline.fetch_crypto_news, ticker, 24
            )

            sentiment = await loop.run_in_executor(
                None, self._news_pipeline.analyze_sentiment, news_items
            )

            logger.info(
                f"  [{symbol}] Sentiment: score={sentiment['overall_score']:.2f}, "
                f"+{sentiment['positive_count']}/-{sentiment['negative_count']}"
            )
            return sentiment
        except Exception as exc:
            logger.warning(f"  [{symbol}] Sentiment fetch failed: {exc}")
            return {"score": 0.0, "positive": 0, "negative": 0}

    def _query_knowledge_graph(self, indicators: dict) -> list[dict]:
        """Query knowledge graph for matching patterns."""
        if not self._knowledge_graph:
            return []

        # Build condition string from current indicators
        rsi = indicators.get("rsi", 50)
        volume_high = indicators.get("volume_high", False)

        condition_parts = []
        if rsi < 30:
            condition_parts.append("RSI oversold")
        elif rsi > 70:
            condition_parts.append("RSI overbought")

        if volume_high:
            condition_parts.append("volume high")

        if not condition_parts:
            return []

        condition = " AND ".join(condition_parts)
        return self._knowledge_graph.query_pattern(condition)

    def _query_mem0_memory(self, symbol: str, indicators: dict) -> str:
        """Query Mem0 for similar past trades."""
        if not self._mem0_memory:
            return ""

        rsi = indicators.get("rsi", 50)
        query = f"{symbol} RSI {rsi:.0f}"

        if rsi < 30:
            query += " oversold bounce"
        elif rsi > 70:
            query += " overbought rejection"

        try:
            similar = self._mem0_memory.query_similar_trades(query, limit=3)
            if similar:
                summary_parts = []
                for s in similar:
                    outcome = s.get("outcome", "pending")
                    pnl = s.get("pnl", 0)
                    summary_parts.append(
                        f"Past: {outcome} (${pnl:+.2f})"
                    )
                return "; ".join(summary_parts)
        except Exception as exc:
            logger.debug(f"Mem0 query failed: {exc}")

        return ""

    async def _run_debate(
        self,
        symbol: str,
        market_data: dict,
        sentiment: dict,
        kg_context: list[dict],
        mem0_context: str,
    ) -> dict[str, Any] | None:
        """Run debate engine with full context."""
        # Build enriched market data
        enriched = dict(market_data)
        enriched["news_sentiment"] = sentiment
        enriched["knowledge_graph_patterns"] = kg_context
        enriched["memory_context"] = mem0_context

        if self._debate_engine:
            try:
                positions = self._positions.get(symbol, {})
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._debate_engine.run_debate(
                        market_data=enriched,
                        current_positions={symbol: positions},
                        portfolio=self._portfolio,
                        symbol=symbol,
                    ),
                )

                return {
                    "action": result.action,
                    "confidence": result.confidence,
                    "reasoning": result.reason,
                    "stop_loss": result.stop_loss,
                    "take_profit": result.take_profit,
                    "risk_decision": result.risk_decision,
                    "bull_argument": result.bull_argument,
                    "bear_argument": result.bear_argument,
                    "devil_argument": result.devil_argument,
                    "rounds": len(result.rounds),
                }
            except Exception as exc:
                logger.warning(f"Debate engine error: {exc}")
                return None

        # Fallback: use strategy signal directly
        return {
            "action": market_data.get("signal", "HOLD"),
            "confidence": 60.0,
            "reasoning": f"Strategy {self.strategy_name} signal (no debate engine)",
            "stop_loss": 0,
            "take_profit": 0,
            "risk_decision": "APPROVE",
            "bull_argument": "",
            "bear_argument": "",
            "devil_argument": "",
            "rounds": 0,
        }

    def _run_risk_check(
        self, symbol: str, action: str, price: float
    ) -> tuple[bool, str]:
        """Run risk engine pre-trade check."""
        if not self._risk_engine:
            return True, ""

        check_result = self._risk_engine.check_pre_trade(
            action=action,
            symbol=symbol,
            price=price,
            portfolio=self._portfolio,
            positions=self._positions,
        )

        return check_result.approved, check_result.reason

    async def _execute_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        debate_result: dict,
    ) -> dict[str, Any] | None:
        """Execute a trade (dry-run or live)."""
        if not self._executor:
            return None

        # Determine quantity (10% of portfolio for demo)
        quantity_pct = 0.10
        available = self._portfolio["cash"] * quantity_pct
        quantity = available / price if price > 0 else 0

        if quantity <= 0:
            logger.warning(f"Insufficient funds for {symbol} {action}")
            return None

        try:
            if action == "BUY":
                result = self._executor.buy(symbol, quantity, price)
                self._positions[symbol] = {
                    "side": "LONG",
                    "quantity": quantity,
                    "entry_price": price,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                }
            elif action == "SELL":
                # Close position if exists
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    pnl = (price - pos["entry_price"]) * pos["quantity"]
                    result = self._executor.sell(symbol, pos["quantity"], price)
                    del self._positions[symbol]
                else:
                    # Short sell
                    result = self._executor.sell(symbol, quantity, price)
                    self._positions[symbol] = {
                        "side": "SHORT",
                        "quantity": quantity,
                        "entry_price": price,
                        "entry_time": datetime.now(timezone.utc).isoformat(),
                    }
            else:
                return None

            return {
                "symbol": symbol,
                "side": action,
                "quantity": quantity,
                "price": price,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            logger.error(f"Trade execution failed for {symbol}: {exc}")
            return None

    async def _log_trade(
        self,
        symbol: str,
        trade_result: dict,
        debate_result: dict,
        market_data: dict,
    ) -> None:
        """Log trade to all memory systems."""
        trade_record = {
            "symbol": symbol,
            "side": trade_result["side"],
            "quantity": trade_result["quantity"],
            "price": trade_result["price"],
            "strategy": self.strategy_name,
            "mode": self.mode,
            "indicators": market_data.get("indicators", {}),
            "market_conditions": market_data.get("market_conditions", {}),
            "stop_loss": debate_result.get("stop_loss"),
            "take_profit": debate_result.get("take_profit"),
            "ai_confidence": debate_result.get("confidence", 0),
        }

        # PostgreSQL memory
        if self._trade_memory:
            try:
                await self._trade_memory.log_trade(trade_record)
            except Exception as exc:
                logger.warning(f"Failed to log trade to PostgreSQL: {exc}")

        # Mem0 semantic memory
        if self._mem0_memory:
            try:
                self._mem0_memory.add_trade_memory(trade_record, debate_result)
            except Exception as exc:
                logger.warning(f"Failed to log trade to Mem0: {exc}")

        # Log debate
        debate_record = {
            "symbol": symbol,
            "bull_arg": debate_result.get("bull_argument", "")[:500],
            "bear_arg": debate_result.get("bear_argument", "")[:500],
            "devil_arg": debate_result.get("devil_argument", "")[:500],
            "judge_action": debate_result.get("action", "HOLD"),
            "judge_confidence": debate_result.get("confidence", 50),
            "risk_action": debate_result.get("risk_decision", "APPROVE"),
            "risk_reasoning": "",
            "rounds": debate_result.get("rounds", 0),
        }

        if self._trade_memory:
            try:
                await self._trade_memory.log_debate(debate_record)
            except Exception as exc:
                logger.warning(f"Failed to log debate to PostgreSQL: {exc}")

    def _update_knowledge_graph(
        self,
        indicators: dict,
        action: str,
        trade_result: dict,
    ) -> None:
        """Add pattern to knowledge graph."""
        if not self._knowledge_graph:
            return

        # Build condition string
        rsi = indicators.get("rsi", 50)
        volume_high = indicators.get("volume_high", False)

        conditions = []
        if rsi < 30:
            conditions.append("RSI < 30")
        elif rsi > 70:
            conditions.append("RSI > 70")

        if volume_high:
            conditions.append("volume > avg")

        if conditions:
            condition_str = " AND ".join(conditions)
            self._knowledge_graph.add_pattern(
                condition=condition_str,
                action=action,
                outcome="pending",  # Will be updated when trade closes
                confidence=0.5,
            )

    async def _send_alert(
        self,
        symbol: str,
        trade_result: dict,
        debate_result: dict,
    ) -> None:
        """Send Telegram alert for trade execution."""
        if not self.config.monitoring.telegram_enabled:
            return

        try:
            from src.monitoring.telegram_bot import TelegramBot
            from src.monitoring.alert_formatter import format_trade_alert

            bot = TelegramBot(
                bot_token=self.config.monitoring.telegram_bot_token,
                chat_id=self.config.monitoring.telegram_chat_id,
            )

            alert = format_trade_alert(trade_result, debate_result)
            await bot.send_message(alert)
            logger.info(f"  [{symbol}] Telegram alert sent")
        except Exception as exc:
            logger.warning(f"Failed to send alert: {exc}")

    # ─── Maintenance Tasks ─────────────────────────────────────────────

    async def _check_weekly_review(self) -> None:
        """Run weekly review if enough time has passed."""
        from src.memory import WeeklyReviewer

        if not self._trade_memory or not hasattr(self, "_last_weekly_review"):
            return

        now = datetime.now(timezone.utc)
        last = getattr(self, "_last_weekly_review", None)

        if last is None or (now - last).days >= 7:
            try:
                reviewer = WeeklyReviewer(self._trade_memory)
                report = await reviewer.generate_report()

                logger.info("📊 Weekly Review:")
                logger.info(f"  {report.get('summary', 'No data')}")

                self._last_weekly_review = now
            except Exception as exc:
                logger.warning(f"Weekly review failed: {exc}")

    async def _check_autotune(self) -> None:
        """Run auto-tune weekly optimization cycle."""
        if not self._auto_tuner:
            return

        now = datetime.now(timezone.utc)

        # Run every 7 days
        if self._last_autotune is None or (now - self._last_autotune).days >= 7:
            try:
                # Check if decay detected
                needs_optimization = self._auto_tuner.detect_strategy_decay()

                if needs_optimization:
                    logger.info("🔧 Strategy decay detected — running optimization...")
                    report = self._auto_tuner.weekly_optimization_cycle()
                    logger.info(f"  Optimization complete: {report.get('status', 'unknown')}")
                else:
                    # Get recommendations regardless
                    recommendations = self._auto_tuner.get_optimization_recommendations()
                    for rec in recommendations:
                        logger.info(f"  💡 {rec}")

                self._last_autotune = now
                await self._save_state()
            except Exception as exc:
                logger.warning(f"Auto-tune check failed: {exc}")

    async def _check_memory_cleanup(self) -> None:
        """Clean up old Mem0 memories periodically."""
        if not self._mem0_memory:
            return

        now = datetime.now(timezone.utc)

        # Run cleanup every 30 days
        if self._last_memory_cleanup is None or (now - self._last_memory_cleanup).days >= 30:
            try:
                removed = self._mem0_memory.clear_old_memories(days=90)
                logger.info(
                    f"🧹 Memory cleanup: removed {removed} old memories (>90 days)"
                )
                self._last_memory_cleanup = now
            except Exception as exc:
                logger.warning(f"Memory cleanup failed: {exc}")


# ─── Backtest Mode ─────────────────────────────────────────────────────

def run_backtest(config, args):
    """Run backtest with the selected strategy."""
    from src.strategy.sma_cross import SMACrossStrategy
    from src.strategy.bbands import BBandsStrategy
    from src.strategy.backtest import BacktestRunner
    from src.strategy.metrics import MetricsCalculator

    logger.info("📈 Starting Backtest...")

    if config.strategy.name == "sma_cross":
        strategy_class = SMACrossStrategy
    elif config.strategy.name == "bbands":
        strategy_class = BBandsStrategy
    else:
        raise ValueError(f"Unknown strategy: {config.strategy.name}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    logger.info(f"📅 Backtest: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"📊 Symbol: {config.trading.symbols[0]}")
    logger.info(f"💰 Capital: ${config.trading.initial_capital:,.2f}")

    params = {
        "symbol": config.trading.symbols[0],
        "sma_fast": config.strategy.sma_fast,
        "sma_slow": config.strategy.sma_slow,
        "rsi_period": config.strategy.rsi_period,
    }

    runner = BacktestRunner(
        strategy_class=strategy_class,
        symbol=config.trading.symbols[0],
        start_date=start_date,
        end_date=end_date,
        parameters=params,
        initial_capital=config.trading.initial_capital,
    )

    results = runner.run()

    if results:
        metrics = MetricsCalculator.summarize(
            results.get("trades", []),
            results.get("equity_curve"),
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

        output_path = Path(__file__).parent.parent / "logs" / "backtest_results.html"
        runner.plot_results(output_path=str(output_path))
        logger.info(f"📊 Chart saved to {output_path}")


# ─── Main Entry Point ──────────────────────────────────────────────────

def main():
    """Main entry point."""
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
    logger.info("🚀 Full Unified AI Trading Architecture (Phase 1+2+4)")
    logger.info("=" * 60)
    logger.info(f"  Mode:        {config.trading.mode}")
    logger.info(f"  Strategy:    {config.strategy.name}")
    logger.info(f"  Symbols:     {config.trading.symbols}")
    logger.info(f"  Memory:      {'Enabled' if not args.no_memory else 'Disabled'}")
    logger.info(f"  News:        {'Enabled' if not args.no_news else 'Disabled'}")
    logger.info(f"  Auto-tune:   {'Enabled' if not args.no_autotune else 'Disabled'}")
    logger.info("=" * 60)

    # Route to appropriate mode
    if args.backtest:
        run_backtest(config, args)
    elif args.debate_only:
        logger.info("🧠 Debate-only mode — running single debate...")
        # Run a single debate without trading
        from src.debate import DebateConfig, DebateEngine
        from src.debate.llm_client import LLMClient

        llm_client = LLMClient(model=config.litellm_model)
        debate_config = DebateConfig(
            max_rounds=3,
            llm_model=config.litellm_model,
            symbols=config.trading.symbols,
        )
        engine = DebateEngine(debate_config, llm_client)

        sample_data = {
            "price": 67500,
            "rsi": 28,
            "sma_fast": 67200,
            "sma_slow": 68000,
            "volume": 2.5,
        }
        result = engine.run_debate(sample_data, symbol=config.trading.symbols[0])
        logger.info(f"Debate result: {result.action} (confidence={result.confidence:.0f}%)")
        logger.info(f"Reasoning: {result.reason[:200]}")
    else:
        # Full unified bot
        bot = FullTradingBot(
            config=config,
            mode=args.mode,
            strategy=args.strategy,
            symbols=args.symbols,
            interval=args.interval,
            enable_memory=not args.no_memory,
            enable_news=not args.no_news,
            enable_autotune=not args.no_autotune,
        )

        async def async_main():
            await bot.setup()
            await bot.run()

        asyncio.run(async_main())


if __name__ == "__main__":
    main()
