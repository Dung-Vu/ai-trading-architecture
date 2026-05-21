"""Full unified trading bot orchestration."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.bot.pipeline import FullTradingPipelineMixin
from src.bot.maintenance import FullTradingMaintenanceMixin
from src.bot_base import BaseTradingBot
from src.config import (
    get_default_database_url,
    get_default_exchange_name,
    get_default_mem0_embedding_model,
    get_default_news_rss_feeds,
    get_default_qdrant_url,
    get_default_redis_url,
)
from src.debate.runtime import build_debate_engine


class FullTradingBot(FullTradingPipelineMixin, FullTradingMaintenanceMixin, BaseTradingBot):
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
        self._order_manager: Any = None          # Order manager
        self._news_pipeline: Any = None          # News + sentiment
        self._auto_tuner: Any = None             # Auto-optimizer
        self._weekly_reviewer: Any = None        # Weekly review generator
        self._telegram_bot: Any = None           # Telegram alerts
        self._redis_cache: Any = None            # Redis cache
        self._strategies: dict[str, dict[str, Any]] = {}  # Strategy instances by symbol

        # ─── State ─────────────────────────────────────────────────────
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._trade_count = 0
        self._loop_count = 0
        self._last_autotune: datetime | None = None
        self._last_weekly_review: datetime | None = None
        self._last_memory_cleanup: datetime | None = None
        self._positions: dict[str, dict[str, Any]] = {}  # symbol → position info
        self._pending_sl_tp: list[dict[str, Any]] = []    # pending stop-loss / take-profit orders
        self._simulation_seed = time.time_ns()
        self._indicator_seed = self._simulation_seed
        self._simulated_price_cache: dict[str, float] = {}
        self._indicator_cache: dict[tuple[str, float], dict[str, Any]] = {}
        self._portfolio = {
            "cash": config.trading.initial_capital,
            "total_value": config.trading.initial_capital,
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
        }

        # Register signal handlers
        self._register_signal_handlers()

        logger.info(f"🤖 FullTradingBot created (mode={mode}, strategy={strategy})")

    # ─── Setup ─────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Initialize all bot components."""
        logger.info("🚀 Setting up Full Unified Trading Bot...")
        try:
            await self._setup_memory_stack()
            self._setup_execution_stack()
            await self._setup_analysis_stack()
            await self._setup_runtime_services()
            await self._load_state()

            logger.info("✅ All components initialized")
            self._print_startup_summary()
        except Exception as exc:
            logger.error(f"FullTradingBot setup failed: {exc}")
            await self._cleanup_setup_state()
            raise

    async def _cleanup_setup_state(self) -> None:
        """Release partially initialized components after a setup failure."""
        await self._cleanup_runtime_components()

    async def _setup_memory_stack(self) -> None:
        """Initialize persistence-oriented components."""
        await self._setup_trade_memory()
        if self.enable_memory:
            await self._setup_mem0_memory()
        await self._setup_knowledge_graph()

    def _setup_execution_stack(self) -> None:
        """Initialize risk, execution, and strategy components."""
        self._setup_risk()
        self._setup_executor()
        self._setup_strategies()

    async def _setup_analysis_stack(self) -> None:
        """Initialize debate, news, and autotune services."""
        if self._requires_debate_engine():
            await self._setup_debate_engine()

        if self.enable_news:
            self._setup_news_pipeline()

        if self.enable_autotune:
            self._setup_auto_tuner()

    async def _setup_runtime_services(self) -> None:
        """Initialize runtime integrations needed after the core stack."""
        await self._setup_redis()
        self._setup_weekly_reviewer()
        self._setup_telegram_alerts()

    def _requires_debate_engine(self) -> bool:
        """Return whether the selected strategy uses the debate engine."""
        return self.strategy_name in {"ai_debate", "sma_cross", "bbands"}

    def _setup_weekly_reviewer(self) -> None:
        """Initialize the weekly reviewer when trade memory is available."""
        if self._trade_memory is None:
            return

        from src.memory import WeeklyReviewer

        self._weekly_reviewer = WeeklyReviewer(self._trade_memory)
        logger.info("✅ WeeklyReviewer initialized")

    def _setup_telegram_alerts(self) -> None:
        """Initialize Telegram alert delivery when configured."""
        if not self.config.monitoring.telegram_enabled:
            return

        token = self.config.monitoring.telegram_bot_token
        chat_id = self.config.monitoring.telegram_chat_id
        if not token or not chat_id:
            logger.warning(
                "⚠️ Telegram alerts enabled but TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are missing"
            )
            return

        from src.monitoring.telegram_bot import TelegramBot

        self._telegram_bot = TelegramBot(bot_token=token, chat_id=chat_id)
        logger.info("✅ Telegram alerts initialized")

    def _runtime_component_attrs(self) -> tuple[str, ...]:
        """Return owned runtime attributes in dependency order."""
        return (
            "_trade_memory",
            "_mem0_memory",
            "_knowledge_graph",
            "_risk_engine",
            "_kill_switch",
            "_executor",
            "_order_manager",
            "_strategies",
            "_debate_engine",
            "_news_pipeline",
            "_auto_tuner",
            "_redis_cache",
            "_weekly_reviewer",
            "_telegram_bot",
        )

    async def _cleanup_runtime_components(self) -> None:
        """Release all owned runtime components using the shared cleanup helper."""
        await self._cleanup_component_attrs(*self._runtime_component_attrs())

    async def _setup_trade_memory(self) -> None:
        """Initialize PostgreSQL trade memory."""
        from src.memory import TradeMemory

        db_url = getattr(self.config, "database_url", get_default_database_url())
        redis_url = getattr(self.config, "redis_url", get_default_redis_url())

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

            qdrant_url = getattr(self.config, "qdrant_url", get_default_qdrant_url())
            self._mem0_memory = Mem0Memory(
                qdrant_url=qdrant_url,
                embedding_model=getattr(
                    self.config,
                    "mem0_embedding_model",
                    get_default_mem0_embedding_model(),
                ),
            )

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
        self._kill_switch.arm()
        logger.info("✅ RiskEngine + KillSwitch initialized")

    def _setup_executor(self) -> None:
        """Initialize trade executor."""
        from src.execution.dry_run import DryRunExecutor
        from src.execution.order_manager import OrderManager
        from src.execution.exchange_client import ExchangeClient

        self._executor = DryRunExecutor(
            initial_balance=self.config.trading.initial_capital
        )
        
        if self.mode == "dryrun":
            self._order_manager = OrderManager(None, dry_run=True)
        else:
            api_key = self.config.binance_testnet_api_key if self.mode == "testnet" else self.config.binance_api_key
            api_secret = self.config.binance_testnet_api_secret if self.mode == "testnet" else self.config.binance_api_secret
            
            exchange_client = ExchangeClient(
                api_key=api_key,
                api_secret=api_secret,
                testnet=(self.mode == "testnet"),
                exchange_name=getattr(
                    self.config,
                    "exchange_name",
                    get_default_exchange_name(),
                ),
            )
            try:
                exchange_client.connect()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to connect exchange client for {self.mode}: {e}"
                ) from e
            self._order_manager = OrderManager(exchange_client, dry_run=False)

        logger.info(f"✅ {self.mode} executor initialized (${self.config.trading.initial_capital:,.2f})")

    async def _setup_debate_engine(self) -> None:
        """Initialize the debate engine."""
        try:
            self._debate_engine, llm_model = build_debate_engine(
                self.config,
                self.symbols,
            )
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
                rss_feeds=getattr(
                    self.config,
                    "news_rss_feeds",
                    get_default_news_rss_feeds(),
                ),
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

            redis_url = getattr(self.config, "redis_url", get_default_redis_url())
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
        logger.info(f"  Total P&L:      ${portfolio.get('total_pnl', 0):+,.2f}")
        logger.info(f"  Total Trades:   {len(trade_log)}")
        logger.info(f"  Loops Executed: {self._loop_count}")
        logger.info("=" * 60)

        # Close connections
        await self._cleanup_runtime_components()

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

                await self._check_drawdown_kill_switch()

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
