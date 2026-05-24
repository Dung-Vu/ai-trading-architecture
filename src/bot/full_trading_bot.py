"""Full unified trading bot orchestration.

Read this module to understand the end-to-end trading runtime:
1. `run()` owns the main loop and maintenance cadence.
2. `_process_symbol()` owns the per-symbol happy path.
3. Setup/shutdown helpers wire the runtime dependencies.

The implementation keeps the full trading flow in one owner so callers and AI
tools do not have to hop between mixins to understand the runtime behavior.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.bot.services import BotServices, FULL_BOT_RUNTIME_ATTRS
from src.bot_base import BaseTradingBot
from src.config import (
    get_default_database_url,
    get_default_exchange_name,
    get_default_mem0_embedding_model,
    get_default_mem0_llm_model,
    get_default_mem0_llm_provider,
    get_default_news_rss_feeds,
    get_default_qdrant_url,
    get_default_redis_url,
    get_default_simulated_base_prices,
)
from src.debate.runtime import build_debate_engine, normalize_debate_result, run_debate_round
from src.runtime_status import RuntimeFailurePolicy, RuntimeStatus
from src.shared_utils import trim_mapping_size


class FullTradingBot(BaseTradingBot):
    """
    Full unified trading bot combining all phases in one flow owner.

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
        self.services = BotServices()

        # ─── Core Components ───────────────────────────────────────────
        self._strategies = {}  # Strategy instances by symbol

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
        return FULL_BOT_RUNTIME_ATTRS

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
                llm_provider=getattr(
                    self.config,
                    "mem0_llm_provider",
                    get_default_mem0_llm_provider(),
                ),
                llm_model=getattr(
                    self.config,
                    "mem0_llm_model",
                    get_default_mem0_llm_model(),
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

    # ─── Maintenance Tasks ────────────────────────────────────────────

    async def _check_drawdown_kill_switch(self) -> None:
        """Auto-trigger the kill switch if portfolio drawdown breaches the limit."""
        if not self._kill_switch or not self._risk_engine:
            return

        portfolio = await self._get_portfolio_state()
        self._risk_engine.snapshot_equity(
            current_equity=portfolio["total_value"],
            start_equity=self.config.trading.initial_capital,
        )

        status = self._risk_engine.get_status()
        if self._kill_switch.auto_check(
            max_drawdown_pct=self._risk_engine.max_drawdown_pct,
            current_drawdown=status.current_drawdown_pct,
        ):
            self._running = False
            self._shutdown_event.set()

    async def _check_weekly_review(self) -> None:
        """Run weekly review if enough time has passed."""
        if not self._trade_memory or not self._weekly_reviewer:
            return

        now = datetime.now(timezone.utc)
        last = self._last_weekly_review

        if last is None or (now - last).days >= 7:
            try:
                report = await self._weekly_reviewer.generate_report()
                self._weekly_reviewer.save_report(report)

                logger.info("📊 Weekly Review:")
                logger.info("  Weekly review report generated and saved")

                self._last_weekly_review = now
            except Exception as exc:
                logger.warning(f"Weekly review failed: {exc}")

    async def _check_autotune(self) -> None:
        """Run auto-tune weekly optimization cycle."""
        if not self._auto_tuner:
            return

        now = datetime.now(timezone.utc)
        if self._last_autotune is None or (now - self._last_autotune).days >= 7:
            try:
                needs_optimization = await self._auto_tuner.detect_strategy_decay()

                if needs_optimization:
                    logger.info("🔧 Strategy decay detected — running optimization...")
                    report = await self._auto_tuner.weekly_optimization_cycle()
                    logger.info(
                        f"  Optimization complete: {report.get('status', 'unknown')}"
                    )
                else:
                    recommendations = (
                        await self._auto_tuner.get_optimization_recommendations()
                    )
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
        if self._last_memory_cleanup is None or (now - self._last_memory_cleanup).days >= 30:
            try:
                removed = self._mem0_memory.clear_old_memories(days=90)
                logger.info(
                    f"🧹 Memory cleanup: removed {removed} old memories (>90 days)"
                )
                self._last_memory_cleanup = now
            except Exception as exc:
                logger.warning(f"Memory cleanup failed: {exc}")

    # ─── Per-Symbol Pipeline ──────────────────────────────────────────

    async def _process_symbol(self, symbol: str) -> None:
        """Process one symbol through the entire trading pipeline."""
        logger.debug(f"🔄 Processing {symbol} (loop #{self._loop_count})")

        market_context = await self._prepare_symbol_market_context(symbol)
        if market_context is None:
            return

        market_data, current_price, indicators = market_context
        signal_action = await self._get_actionable_signal(
            symbol,
            market_data,
            current_price,
        )
        if signal_action is None:
            return

        debate_result = await self._run_symbol_debate(symbol, market_data, indicators)
        if debate_result is None:
            return

        await self._execute_symbol_trade_pipeline(
            symbol,
            current_price,
            indicators,
            market_data,
            debate_result,
        )

    async def _prepare_symbol_market_context(
        self,
        symbol: str,
    ) -> tuple[dict[str, Any], float, dict[str, Any]] | None:
        """Fetch market inputs for one symbol and process pending exit orders."""
        market_data = await self._build_market_data(symbol)
        if market_data is None:
            logger.debug(f"  No market data for {symbol}, skipping")
            return None

        current_price = market_data.get("price", 0.0)
        indicators = market_data.get("indicators", {})
        await self._handle_pending_exit_orders(
            symbol,
            current_price,
            self._send_alert,
        )
        return market_data, current_price, indicators

    async def _get_actionable_signal(
        self,
        symbol: str,
        market_data: dict[str, Any],
        current_price: float,
    ) -> str | None:
        """Run the strategy and stop early when no trade should be considered."""
        signal_action = self._run_strategy(symbol, market_data)
        if signal_action == "HOLD":
            logger.debug(f"  [{symbol}] Strategy says HOLD")
            return None

        logger.info(
            f"  [{symbol}] Strategy signal: {signal_action} "
            f"(price=${current_price:,.2f})"
        )
        return signal_action

    def _build_symbol_memory_context(
        self,
        symbol: str,
        indicators: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str]:
        """Gather reusable memory context for one debate round."""
        kg_context = self._query_knowledge_graph(indicators)
        mem0_context = self._query_mem0_memory(symbol, indicators)
        return kg_context, mem0_context

    async def _run_symbol_debate(
        self,
        symbol: str,
        market_data: dict[str, Any],
        indicators: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fetch debate inputs and return only actionable debate decisions."""
        sentiment = await self._fetch_sentiment(symbol)
        kg_context, mem0_context = self._build_symbol_memory_context(symbol, indicators)
        debate_result, _status = await self._run_debate_with_status(
            symbol,
            market_data,
            sentiment,
            kg_context,
            mem0_context,
        )
        if debate_result is None:
            logger.warning(f"  [{symbol}] Debate engine unavailable")
            return None

        final_action = debate_result.get("action", "HOLD")
        if final_action == "HOLD":
            logger.info(f"  [{symbol}] Debate says HOLD")
            return None

        return debate_result

    async def _execute_symbol_trade_pipeline(
        self,
        symbol: str,
        current_price: float,
        indicators: dict[str, Any],
        market_data: dict[str, Any],
        debate_result: dict[str, Any],
    ) -> None:
        """Apply risk checks, execute the trade, and record side effects."""
        final_action = debate_result.get("action", "HOLD")
        approved, reason = await self._run_risk_check(
            symbol,
            final_action,
            current_price,
        )
        if not approved:
            logger.info(f"  [{symbol}] Risk rejected: {reason}")
            return

        trade_result = await self._execute_trade(
            symbol,
            final_action,
            current_price,
            debate_result,
        )
        if trade_result is None:
            return

        await self._log_trade(symbol, trade_result, debate_result, market_data)
        self._update_knowledge_graph(indicators, final_action, trade_result)
        await self._send_alert(symbol, trade_result, debate_result)

    async def _build_market_data(self, symbol: str) -> dict[str, Any] | None:
        """Build market data with price and derived indicators."""
        price = await self._get_latest_price(symbol)
        if price is None:
            if self.mode != "dryrun":
                logger.debug(f"  No live price available for {symbol}, skipping")
                return None
            price = self._simulate_price(symbol)
            if price is None:
                return None

        indicators = self._build_simulated_indicators(symbol, price)
        market_conditions = self._derive_market_conditions(indicators)

        return {
            "symbol": symbol,
            "price": price,
            "indicators": indicators,
            "market_conditions": market_conditions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _simulate_price(self, symbol: str) -> float | None:
        """Simulate a realistic price for dry-run mode."""
        cached = self._simulated_price_cache.get(symbol)
        if cached is not None:
            return cached

        base_prices = getattr(
            self.config,
            "simulated_base_prices",
            get_default_simulated_base_prices(),
        )
        base = base_prices.get(symbol)
        if base is None:
            return None

        seed_material = f"{self._simulation_seed}:{symbol}".encode("utf-8")
        seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
        rng = random.Random(seed)
        price = round(base * (1 + rng.uniform(-0.03, 0.03)), 2)
        self._simulated_price_cache[symbol] = price
        return price

    def _build_simulated_indicators(
        self,
        symbol: str,
        price: float,
    ) -> dict[str, Any]:
        """Generate deterministic session-scoped indicators for a symbol/price pair."""
        cache_key = (symbol, round(price, 8))
        cached = self._indicator_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        seed_material = (
            f"{self._indicator_seed}:{symbol}:{cache_key[1]}".encode("utf-8")
        )
        seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
        rng = random.Random(seed)

        indicators = {
            "price": price,
            "rsi": round(rng.uniform(20, 80), 1),
            "sma_fast": round(price * (1 + rng.uniform(-0.02, 0.02)), 2),
            "sma_slow": round(price * (1 + rng.uniform(-0.05, 0.05)), 2),
            "volume": round(rng.uniform(0.5, 3.0), 2),
            "volume_high": rng.choice([True, False]),
        }

        self._indicator_cache[cache_key] = dict(indicators)
        trim_mapping_size(self._indicator_cache, 256)
        return indicators

    @staticmethod
    def _derive_market_conditions(indicators: dict[str, Any]) -> dict[str, Any]:
        """Derive stable market-condition labels from the indicator snapshot."""
        sma_fast = float(indicators.get("sma_fast", indicators.get("price", 0.0)))
        sma_slow = float(indicators.get("sma_slow", indicators.get("price", 0.0)))
        price = float(indicators.get("price", 0.0))
        volume_high = bool(indicators.get("volume_high", False))

        if sma_fast > sma_slow:
            trend = "uptrend"
        elif sma_fast < sma_slow:
            trend = "downtrend"
        else:
            trend = "sideways"

        divergence = abs(sma_fast - sma_slow) / price if price > 0 else 0.0
        if divergence >= 0.03:
            volatility = "high"
        elif divergence >= 0.01:
            volatility = "medium"
        else:
            volatility = "low"

        return {
            "trend": trend,
            "volatility": volatility,
            "volume_high": volume_high,
        }

    async def _fetch_sentiment(self, symbol: str) -> dict[str, Any]:
        """Fetch news sentiment for a symbol."""
        if not self._news_pipeline:
            return {"score": 0.0, "positive": 0, "negative": 0}

        try:
            ticker = symbol.split("/")[0]
            loop = asyncio.get_running_loop()
            news_items = await loop.run_in_executor(
                None,
                self._news_pipeline.fetch_crypto_news,
                ticker,
                24,
            )

            sentiment = await loop.run_in_executor(
                None,
                self._news_pipeline.analyze_sentiment,
                news_items,
            )

            logger.info(
                f"  [{symbol}] Sentiment: score={sentiment['overall_score']:.2f}, "
                f"+{sentiment['positive_count']}/-{sentiment['negative_count']}"
            )
            return sentiment
        except Exception as exc:
            logger.warning(f"  [{symbol}] Sentiment fetch failed: {exc}")
            return {"score": 0.0, "positive": 0, "negative": 0}

    def _query_knowledge_graph(self, indicators: dict[str, Any]) -> list[dict[str, Any]]:
        """Query knowledge graph for matching patterns."""
        if self._knowledge_graph is None:
            return []

        rsi = indicators.get("rsi", 50)
        volume_high = indicators.get("volume_high", False)

        condition_parts: list[str] = []
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

    def _query_mem0_memory(self, symbol: str, indicators: dict[str, Any]) -> str:
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
                for item in similar:
                    outcome = item.get("outcome", "pending")
                    pnl = item.get("pnl", 0)
                    summary_parts.append(f"Past: {outcome} (${pnl:+.2f})")
                return "; ".join(summary_parts)
        except Exception as exc:
            logger.debug(f"Mem0 query failed: {exc}")

        return ""

    async def _run_debate_with_status(
        self,
        symbol: str,
        market_data: dict[str, Any],
        sentiment: dict[str, Any],
        kg_context: list[dict[str, Any]],
        mem0_context: str,
    ) -> tuple[dict[str, Any] | None, RuntimeStatus]:
        """Run debate engine with full context and surface a typed runtime status."""
        enriched = dict(market_data)
        enriched["news_sentiment"] = sentiment
        enriched["knowledge_graph_patterns"] = kg_context
        enriched["memory_context"] = mem0_context

        if self._debate_engine:
            try:
                positions = self._positions.get(symbol, {})
                result = await run_debate_round(
                    self._debate_engine,
                    market_data=enriched,
                    current_positions={symbol: positions},
                    portfolio=self._portfolio,
                    symbol=symbol,
                )

                return (
                    normalize_debate_result(result, include_round_count=True),
                    self._runtime_success(
                        "debate_executed",
                        f"Debate completed for {symbol}",
                    ),
                )
            except Exception as exc:
                return None, self._runtime_failure(
                    "debate_execution_failed",
                    f"Debate engine error for {symbol}: {exc}",
                    policy=RuntimeFailurePolicy.RETURN_STATUS,
                    log_level="warning",
                )

        return (
            {
                "action": market_data.get("signal", "HOLD"),
                "confidence": 60.0,
                "reasoning": (
                    f"Strategy {self.strategy_name} signal (no debate engine)"
                ),
                "stop_loss": 0,
                "take_profit": 0,
                "risk_decision": "APPROVE",
                "bull_argument": "",
                "bear_argument": "",
                "devil_argument": "",
                "rounds": 0,
            },
            self._runtime_success(
                "debate_engine_fallback",
                f"Debate engine unavailable for {symbol}; using strategy fallback",
            ),
        )

    async def _run_debate(
        self,
        symbol: str,
        market_data: dict[str, Any],
        sentiment: dict[str, Any],
        kg_context: list[dict[str, Any]],
        mem0_context: str,
    ) -> dict[str, Any] | None:
        result, _status = await self._run_debate_with_status(
            symbol,
            market_data,
            sentiment,
            kg_context,
            mem0_context,
        )
        return result

    async def _log_trade(
        self,
        symbol: str,
        trade_result: dict[str, Any],
        debate_result: dict[str, Any],
        market_data: dict[str, Any],
    ) -> None:
        """Log trade and debate artifacts to the configured memory layers."""
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

        if self._trade_memory:
            try:
                await self._trade_memory.log_trade(trade_record)
            except Exception as exc:
                self._runtime_failure(
                    "trade_log_postgres_failed",
                    f"Failed to log trade to PostgreSQL: {exc}",
                    policy=RuntimeFailurePolicy.FALLBACK,
                    log_level="warning",
                )

        if self._mem0_memory:
            try:
                self._mem0_memory.add_trade_memory(trade_record, debate_result)
            except Exception as exc:
                self._runtime_failure(
                    "trade_log_mem0_failed",
                    f"Failed to log trade to Mem0: {exc}",
                    policy=RuntimeFailurePolicy.FALLBACK,
                    log_level="warning",
                )

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
                self._runtime_failure(
                    "debate_log_postgres_failed",
                    f"Failed to log debate to PostgreSQL: {exc}",
                    policy=RuntimeFailurePolicy.FALLBACK,
                    log_level="warning",
                )

    def _update_knowledge_graph(
        self,
        indicators: dict[str, Any],
        action: str,
        trade_result: dict[str, Any],
    ) -> None:
        """Add pattern to the knowledge graph for later trade recall."""
        if self._knowledge_graph is None or action.upper() != "BUY":
            return

        rsi = indicators.get("rsi", 50)
        volume_high = indicators.get("volume_high", False)

        conditions: list[str] = []
        if rsi < 30:
            conditions.append("RSI < 30")
        elif rsi > 70:
            conditions.append("RSI > 70")

        if volume_high:
            conditions.append("volume > avg")

        if conditions:
            condition_str = " AND ".join(conditions)
            pattern_id = self._knowledge_graph.add_pattern(
                condition=condition_str,
                action=action,
                outcome="pending",
                confidence=0.5,
            )

            position = self._positions.get(trade_result.get("symbol", ""))
            if position is not None:
                position["knowledge_graph_pattern_id"] = pattern_id
                position["knowledge_graph_condition"] = condition_str

    def _on_position_closed(
        self,
        symbol: str,
        trade_result: dict[str, Any],
        closed_position: dict[str, Any] | None = None,
    ) -> None:
        """Update knowledge-graph outcomes when a tracked position closes."""
        del symbol

        if self._knowledge_graph is None or not closed_position:
            return

        pattern_id = closed_position.get("knowledge_graph_pattern_id")
        if not pattern_id:
            return

        pnl = float(trade_result.get("pnl", 0.0) or 0.0)
        if pnl > 0:
            outcome = "win"
        elif pnl < 0:
            outcome = "loss"
        else:
            outcome = "break_even"

        self._knowledge_graph.update_pattern_by_id(
            pattern_id,
            outcome=outcome,
            pnl=pnl,
        )

    async def _send_alert(
        self,
        symbol: str,
        trade_result: dict[str, Any],
        debate_result: dict[str, Any],
    ) -> None:
        """Send Telegram alert for a completed trade."""
        if self._telegram_bot is None:
            return

        try:
            await self._telegram_bot.send_trade_alert(
                side=trade_result.get("side", "BUY"),
                symbol=symbol,
                quantity=trade_result.get("quantity", 0),
                price=trade_result.get("price", 0),
                pnl=trade_result.get("pnl"),
                strategy=self.strategy_name,
                ai_confidence=debate_result.get("confidence"),
                mode=self.mode,
            )
            logger.info(f"  [{symbol}] Telegram alert sent")
        except Exception as exc:
            self._runtime_failure(
                "telegram_alert_failed",
                f"Failed to send alert: {exc}",
                policy=RuntimeFailurePolicy.FALLBACK,
                log_level="warning",
            )

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


TradingBot = FullTradingBot
