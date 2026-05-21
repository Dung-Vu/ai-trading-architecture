"""AI trading bot compatibility layer built on the full trading bot."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.bot.full_trading_bot import FullTradingBot


class AITradingBot(FullTradingBot):
    """Lean AI bot facade that reuses FullTradingBot execution plumbing."""

    def __init__(
        self,
        config: Any,
        mode: str = "dryrun",
        strategy: str = "ai_debate",
        symbols: list[str] | None = None,
        interval: int = 60,
    ) -> None:
        super().__init__(
            config=config,
            mode=mode,
            strategy=strategy,
            symbols=symbols,
            interval=interval,
            enable_memory=False,
            enable_news=False,
            enable_autotune=False,
        )
        self.strategy = strategy
        self._dry_run_executor: Any = None
        self._dspy_optimizer: Any = None

    async def setup(self) -> None:
        """Initialize the lean AI stack without news, Mem0, KG, or autotune."""
        logger.info("🚀 Setting up AI Trading Bot...")
        try:
            await self._setup_trade_memory()
            self._setup_risk()
            self._setup_executor()
            if self.mode == "dryrun":
                self._dry_run_executor = self._executor

            await self._setup_redis()
            self._setup_strategies()

            if self.strategy in ("ai_debate", "sma_cross", "bbands"):
                await self._setup_debate_engine()

            if self._trade_memory:
                from src.memory import WeeklyReviewer

                self._weekly_reviewer = WeeklyReviewer(self._trade_memory)
                logger.info("✅ WeeklyReviewer initialized")

            if self.config.monitoring.telegram_enabled:
                token = self.config.monitoring.telegram_bot_token
                chat_id = self.config.monitoring.telegram_chat_id
                if token and chat_id:
                    from src.monitoring.telegram_bot import TelegramBot

                    self._telegram_bot = TelegramBot(
                        bot_token=token,
                        chat_id=chat_id,
                    )
                    logger.info("✅ Telegram alerts initialized")
                else:
                    logger.warning(
                        "⚠️ Telegram alerts enabled but TELEGRAM_BOT_TOKEN/"
                        "TELEGRAM_CHAT_ID are missing"
                    )

            logger.info("✅ All components initialized")
        except Exception as exc:
            logger.error(f"AITradingBot setup failed: {exc}")
            await self._cleanup_setup_state()
            raise

    async def _cleanup_setup_state(self) -> None:
        """Release partially initialized components after a setup failure."""
        await self._cleanup_component_attrs(
            "_telegram_bot",
            "_weekly_reviewer",
            "_debate_engine",
            "_strategies",
            "_redis_cache",
            "_order_manager",
            "_dry_run_executor",
            "_executor",
            "_kill_switch",
            "_risk_engine",
            "_trade_memory",
        )

    async def shutdown(self) -> None:
        """Graceful shutdown for the lean AI bot."""
        logger.info("🛑 Shutting down AI Trading Bot...")
        self._running = False

        if self._dry_run_executor:
            portfolio = self._dry_run_executor.get_portfolio()
            logger.info(
                f"📊 Final portfolio: ${portfolio['total_value']:,.2f} "
                f"(P&L: ${portfolio['total_pnl']:+,.2f})"
            )

        await self._cleanup_component_attrs(
            "_telegram_bot",
            "_debate_engine",
            "_redis_cache",
            "_trade_memory",
        )
        logger.info("👋 AI Trading Bot shut down complete")

    async def _build_market_data(
        self,
        symbol: str,
        current_price: float | None = None,
    ) -> dict[str, Any] | None:
        """Build AI-bot market data, preserving the old optional price API."""
        if current_price is None:
            return await super()._build_market_data(symbol)

        market_data: dict[str, Any] = {
            "symbol": symbol,
            "price": current_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            import pandas as pd  # noqa: F401
            import ta  # noqa: F401

            market_data["indicators"] = {
                "rsi": 50.0,
                "macd": 0.0,
                "bb_upper": current_price * 1.02,
                "bb_lower": current_price * 0.98,
                "volume": 0.0,
                "volume_high": False,
            }
        except ImportError:
            market_data["indicators"] = {}

        return market_data

    async def _run_debate(
        self,
        symbol: str,
        market_data: dict[str, Any],
        sentiment: dict[str, Any] | None = None,
        kg_context: list[dict[str, Any]] | None = None,
        mem0_context: str = "",
    ) -> dict[str, Any] | None:
        """Run debate and return the historical AITradingBot dict shape."""
        del sentiment, kg_context, mem0_context

        if self._debate_engine is None:
            logger.warning(f"[{symbol}] Debate engine not available")
            return None

        try:
            portfolio = await self._get_portfolio_state()
            current_positions = dict(portfolio.get("positions") or {})
            if symbol in self._positions:
                current_positions[symbol] = {
                    **current_positions.get(symbol, {}),
                    **self._positions[symbol],
                }

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._debate_engine.run_debate(
                    market_data=market_data,
                    current_positions=current_positions,
                    portfolio=portfolio,
                    symbol=symbol,
                ),
            )

            if hasattr(result, "model_dump"):
                return result.model_dump()
            if hasattr(result, "dict"):
                return result.dict()
            return {
                "action": getattr(result, "action", "HOLD"),
                "confidence": getattr(result, "confidence", 50.0),
                "reason": getattr(result, "reason", ""),
                "reasoning": getattr(result, "reason", ""),
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

    async def _log_trade_and_debate(
        self,
        symbol: str,
        trade_result: dict[str, Any],
        debate_result: dict[str, Any],
    ) -> None:
        """Log trade and debate result to memory using the AI bot schema."""
        if self._trade_memory is None:
            return

        try:
            ts = datetime.now(timezone.utc).isoformat()
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

            debate_data = {
                "timestamp": ts,
                "symbol": symbol,
                "bull_arg": debate_result.get("bull_argument", ""),
                "bear_arg": debate_result.get("bear_argument", ""),
                "devil_arg": debate_result.get("devil_argument", ""),
                "judge_action": debate_result.get("action", "HOLD"),
                "judge_confidence": debate_result.get("confidence", 50),
                "risk_action": debate_result.get("risk_decision", "APPROVE"),
                "risk_reasoning": debate_result.get("risk_reasoning", ""),
            }
            await self._trade_memory.log_debate(debate_data)
        except Exception as exc:
            logger.error(f"Failed to log trade/debate: {exc}")

    async def _send_alert(
        self,
        symbol: str,
        trade_result: dict[str, Any],
        debate_result: dict[str, Any],
    ) -> None:
        """Send Telegram alert for executed trade."""
        if self._telegram_bot is None:
            return

        try:
            await self._telegram_bot.send_trade_alert(
                side=trade_result.get("side", "BUY"),
                symbol=symbol,
                quantity=trade_result.get("quantity", 0),
                price=trade_result.get("price", 0),
                pnl=trade_result.get("pnl"),
                strategy=self.strategy,
                ai_confidence=debate_result.get("confidence"),
                mode=self.mode,
            )
        except Exception as exc:
            logger.debug(f"Failed to send Telegram alert: {exc}")

    async def _check_weekly_review(self) -> None:
        """Run weekly review and preserve the AI bot insight logging."""
        if self._weekly_reviewer is None:
            return

        now = datetime.now(timezone.utc)
        if (
            self._last_weekly_review is None
            or (now - self._last_weekly_review).days >= 7
        ):
            try:
                logger.info("📊 Running weekly review...")
                report = await self._weekly_reviewer.generate_report()
                self._weekly_reviewer.save_report(report)
                self._last_weekly_review = now

                insights = await self._weekly_reviewer.extract_insights()
                logger.info(f"📝 Weekly insights: {len(insights)} found")
                for insight in insights[:3]:
                    logger.info(f"  - {insight[:100]}...")
            except Exception as exc:
                logger.error(f"Weekly review failed: {exc}")
