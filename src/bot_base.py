"""Shared runtime helpers for AITradingBot and FullTradingBot."""

from __future__ import annotations

import inspect
import signal
from typing import Any, TypedDict

from loguru import logger

from src.bot.trade_executor import DryRunExecutorLike, TradeExecutionMixin


class StrategyBundle(TypedDict):
    """Per-symbol strategy instances managed by the shared bot base."""

    sma_cross: Any
    bbands: Any


class BaseTradingBot(TradeExecutionMixin):
    """Shared bot behavior for portfolio, strategy, and execution flows.

    Error handling contract:
    - setup-time failures should raise from the caller's setup method.
    - runtime integration failures should log and fall back when trading can continue.
    - business-rule rejections should return None or False instead of raising.
    """

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


    def _get_strategy_name(self) -> str:
        """Return the active strategy attribute used by the current bot."""
        return getattr(self, "strategy_name", getattr(self, "strategy", "ai_debate"))

    async def _cleanup_component_attr(self, attr_name: str) -> None:
        """Close an owned component if it exposes close(), then clear the attribute."""
        component = getattr(self, attr_name, None)
        if component is None:
            return

        close_method = getattr(component, "close", None)
        if callable(close_method):
            try:
                close_result = close_method()
                if inspect.isawaitable(close_result):
                    await close_result
            except Exception as exc:
                logger.warning(f"Failed to close {attr_name}: {exc}")

        setattr(self, attr_name, None)

    async def _cleanup_component_attrs(self, *attr_names: str) -> None:
        """Close owned components in reverse order to support setup rollback."""
        for attr_name in reversed(attr_names):
            await self._cleanup_component_attr(attr_name)


    def _ensure_strategy_registry(self) -> dict[str, StrategyBundle]:
        """Ensure the bot has a per-symbol strategy registry."""
        if getattr(self, "_strategies", None) is None:
            self._strategies = {}
        return self._strategies

    def _create_strategy_bundle(self, symbol: str) -> StrategyBundle:
        """Create signal-generating strategy instances for a symbol."""
        from src.strategy.bbands import BBandsStrategy
        from src.strategy.sma_cross import SMACrossStrategy

        strategy_config = getattr(self.config, "strategy", None)
        sma_fast = getattr(strategy_config, "sma_fast", 20)
        sma_slow = getattr(strategy_config, "sma_slow", 50)
        rsi_period = getattr(strategy_config, "rsi_period", 14)
        rsi_overbought = getattr(strategy_config, "rsi_overbought", 70)
        rsi_oversold = getattr(strategy_config, "rsi_oversold", 30)

        return {
            "sma_cross": SMACrossStrategy(
                symbol=symbol,
                sma_fast=sma_fast,
                sma_slow=sma_slow,
                rsi_period=rsi_period,
            ),
            "bbands": BBandsStrategy(
                symbol=symbol,
                rsi_period=rsi_period,
                rsi_overbought=rsi_overbought,
                rsi_oversold=rsi_oversold,
            ),
        }

    def _setup_strategies(self) -> None:
        """Initialize strategy instances in the strategy layer, per symbol."""
        registry = self._ensure_strategy_registry()
        for symbol in self.symbols:
            registry.setdefault(symbol, self._create_strategy_bundle(symbol))

        logger.info(
            f"✅ Strategies initialized for {len(registry)} symbol(s): {list(registry)}"
        )

    def _get_strategy_instance(self, strategy_name: str, symbol: str) -> Any:
        """Return the strategy instance for a specific symbol."""
        registry = self._ensure_strategy_registry()
        if symbol not in registry:
            registry[symbol] = self._create_strategy_bundle(symbol)

        return registry[symbol].get(strategy_name)

    def _generate_strategy_signal(
        self,
        symbol: str,
        market_data: dict[str, Any],
        strategy_name: str | None = None,
    ) -> str:
        """Generate a signal using the strategy layer instead of bot-local logic."""
        active_strategy = strategy_name or self._get_strategy_name()
        if active_strategy == "ai_debate":
            return "BUY"

        strategy = self._get_strategy_instance(active_strategy, symbol)
        if strategy is None:
            logger.warning(f"Strategy {active_strategy} not found for {symbol}")
            return "HOLD"

        return strategy.generate_signal(
            price=market_data.get("price", 0),
            indicators=market_data.get("indicators", {}),
            market_data=market_data,
        )

    def _run_strategy(self, symbol: str, market_data: dict[str, Any]) -> str:
        """Run the configured strategy for a symbol."""
        return self._generate_strategy_signal(symbol, market_data)

    def _sma_cross_signal(self, market_data: dict[str, Any]) -> str:
        """Compatibility wrapper for SMA signal generation."""
        symbol = market_data.get("symbol") or self.symbols[0]
        return self._generate_strategy_signal(symbol, market_data, "sma_cross")

    def _bbands_signal(self, market_data: dict[str, Any]) -> str:
        """Compatibility wrapper for Bollinger Bands signal generation."""
        symbol = market_data.get("symbol") or self.symbols[0]
        return self._generate_strategy_signal(symbol, market_data, "bbands")

