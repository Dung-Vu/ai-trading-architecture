"""Base strategy class for crypto trading."""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import pandas as pd
from loguru import logger
from lumibot.entities import Asset, Order
from lumibot.strategies import Strategy

from .indicator_utils import calculate_indicator
from .mock_runtime import ensure_strategy_runtime_kwargs


class BaseStrategy(Strategy):
    """Abstract base class for all trading strategies.

    Inherits from Lumibot's Strategy and provides common utilities:
    - Indicator calculation helper
    - Risk/drawdown monitoring
    - Trade logging
    - Crypto market configuration (24/7)

    Subclasses must implement ``on_trading_iteration()``.
    """

    parameters = {
        "symbol": "BTC",
        "quote_asset": "USDT",
        "initial_capital": 100_000,
        "risk_pct": 0.02,
        "max_drawdown": 0.15,
    }

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the strategy and bypass broker checks if running out of Lumibot loop."""
        import os

        kwargs = ensure_strategy_runtime_kwargs(kwargs)

        prev_is_backtesting = os.environ.get("IS_BACKTESTING")
        os.environ["IS_BACKTESTING"] = "true"
        try:
            super().__init__(*args, **kwargs)
        finally:
            if prev_is_backtesting is not None:
                os.environ["IS_BACKTESTING"] = prev_is_backtesting
            else:
                os.environ.pop("IS_BACKTESTING", None)

    def generate_signal(
        self, price: float, indicators: dict, market_data: dict
    ) -> str:
        """Stub for manual signal generation outside the Lumibot lifecycle."""
        return "HOLD"

    def initialize(self) -> None:
        """Set up the strategy for crypto (24/7) markets and initialise state.

        This is called once before any trading iteration.  It configures
        the market type, sets the polling interval, and seeds ``self.vars``
        with persistent state variables.
        """
        # Crypto markets never close
        self.set_market("24/7")
        self.sleeptime = "1M"  # 1-minute candles between iterations

        # Persistent state via self.vars (required by Lumibot)
        self.vars.symbol = self.parameters["symbol"]
        self.vars.quote_asset = self.parameters["quote_asset"]
        self.vars.initial_capital = self.parameters["initial_capital"]
        self.vars.risk_pct = self.parameters["risk_pct"]
        self.vars.max_drawdown = self.parameters.get("max_drawdown", 0.15)

        # Trading state
        self.vars.entry_price: Optional[float] = None
        self.vars.trade_count = 0
        self.vars.peak_value = self.parameters["initial_capital"]

        # Build Asset objects
        self.vars.base_asset = Asset(
            self.vars.symbol, asset_type=Asset.AssetType.CRYPTO
        )
        self.vars.quote = Asset(
            self.vars.quote_asset, asset_type=Asset.AssetType.CRYPTO
        )

        logger.info(
            "BaseStrategy initialized: symbol={}/{}, capital={}",
            self.vars.symbol,
            self.vars.quote_asset,
            self.vars.initial_capital,
        )

    # ------------------------------------------------------------------ #
    #  Indicator helper
    # ------------------------------------------------------------------ #

    def _get_indicator(
        self, bars_df: pd.DataFrame, indicator_name: str, **params
    ) -> pd.Series | tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate a technical indicator from a price DataFrame."""
        return calculate_indicator(bars_df, indicator_name, **params)

    # ------------------------------------------------------------------ #
    #  Risk management
    # ------------------------------------------------------------------ #

    def _check_risk(self) -> bool:
        """Check whether the portfolio has breached the max drawdown limit.

        Updates the peak portfolio value and compares current value to the
        peak.  If the drawdown exceeds ``max_drawdown``, logs a warning
        and returns ``False`` to signal that no new trades should be
        opened.

        Returns
        -------
        bool
            ``True`` if risk limits are respected, ``False`` otherwise.
        """
        current_value = self.portfolio_value
        if current_value > self.vars.peak_value:
            self.vars.peak_value = current_value

        if self.vars.peak_value > 0:
            drawdown = (self.vars.peak_value - current_value) / self.vars.peak_value
        else:
            drawdown = 0.0

        if drawdown >= self.vars.max_drawdown:
            logger.warning(
                "Max drawdown breached: {:.2%} >= {:.2%} — halting new trades",
                drawdown,
                self.vars.max_drawdown,
            )
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Trade logging
    # ------------------------------------------------------------------ #

    def log_trade(
        self, side: str, symbol: str, quantity: float, price: float
    ) -> None:
        """Log a trade to the logger and update internal counters.

        Parameters
        ----------
        side :
            ``"buy"`` or ``"sell"``.
        symbol :
            Trading symbol (e.g. ``"BTC"``).
        quantity :
            Number of units traded.
        price :
            Execution price.
        """
        self.vars.trade_count += 1
        notional = quantity * price
        logger.info(
            "TRADE #{} | {} {} {} @ ${:,.2f} (notional ${:,.2f})",
            self.vars.trade_count,
            side.upper(),
            quantity,
            symbol,
            price,
            notional,
        )

    # ------------------------------------------------------------------ #
    #  Abstract hook
    # ------------------------------------------------------------------ #

    @abstractmethod
    def on_trading_iteration(self) -> None:
        """Main trading logic — must be overridden by subclasses."""
        raise NotImplementedError
