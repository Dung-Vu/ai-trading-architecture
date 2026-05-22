"""Risk engine for pre-trade validation and risk monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from loguru import logger


@dataclass
class RiskStatus:
    """Current risk status snapshot."""

    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    active_positions: int = 0
    daily_loss_limit_exceeded: bool = False
    drawdown_limit_exceeded: bool = False
    kill_switch_active: bool = False


class RiskEngine:
    """Pre-trade risk validation and monitoring engine.

    Enforces limits on daily loss, drawdown, position concentration,
    and leverage before allowing trades to execute.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.10,
        max_position_pct: float = 0.20,
        max_leverage: int = 3,
    ) -> None:
        """Initialize risk engine.

        Args:
            max_daily_loss_pct: Maximum daily loss as fraction of equity.
            max_drawdown_pct: Maximum drawdown from peak equity.
            max_position_pct: Maximum single position as fraction of equity.
            max_leverage: Maximum allowed leverage.
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage

        # State tracking
        self._daily_pnl: float = 0.0
        self._start_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._current_equity: float | None = None
        self._last_reset_date: date = datetime.now(timezone.utc).date()

    def _reset_if_new_day(self) -> None:
        """Reset daily P&L if we've crossed a day boundary."""
        today = datetime.now(timezone.utc).date()
        if today > self._last_reset_date:
            logger.info(f"New trading day detected ({today}), resetting daily P&L")
            self._daily_pnl = 0.0
            self._last_reset_date = today

    def pre_trade_checks(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        current_equity: float,
        start_equity: float,
        positions: dict[str, dict],
    ) -> tuple[bool, str]:
        """Run all pre-trade risk checks.

        Args:
            symbol: Trading pair symbol.
            side: 'buy' or 'sell'.
            quantity: Order quantity.
            price: Order price.
            current_equity: Current account equity.
            start_equity: Equity at start of day.
            positions: Current open positions dict.

        Returns:
            Tuple of (approved, reason). approved=True means trade passes.
        """
        self._reset_if_new_day()
        self.snapshot_equity(current_equity=current_equity, start_equity=start_equity)

        trade_value = quantity * price

        # Check 1: Daily loss limit
        if start_equity > 0:
            daily_loss_pct = abs(self._daily_pnl) / start_equity
            if self._daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss_pct:
                reason = (
                    f"Daily loss limit reached: {daily_loss_pct:.2%} >= "
                    f"{self.max_daily_loss_pct:.2%} (PnL={self._daily_pnl:.2f})"
                )
                logger.warning(f"[RISK] Trade rejected for {symbol}: {reason}")
                return False, reason

        # Check 2: Drawdown limit
        if self._peak_equity > 0 and current_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown >= self.max_drawdown_pct:
                reason = (
                    f"Drawdown limit reached: {drawdown:.2%} >= "
                    f"{self.max_drawdown_pct:.2%} (peak={self._peak_equity:.2f}, "
                    f"current={current_equity:.2f})"
                )
                logger.warning(f"[RISK] Trade rejected for {symbol}: {reason}")
                return False, reason

        # Check 3: Position concentration
        if current_equity > 0:
            # Include existing position value for this symbol
            existing_value = 0.0
            if symbol in positions:
                existing_value = positions[symbol].get("market_value", 0.0)
            total_exposure = existing_value + trade_value
            concentration = total_exposure / current_equity
            if concentration >= self.max_position_pct:
                reason = (
                    f"Position concentration exceeded: {concentration:.2%} >= "
                    f"{self.max_position_pct:.2%} (symbol={symbol})"
                )
                logger.warning(f"[RISK] Trade rejected for {symbol}: {reason}")
                return False, reason

        # Check 4: Leverage (simplified: total exposure / equity)
        if current_equity > 0:
            total_exposure_all = sum(
                p.get("market_value", 0.0) for p in positions.values()
            ) + trade_value
            effective_leverage = total_exposure_all / current_equity
            if effective_leverage >= self.max_leverage:
                reason = (
                    f"Leverage limit exceeded: {effective_leverage:.2f} >= "
                    f"{self.max_leverage}"
                )
                logger.warning(f"[RISK] Trade rejected for {symbol}: {reason}")
                return False, reason

        reason = "All risk checks passed"
        logger.info(f"[RISK] Trade approved for {symbol}: {reason}")
        return True, reason

    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily P&L tracker.

        Args:
            pnl: P&L to add (can be positive or negative).
        """
        self._reset_if_new_day()
        self._daily_pnl += pnl
        logger.debug(f"Daily P&L updated: {self._daily_pnl:.2f}")

    def update_peak_equity(self, equity: float) -> None:
        """Update peak equity for drawdown calculation.

        Args:
            equity: Current equity value.
        """
        if self._current_equity is None:
            self._current_equity = equity

        if self._start_equity == 0.0 and equity > 0:
            self._start_equity = equity

        if equity > self._peak_equity:
            self._peak_equity = equity
            logger.debug(f"New peak equity recorded: {self._peak_equity:.2f}")

    def snapshot_equity(
        self,
        current_equity: float,
        start_equity: float | None = None,
    ) -> None:
        """Store the latest equity snapshot for status and drawdown checks."""
        if start_equity is not None and start_equity > 0:
            self._start_equity = start_equity

        if current_equity >= 0:
            self._current_equity = current_equity
            self.update_peak_equity(current_equity)

    def reset_daily(self) -> None:
        """Reset daily P&L tracker at day boundary."""
        logger.info(
            f"Daily reset: previous P&L was {self._daily_pnl:.2f}"
        )
        self._daily_pnl = 0.0
        self._last_reset_date = datetime.now(timezone.utc).date()

    def get_status(self) -> RiskStatus:
        """Get current risk status.

        Returns:
            RiskStatus dataclass with current metrics.
        """
        self._reset_if_new_day()

        daily_pnl_pct = (
            self._daily_pnl / self._start_equity
            if self._start_equity > 0
            else 0.0
        )

        drawdown = (
            max(0.0, (self._peak_equity - self._current_equity) / self._peak_equity)
            if self._peak_equity > 0 and self._current_equity is not None
            else 0.0
        )

        daily_exceeded = (
            self._start_equity > 0
            and self._daily_pnl < 0
            and abs(self._daily_pnl) / self._start_equity >= self.max_daily_loss_pct
        )

        drawdown_exceeded = (
            self._peak_equity > 0
            and self._current_equity is not None
            and drawdown >= self.max_drawdown_pct
        )

        return RiskStatus(
            daily_pnl=self._daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            peak_equity=self._peak_equity,
            current_drawdown_pct=drawdown,
            daily_loss_limit_exceeded=daily_exceeded,
            drawdown_limit_exceeded=drawdown_exceeded,
        )
