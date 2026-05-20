"""Position sizing calculator using Kelly and Van Tharp methods."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""

    size_quote: float  # Size in quote currency (e.g., USDT)
    size_base: float  # Size in base currency (e.g., BTC)
    method_used: str
    kelly_fraction: float
    van_tharp_size: float


class PositionSizer:
    """Calculates optimal position sizes using multiple methods."""

    def __init__(
        self,
        max_position_pct: float = 0.20,
        max_leverage: int = 3,
        daily_loss_limit_pct: float = 0.03,
    ) -> None:
        """Initialize position sizer.

        Args:
            max_position_pct: Maximum position size as fraction of equity.
            max_leverage: Maximum allowed leverage.
            daily_loss_limit_pct: Maximum daily loss as fraction of equity.
        """
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
        self.daily_loss_limit_pct = daily_loss_limit_pct

    @staticmethod
    def calc_half_kelly(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        equity: float,
    ) -> float:
        """Calculate position size using the Half-Kelly formula.

        Kelly formula: f = (b*p - q) / b
        Half-Kelly: f/2 (more conservative)

        Args:
            win_rate: Probability of winning (0.0 to 1.0).
            avg_win: Average win amount (absolute value).
            avg_loss: Average loss amount (absolute value, positive).
            equity: Total account equity.

        Returns:
            Position size in quote currency. 0 if Kelly is negative.
        """
        if avg_loss <= 0:
            logger.warning("avg_loss must be positive, returning 0")
            return 0.0

        b = avg_win / avg_loss  # Win/loss ratio (odds)
        p = win_rate  # Win probability
        q = 1.0 - p  # Loss probability

        kelly = (b * p - q) / b

        if kelly <= 0:
            logger.info(f"Kelly criterion negative ({kelly:.4f}), no position recommended")
            return 0.0

        half_kelly = kelly / 2.0
        position_size = half_kelly * equity

        logger.info(
            f"Half-Kelly: b={b:.2f}, p={p:.2f}, q={q:.2f}, "
            f"kelly={kelly:.4f}, half_kelly={half_kelly:.4f}, "
            f"size={position_size:.2f}"
        )

        return position_size

    @staticmethod
    def calc_van_tharp(
        entry_price: float,
        stop_loss_price: float,
        equity: float,
        risk_pct: float = 0.02,
    ) -> float:
        """Calculate position size using Van Tharp's method.

        size = (equity * risk_pct) / stop_distance

        Args:
            entry_price: Entry price per unit.
            stop_loss_price: Stop-loss price per unit.
            equity: Total account equity.
            risk_pct: Percentage of equity to risk (default 2%).

        Returns:
            Position size in base currency units.
        """
        stop_distance = abs(entry_price - stop_loss_price)

        if stop_distance <= 0:
            logger.warning("Stop distance is zero, returning 0")
            return 0.0

        risk_amount = equity * risk_pct
        size_base = risk_amount / stop_distance

        logger.info(
            f"Van Tharp: risk_amount={risk_amount:.2f}, "
            f"stop_distance={stop_distance:.4f}, size_base={size_base:.6f}"
        )

        return size_base

    def calc_position_size(
        self,
        strategy: str,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        equity: float,
        win_rate: float = 0.5,
        avg_win: float = 0.03,
        avg_loss: float = 0.02,
    ) -> PositionSizeResult:
        """Calculate optimal position size using min of all methods.

        Uses the most conservative (smallest) size from:
        - Half-Kelly criterion
        - Van Tharp risk model
        - Maximum position percentage limit

        Args:
            strategy: Strategy name for logging.
            symbol: Trading pair symbol.
            entry_price: Planned entry price.
            stop_loss_price: Stop-loss price.
            equity: Current account equity.
            win_rate: Historical win rate for Kelly.
            avg_win: Average win for Kelly.
            avg_loss: Average loss for Kelly.

        Returns:
            PositionSizeResult with sizes from all methods.
        """
        # Calculate Kelly size (in quote currency)
        kelly_size = self.calc_half_kelly(win_rate, avg_win, avg_loss, equity)

        # Calculate Van Tharp size (in base currency), convert to quote
        vt_size_base = self.calc_van_tharp(entry_price, stop_loss_price, equity)
        vt_size_quote = vt_size_base * entry_price

        # Maximum allowed position size
        max_size = equity * self.max_position_pct

        # Use the minimum of all three methods
        sizes = [s for s in [kelly_size, vt_size_quote, max_size] if s > 0]
        if not sizes:
            chosen_size = 0.0
            method = "none (all methods returned 0)"
        else:
            chosen_size = min(sizes)
            if chosen_size == kelly_size:
                method = "half_kelly"
            elif chosen_size == vt_size_quote:
                method = "van_tharp"
            else:
                method = "max_position_limit"

        # Convert chosen size to base currency
        chosen_size_base = chosen_size / entry_price if entry_price > 0 else 0.0

        result = PositionSizeResult(
            size_quote=round(chosen_size, 2),
            size_base=round(chosen_size_base, 8),
            method_used=method,
            kelly_fraction=round(kelly_size / equity, 4) if equity > 0 else 0.0,
            van_tharp_size=round(vt_size_quote, 2),
        )

        logger.info(
            f"[{strategy}] Position size for {symbol}: "
            f"quote={result.size_quote}, base={result.size_base}, "
            f"method={result.method_used}"
        )

        return result

    @staticmethod
    def check_daily_loss(
        current_equity: float,
        start_equity: float,
        limit_pct: float = 0.03,
    ) -> bool:
        """Check if daily loss limit has been exceeded.

        Args:
            current_equity: Current account equity.
            start_equity: Equity at start of day.
            limit_pct: Maximum allowed daily loss fraction.

        Returns:
            True if daily loss limit exceeded.
        """
        if start_equity <= 0:
            return True

        daily_loss = (start_equity - current_equity) / start_equity
        exceeded = daily_loss >= limit_pct

        if exceeded:
            logger.warning(
                f"Daily loss limit exceeded: {daily_loss:.2%} >= {limit_pct:.2%}"
            )
        return exceeded

    @staticmethod
    def check_concentration(
        position_value: float,
        total_equity: float,
        max_pct: float = 0.20,
    ) -> bool:
        """Check if position concentration is too high.

        Args:
            position_value: Value of the position.
            total_equity: Total account equity.
            max_pct: Maximum allowed concentration fraction.

        Returns:
            True if concentration exceeds limit.
        """
        if total_equity <= 0:
            return True

        concentration = position_value / total_equity
        exceeded = concentration >= max_pct

        if exceeded:
            logger.warning(
                f"Position concentration too high: {concentration:.2%} >= {max_pct:.2%}"
            )
        return exceeded

    @staticmethod
    def check_leverage(
        leverage: float,
        max_leverage: float = 3.0,
    ) -> bool:
        """Check if leverage exceeds maximum.

        Args:
            leverage: Current leverage ratio.
            max_leverage: Maximum allowed leverage.

        Returns:
            True if leverage exceeds limit.
        """
        exceeded = leverage >= max_leverage
        if exceeded:
            logger.warning(f"Leverage exceeded: {leverage:.2f} >= {max_leverage:.2f}")
        return exceeded
