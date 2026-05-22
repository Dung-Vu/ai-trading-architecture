"""Quality gates for validating incoming market data."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class TradeValidationCode(str, Enum):
    """Stable reason codes for trade-validation outcomes."""

    OK = "ok"
    INVALID_PRICE = "invalid_price"
    LATENCY = "latency_exceeded"
    SPREAD = "spread_exceeded"
    PRICE_SPIKE = "price_spike"


@dataclass(frozen=True, slots=True)
class TradeValidationResult:
    """Typed validation outcome that remains backward-compatible with tuple unpacking."""

    passed: bool
    reason: str
    code: TradeValidationCode

    def __iter__(self):
        yield self.passed
        yield self.reason

    def __bool__(self) -> bool:
        return self.passed


class QualityGates:
    """Validates market data against latency, spread, and anomaly checks.

    Parameters
    ----------
    max_latency_ms : int
        Maximum acceptable message latency in milliseconds.
    z_score_threshold : float
        Z-score threshold for price spike detection.
    max_spread_pct : float
        Maximum acceptable bid-ask spread as percentage of mid price.
    """

    def __init__(
        self,
        max_latency_ms: int = 5000,
        z_score_threshold: float = 3.0,
        max_spread_pct: float = 1.0,
    ) -> None:
        if max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive")
        if z_score_threshold <= 0:
            raise ValueError("z_score_threshold must be positive")
        if max_spread_pct <= 0:
            raise ValueError("max_spread_pct must be positive")

        self.max_latency_ms = max_latency_ms
        self.z_score_threshold = z_score_threshold
        self.max_spread_pct = max_spread_pct

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_latency(self, receipt_timestamp: float) -> bool:
        """Reject data if latency exceeds threshold.

        Parameters
        ----------
        receipt_timestamp : float
            Timestamp when the message was received (epoch seconds).

        Returns
        -------
        bool
            True if latency is acceptable, False if rejected.
        """
        import time

        now = time.time()
        latency_ms = (now - receipt_timestamp) * 1000

        if latency_ms > self.max_latency_ms:
            logger.warning(
                f"Latency check FAILED: {latency_ms:.0f}ms > {self.max_latency_ms}ms threshold"
            )
            return False
        return True

    def check_spread(self, bid: float, ask: float) -> bool:
        """Reject if bid-ask spread exceeds threshold.

        Parameters
        ----------
        bid : float
            Current bid price.
        ask : float
            Current ask price.

        Returns
        -------
        bool
            True if spread is acceptable, False if rejected.
        """
        if bid <= 0 or ask <= 0:
            logger.warning(f"Spread check FAILED: invalid prices bid={bid}, ask={ask}")
            return False

        mid = (bid + ask) / 2.0
        spread_pct = ((ask - bid) / mid) * 100.0

        if spread_pct > self.max_spread_pct:
            logger.warning(
                f"Spread check FAILED: {spread_pct:.3f}% > {self.max_spread_pct}% threshold "
                f"(bid={bid}, ask={ask}, mid={mid})"
            )
            return False
        return True

    def check_price_spike(
        self,
        price: float,
        recent_prices: list[float],
        threshold: float | None = None,
    ) -> bool:
        """Reject if price deviates significantly from recent history.

        Uses Z-score: |price - mean| / std > threshold.

        Parameters
        ----------
        price : float
            The price to validate.
        recent_prices : list[float]
            Recent price history for comparison (needs >= 2 values).
        threshold : float | None
            Override z_score_threshold for this check.

        Returns
        -------
        bool
            True if price is normal, False if spike detected.
        """
        if threshold is None:
            threshold = self.z_score_threshold

        if len(recent_prices) < 2:
            logger.debug(
                f"Spike check SKIPPED: need >= 2 recent prices, got {len(recent_prices)}"
            )
            return True  # Not enough data to judge

        mean = statistics.mean(recent_prices)
        std = statistics.stdev(recent_prices)

        if std == 0:
            # All recent prices identical; reject if different
            if price != mean:
                logger.warning(
                    f"Spike check FAILED: price={price} vs constant history={mean}"
                )
                return False
            return True

        z_score = abs(price - mean) / std

        if z_score > threshold:
            logger.warning(
                f"Spike check FAILED: z-score={z_score:.2f} > {threshold} "
                f"(price={price}, mean={mean:.4f}, std={std:.4f})"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Combined validation
    # ------------------------------------------------------------------

    def validate_trade(
        self,
        trade_data: dict,
        receipt_ts: float,
        recent_prices: list[float],
    ) -> TradeValidationResult:
        """Run all quality gates on a trade.

        Checks performed (in order):
        1. Latency — message freshness
        2. Spread — bid/ask sanity (if available in trade_data)
        3. Price spike — Z-score against recent history

        Parameters
        ----------
        trade_data : dict
            Trade data with keys: 'symbol', 'side', 'price', 'amount',
            optionally 'bid', 'ask'.
        receipt_ts : float
            Epoch timestamp when message was received.
        recent_prices : list[float]
            Recent prices for spike detection.

        Returns
        -------
        TradeValidationResult
            Typed outcome with a stable reason code.
        """
        symbol = trade_data.get("symbol", "UNKNOWN")
        price = trade_data.get("price")

        if price is None or price <= 0:
            return TradeValidationResult(
                passed=False,
                reason=f"Invalid price in trade data: {price}",
                code=TradeValidationCode.INVALID_PRICE,
            )

        # 1. Latency check
        if not self.check_latency(receipt_ts):
            return TradeValidationResult(
                passed=False,
                reason=f"Latency exceeded threshold ({self.max_latency_ms}ms)",
                code=TradeValidationCode.LATENCY,
            )

        # 2. Spread check (if bid/ask available)
        bid = trade_data.get("bid")
        ask = trade_data.get("ask")
        if bid is not None and ask is not None:
            if not self.check_spread(bid, ask):
                return TradeValidationResult(
                    passed=False,
                    reason=f"Spread exceeded threshold ({self.max_spread_pct}%)",
                    code=TradeValidationCode.SPREAD,
                )

        # 3. Price spike check
        if not self.check_price_spike(price, recent_prices):
            return TradeValidationResult(
                passed=False,
                reason=(
                    f"Price spike detected for {symbol}: "
                    f"z-score > {self.z_score_threshold}"
                ),
                code=TradeValidationCode.PRICE_SPIKE,
            )

        logger.debug(
            f"Trade validation PASSED for {symbol}: "
            f"price={price}, side={trade_data.get('side')}"
        )
        return TradeValidationResult(
            passed=True,
            reason="OK",
            code=TradeValidationCode.OK,
        )
