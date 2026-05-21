"""Dry-run executor for simulating trades without real exchange connections."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.execution.trade_utils import calculate_realized_pnl, is_exit_order_triggered


@dataclass
class Position:
    """Represents an open position."""

    symbol: str
    quantity: float
    avg_price: float
    timestamp: str
    trades: list[dict] = field(default_factory=list)

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price


@dataclass
class TradeLog:
    """Record of a single trade."""

    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: str
    pnl: float = 0.0
    pnl_pct: float = 0.0


class DryRunExecutor:
    """Simulates trading operations without real exchange connections.

    Tracks cash, positions, and P&L in memory for backtesting and strategy
    validation.
    """

    def __init__(self, initial_balance: float = 10_000.0) -> None:
        """Initialize dry-run executor.

        Args:
            initial_balance: Starting cash balance in quote currency.
        """
        self._initial_balance = initial_balance
        self._cash = initial_balance
        self._positions: dict[str, Position] = {}
        self._trade_log: list[TradeLog] = []
        self._equity_curve: list[tuple[str, float]] = []
        self._order_book: dict[str, dict] = {}  # pending SL/TP orders
        self._trade_counter = 0

        # Record initial equity
        self._record_equity(
            datetime.now(timezone.utc).isoformat(), initial_balance
        )

    def _record_equity(self, timestamp: str, total_value: float) -> None:
        """Record an equity curve data point."""
        self._equity_curve.append((timestamp, total_value))

    def _get_total_value(self, current_prices: dict[str, float] | None = None) -> float:
        """Calculate total portfolio value.

        Args:
            current_prices: Optional dict of symbol -> current price.
                           If None, uses position avg_price as proxy.

        Returns:
            Total portfolio value (cash + positions).
        """
        total = self._cash
        for sym, pos in self._positions.items():
            price = current_prices.get(sym, pos.avg_price) if current_prices else pos.avg_price
            total += pos.quantity * price
        return total

    def simulate_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: str | None = None,
    ) -> dict:
        """Simulate a buy order.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT').
            quantity: Amount to buy in base currency.
            price: Execution price in quote currency.
            timestamp: Optional ISO timestamp.

        Returns:
            Trade result dictionary.

        Raises:
            ValueError: If insufficient cash.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        cost = quantity * price
        if cost > self._cash:
            raise ValueError(
                f"Insufficient cash: need {cost:.2f}, have {self._cash:.2f}"
            )

        self._cash -= cost
        self._trade_counter += 1

        if symbol in self._positions:
            # Average down/up
            pos = self._positions[symbol]
            total_qty = pos.quantity + quantity
            new_avg = (pos.cost_basis + cost) / total_qty
            pos.quantity = total_qty
            pos.avg_price = new_avg
            pos.timestamp = timestamp
            pos.trades.append({
                "trade_id": self._trade_counter,
                "side": "buy",
                "quantity": quantity,
                "price": price,
                "timestamp": timestamp,
            })
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                timestamp=timestamp,
                trades=[{
                    "trade_id": self._trade_counter,
                    "side": "buy",
                    "quantity": quantity,
                    "price": price,
                    "timestamp": timestamp,
                }],
            )

        trade = TradeLog(
            symbol=symbol,
            side="buy",
            quantity=quantity,
            price=price,
            timestamp=timestamp,
        )
        self._trade_log.append(trade)

        total_value = self._get_total_value()
        self._record_equity(timestamp, total_value)

        logger.info(
            f"[DRY-RUN] BUY {quantity} {symbol} @ {price} "
            f"(cost={cost:.2f}, cash={self._cash:.2f})"
        )

        return {
            "trade_id": self._trade_counter,
            "symbol": symbol,
            "side": "buy",
            "quantity": quantity,
            "price": price,
            "cost": cost,
            "cash_remaining": self._cash,
            "timestamp": timestamp,
        }

    def simulate_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: str | None = None,
    ) -> dict:
        """Simulate a sell order.

        Args:
            symbol: Trading pair symbol.
            quantity: Amount to sell in base currency.
            price: Execution price in quote currency.
            timestamp: Optional ISO timestamp.

        Returns:
            Trade result dictionary.

        Raises:
            ValueError: If no position or insufficient quantity.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        if symbol not in self._positions:
            raise ValueError(f"No position found for {symbol}")

        pos = self._positions[symbol]
        if quantity > pos.quantity:
            raise ValueError(
                f"Insufficient quantity: trying to sell {quantity}, "
                f"have {pos.quantity}"
            )

        revenue = quantity * price
        cost_basis = quantity * pos.avg_price
        pnl, pnl_pct = calculate_realized_pnl(
            pos.avg_price,
            price,
            quantity,
            side_to_execute="sell",
        )

        self._cash += revenue
        pos.quantity -= quantity
        self._trade_counter += 1

        pos.trades.append({
            "trade_id": self._trade_counter,
            "side": "sell",
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })

        # Remove position if fully sold
        if pos.quantity <= 1e-10:
            del self._positions[symbol]

        trade = TradeLog(
            symbol=symbol,
            side="sell",
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )
        self._trade_log.append(trade)

        total_value = self._get_total_value()
        self._record_equity(timestamp, total_value)

        logger.info(
            f"[DRY-RUN] SELL {quantity} {symbol} @ {price} "
            f"(revenue={revenue:.2f}, pnl={pnl:.2f}, pnl%={pnl_pct:.2f}%)"
        )

        return {
            "trade_id": self._trade_counter,
            "symbol": symbol,
            "side": "sell",
            "quantity": quantity,
            "price": price,
            "revenue": revenue,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "cash_total": self._cash,
            "timestamp": timestamp,
        }

    def simulate_sl_tp(
        self,
        pending_orders: list[dict],
        current_prices: dict[str, float],
        timestamp: str | None = None,
    ) -> list[dict]:
        """Check and trigger stop-loss / take-profit orders.

        Args:
            pending_orders: List of pending SL/TP order dicts with keys:
                - id: order ID
                - symbol: trading pair
                - side: 'buy' or 'sell'
                - type: 'stop_loss' or 'take_profit'
                - stop_price: trigger price
                - quantity: order quantity
                - direction: 'above' or 'below' (price crosses this to trigger)
            current_prices: Dict of symbol -> current price.
            timestamp: Optional ISO timestamp.

        Returns:
            List of triggered trade results.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        triggered = []
        remaining = []

        for order in pending_orders:
            sym = order["symbol"]
            order_type = order.get("type", "stop_loss")
            current_price = current_prices.get(sym)

            if current_price is None:
                remaining.append(order)
                continue

            if is_exit_order_triggered(
                order.get("direction", ""),
                current_price,
                order["stop_price"],
            ):
                try:
                    if order.get("side") == "sell":
                        result = self.simulate_sell(
                            sym, order["quantity"], current_price, timestamp
                        )
                    else:
                        result = self.simulate_buy(
                            sym, order["quantity"], current_price, timestamp
                        )
                    result["triggered_by"] = order_type
                    result["trigger_order_id"] = order["id"]
                    triggered.append(result)
                    logger.info(
                        f"[DRY-RUN] {order_type} triggered for {sym} @ {current_price}"
                    )
                except ValueError as e:
                    logger.warning(f"[DRY-RUN] Failed to trigger order: {e}")
                    remaining.append(order)
            else:
                remaining.append(order)

        return triggered

    def get_portfolio(self) -> dict:
        """Get current portfolio state.

        Returns:
            Dictionary with cash, positions, and total_value.
        """
        total_value = self._get_total_value()
        positions_data = {}
        for sym, pos in self._positions.items():
            positions_data[sym] = {
                "quantity": pos.quantity,
                "avg_price": pos.avg_price,
                "cost_basis": pos.cost_basis,
                "market_value": pos.market_value,
                "unrealized_pnl": 0.0,  # Requires external price feed
                "timestamp": pos.timestamp,
            }

        return {
            "cash": self._cash,
            "positions": positions_data,
            "total_value": total_value,
            "initial_balance": self._initial_balance,
            "total_pnl": total_value - self._initial_balance,
            "total_pnl_pct": (
                (total_value - self._initial_balance) / self._initial_balance * 100
            ) if self._initial_balance > 0 else 0.0,
        }

    def get_trade_log(self) -> list[dict]:
        """Get all simulated trades.

        Returns:
            List of trade log entries as dictionaries.
        """
        return [
            {
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "price": t.price,
                "timestamp": t.timestamp,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
            }
            for t in self._trade_log
        ]

    def get_equity_curve(self) -> list[tuple[str, float]]:
        """Get equity curve data points.

        Returns:
            List of (timestamp, total_value) tuples.
        """
        return list(self._equity_curve)

    def reset(self) -> None:
        """Reset the executor to initial state."""
        self._cash = self._initial_balance
        self._positions.clear()
        self._trade_log.clear()
        self._equity_curve.clear()
        self._order_book.clear()
        self._trade_counter = 0
        self._record_equity(
            datetime.now(timezone.utc).isoformat(), self._initial_balance
        )
        logger.info("[DRY-RUN] Executor reset to initial state")
