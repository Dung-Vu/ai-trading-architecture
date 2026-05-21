"""Shared trade-domain helpers for exit triggers and realized PnL."""

from __future__ import annotations


def is_exit_order_triggered(
    direction: str,
    current_price: float,
    stop_price: float,
) -> bool:
    """Return whether a stop-loss / take-profit order should trigger."""
    if direction == "below":
        return current_price <= stop_price
    if direction == "above":
        return current_price >= stop_price
    return False


def calculate_realized_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    *,
    side_to_execute: str = "sell",
) -> tuple[float, float]:
    """Return realized PnL and PnL percent for a closing trade."""
    if entry_price <= 0 or quantity <= 0:
        return 0.0, 0.0

    price_delta = exit_price - entry_price
    if side_to_execute.lower() != "sell":
        price_delta *= -1

    pnl = price_delta * quantity
    pnl_pct = price_delta / entry_price * 100
    return pnl, pnl_pct