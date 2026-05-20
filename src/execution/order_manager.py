"""Order manager for creating and managing exchange orders."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from .exchange_client import ExchangeClient


class OrderManager:
    """Manages order creation, cancellation, and status tracking."""

    def __init__(self, exchange_client: ExchangeClient, dry_run: bool = False) -> None:
        """Initialize order manager.

        Args:
            exchange_client: Connected exchange client instance.
            dry_run: If True, simulate orders without calling the exchange.
        """
        self._client = exchange_client
        self._dry_run = dry_run
        self._dry_run_orders: dict[str, dict] = {}
        self._dry_run_order_counter = 0

    def _precision_amount(self, symbol: str, amount: float) -> float:
        """Format amount to exchange precision."""
        if self._dry_run:
            return amount
        return self._client.exchange.amount_to_precision(symbol, amount)

    def _precision_price(self, symbol: str, price: float) -> float:
        """Format price to exchange precision."""
        if self._dry_run:
            return price
        return self._client.exchange.price_to_precision(symbol, price)

    def _make_dry_run_id(self) -> str:
        """Generate a dry-run order ID."""
        self._dry_run_order_counter += 1
        return f"DRY-{self._dry_run_order_counter:06d}-{uuid.uuid4().hex[:8]}"

    def _dry_run_order(
        self,
        order_id: str,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> dict:
        """Create a simulated order for dry-run mode."""
        order = {
            "id": order_id,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "price": price,
            "stopPrice": stop_price,
            "status": "open",
            "filled": 0.0,
            "remaining": amount,
            "cost": 0.0,
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "datetime": datetime.now(timezone.utc).isoformat(),
            "dry_run": True,
        }
        self._dry_run_orders[order_id] = order
        logger.info(f"[DRY-RUN] Created {order_type} {side} order: {amount} {symbol} @ {price}")
        return order

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
    ) -> dict:
        """Create a market order.

        Args:
            symbol: Trading pair symbol.
            side: 'buy' or 'sell'.
            amount: Order quantity in base currency.

        Returns:
            Order result dictionary.
        """
        try:
            if self._dry_run:
                order_id = self._make_dry_run_id()
                return self._dry_run_order(
                    order_id, symbol, "market", side, amount, price=0.0
                )

            precise_amount = self._client.exchange.amount_to_precision(symbol, amount)
            logger.info(f"Creating market {side} order: {precise_amount} {symbol}")
            return self._client.exchange.create_market_order(
                symbol, side, precise_amount
            )
        except Exception as e:
            logger.error(f"Failed to create market order: {e}")
            raise

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> dict:
        """Create a limit order.

        Args:
            symbol: Trading pair symbol.
            side: 'buy' or 'sell'.
            amount: Order quantity in base currency.
            price: Limit price.

        Returns:
            Order result dictionary.
        """
        try:
            if self._dry_run:
                order_id = self._make_dry_run_id()
                return self._dry_run_order(
                    order_id, symbol, "limit", side, amount, price=price
                )

            precise_amount = self._client.exchange.amount_to_precision(symbol, amount)
            precise_price = self._client.exchange.price_to_precision(symbol, price)
            logger.info(f"Creating limit {side} order: {precise_amount} {symbol} @ {precise_price}")
            return self._client.exchange.create_order(
                symbol, "limit", side, precise_amount, precise_price
            )
        except Exception as e:
            logger.error(f"Failed to create limit order: {e}")
            raise

    def create_stop_loss(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        limit_price: float,
    ) -> dict:
        """Create a stop-loss order (stop_loss_limit).

        Args:
            symbol: Trading pair symbol.
            side: 'buy' or 'sell'.
            amount: Order quantity.
            stop_price: Trigger price.
            limit_price: Limit price after trigger.

        Returns:
            Order result dictionary.
        """
        try:
            if self._dry_run:
                order_id = self._make_dry_run_id()
                return self._dry_run_order(
                    order_id, symbol, "stop_loss_limit", side, amount,
                    price=limit_price, stop_price=stop_price
                )

            precise_amount = self._client.exchange.amount_to_precision(symbol, amount)
            precise_stop = self._client.exchange.price_to_precision(symbol, stop_price)
            precise_limit = self._client.exchange.price_to_precision(symbol, limit_price)

            params = {"stopPrice": precise_stop}
            logger.info(
                f"Creating stop-loss {side} order: {precise_amount} {symbol} "
                f"stop={precise_stop} limit={precise_limit}"
            )
            return self._client.exchange.create_order(
                symbol, "stop_loss_limit", side, precise_amount, precise_limit, params
            )
        except Exception as e:
            logger.error(f"Failed to create stop-loss order: {e}")
            raise

    def create_take_profit(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        limit_price: float,
    ) -> dict:
        """Create a take-profit order (take_profit_limit).

        Args:
            symbol: Trading pair symbol.
            side: 'buy' or 'sell'.
            amount: Order quantity.
            stop_price: Trigger price.
            limit_price: Limit price after trigger.

        Returns:
            Order result dictionary.
        """
        try:
            if self._dry_run:
                order_id = self._make_dry_run_id()
                return self._dry_run_order(
                    order_id, symbol, "take_profit_limit", side, amount,
                    price=limit_price, stop_price=stop_price
                )

            precise_amount = self._client.exchange.amount_to_precision(symbol, amount)
            precise_stop = self._client.exchange.price_to_precision(symbol, stop_price)
            precise_limit = self._client.exchange.price_to_precision(symbol, limit_price)

            params = {"stopPrice": precise_stop}
            logger.info(
                f"Creating take-profit {side} order: {precise_amount} {symbol} "
                f"stop={precise_stop} limit={precise_limit}"
            )
            return self._client.exchange.create_order(
                symbol, "take_profit_limit", side, precise_amount, precise_limit, params
            )
        except Exception as e:
            logger.error(f"Failed to create take-profit order: {e}")
            raise

    def create_bracket_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> dict:
        """Create a bracket order (entry + SL + TP).

        Args:
            symbol: Trading pair symbol.
            side: 'buy' or 'sell' for the entry order.
            amount: Entry order quantity.
            entry_price: Limit entry price.
            stop_loss: Stop-loss trigger price.
            take_profit: Take-profit trigger price.

        Returns:
            Dict with entry_order_id, sl_order_id, tp_order_id.
        """
        try:
            # Determine SL/TP sides (opposite of entry)
            sl_side = "sell" if side == "buy" else "buy"
            tp_side = "sell" if side == "buy" else "buy"

            # For dry-run, compute SL/TP limit prices
            if self._dry_run:
                # SL limit slightly beyond stop for execution
                sl_limit_price = stop_loss * 0.995 if side == "buy" else stop_loss * 1.005
                tp_limit_price = take_profit * 0.995 if side == "sell" else take_profit * 1.005
            else:
                sl_limit_price = self._precision_price(symbol, stop_loss)
                tp_limit_price = self._precision_price(symbol, take_profit)

            entry = self.create_limit_order(symbol, side, amount, entry_price)

            sl = self.create_stop_loss(
                symbol, sl_side, amount, stop_loss, sl_limit_price
            )

            tp = self.create_take_profit(
                symbol, tp_side, amount, take_profit, tp_limit_price
            )

            result = {
                "entry_order_id": entry["id"],
                "sl_order_id": sl["id"],
                "tp_order_id": tp["id"],
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

            logger.info(
                f"Bracket order created: entry={entry['id']}, "
                f"SL={sl['id']}, TP={tp['id']}"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to create bracket order: {e}")
            # Attempt cleanup on failure
            raise

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel a specific order.

        Args:
            order_id: Order ID to cancel.
            symbol: Trading pair symbol.

        Returns:
            Cancellation result dictionary.
        """
        try:
            if self._dry_run:
                if order_id in self._dry_run_orders:
                    self._dry_run_orders[order_id]["status"] = "canceled"
                    logger.info(f"[DRY-RUN] Canceled order {order_id}")
                    return self._dry_run_orders[order_id]
                logger.warning(f"[DRY-RUN] Order {order_id} not found")
                return {"id": order_id, "status": "not_found"}

            logger.info(f"Canceling order {order_id} for {symbol}")
            return self._client.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise

    def cancel_all_orders(self, symbol: str) -> list[dict]:
        """Cancel all open orders for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            List of cancellation results.
        """
        try:
            if self._dry_run:
                results = []
                for oid, order in list(self._dry_run_orders.items()):
                    if order["symbol"] == symbol and order["status"] == "open":
                        order["status"] = "canceled"
                        results.append(order)
                        logger.info(f"[DRY-RUN] Canceled order {oid}")
                return results

            logger.info(f"Canceling all orders for {symbol}")
            return self._client.exchange.cancel_all_orders(symbol)
        except Exception as e:
            logger.error(f"Failed to cancel all orders for {symbol}: {e}")
            raise

    def get_order_status(self, order_id: str, symbol: str) -> str:
        """Get the status of an order.

        Args:
            order_id: Order ID.
            symbol: Trading pair symbol.

        Returns:
            Order status: 'open', 'closed', 'canceled', or 'expired'.
        """
        try:
            if self._dry_run:
                if order_id in self._dry_run_orders:
                    return self._dry_run_orders[order_id].get("status", "unknown")
                return "not_found"

            order = self._client.exchange.fetch_order(order_id, symbol)
            return order.get("status", "unknown")
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            raise
