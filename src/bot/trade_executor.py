"""Shared price, risk, SL/TP, and order execution helpers for trading bots."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from loguru import logger

from src.config import get_default_quantity_pct
from src.runtime_status import RuntimeFailurePolicy, RuntimeStatus
from src.execution.trade_utils import calculate_realized_pnl, is_exit_order_triggered
from src.shared_utils import normalize_market_symbol


class DryRunExecutorLike(Protocol):
    """Protocol for the dry-run executors used by the trading bots."""

    def get_portfolio(self) -> dict[str, Any]: ...

    def simulate_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: str,
    ) -> dict[str, Any]: ...

    def simulate_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: str,
    ) -> dict[str, Any]: ...

    def simulate_sl_tp(
        self,
        orders: list[dict[str, Any]],
        prices: dict[str, float],
    ) -> list[dict[str, Any]]: ...


class TradeExecutionMixin:
    def _get_initial_capital(self) -> float:
        """Return configured initial capital without embedding a money default."""
        trading_config = getattr(self.config, "trading", None)
        value = getattr(trading_config, "initial_capital", None)
        if value is None:
            value = getattr(self.config, "initial_capital", 0.0)
        return float(value or 0.0)

    def _make_pending_order_id(self, prefix: str, symbol: str) -> str:
        """Generate collision-resistant IDs for simulated SL/TP orders."""
        safe_symbol = symbol.replace("/", "_")
        return f"{prefix}_{safe_symbol}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"

    def _register_exit_orders(
        self,
        symbol: str,
        quantity: float,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> None:
        """Register sell-side exit orders for a long spot position."""
        if stop_loss and stop_loss > 0:
            self._pending_sl_tp.append({
                "id": self._make_pending_order_id("sl", symbol),
                "symbol": symbol,
                "side": "sell",
                "type": "stop_loss",
                "stop_price": float(stop_loss),
                "quantity": quantity,
                "direction": "below",
            })
            logger.info(f"Registered Stop Loss at ${stop_loss:.2f} for {symbol}")

        if take_profit and take_profit > 0:
            self._pending_sl_tp.append({
                "id": self._make_pending_order_id("tp", symbol),
                "symbol": symbol,
                "side": "sell",
                "type": "take_profit",
                "stop_price": float(take_profit),
                "quantity": quantity,
                "direction": "above",
            })
            logger.info(f"Registered Take Profit at ${take_profit:.2f} for {symbol}")

    def _get_dry_run_executor(self) -> DryRunExecutorLike | None:
        """Return whichever dry-run executor attribute the bot uses."""
        return getattr(self, "_executor", None) or getattr(self, "_dry_run_executor", None)

    async def _get_latest_price(
        self,
        symbol: str,
        *,
        log_failures: bool = False,
    ) -> float | None:
        """Get latest price from Redis, falling back to exchange ticker when available."""
        if self._redis_cache is not None:
            try:
                redis_symbol = normalize_market_symbol(symbol)
                data = await self._redis_cache.get_latest_price(redis_symbol)
                if data and "price" in data:
                    return float(data["price"])
            except Exception as exc:
                log_method = logger.warning if log_failures else logger.debug
                log_method(f"Failed to get price from Redis for {symbol}: {exc}")

        if self.mode != "dryrun" and self._order_manager is not None:
            client = getattr(self._order_manager, "_client", None)
            fetch_ticker = getattr(client, "fetch_ticker", None)
            if callable(fetch_ticker):
                try:
                    loop = asyncio.get_running_loop()
                    ticker = await loop.run_in_executor(None, lambda: fetch_ticker(symbol))
                    if ticker and ticker.get("last") is not None:
                        return float(ticker["last"])
                except Exception as exc:
                    log_method = logger.warning if log_failures else logger.debug
                    log_method(f"Failed to get exchange ticker for {symbol}: {exc}")

        return None

    def _build_default_portfolio_state(self) -> dict[str, Any]:
        """Return a safe portfolio fallback when no live state is available."""
        initial_capital = self._get_initial_capital()
        return {
            "cash": initial_capital,
            "positions": {},
            "total_value": initial_capital,
            "initial_balance": initial_capital,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
        }

    def _get_quote_asset(self) -> str:
        """Return the quote asset used for portfolio valuation."""
        if not self.symbols:
            return "USDT"

        parts = self.symbols[0].split("/")
        return parts[1] if len(parts) > 1 else "USDT"

    async def _build_live_position_snapshot(
        self,
        asset_symbol: str,
        total_qty: float,
    ) -> tuple[dict[str, Any], float]:
        """Build a normalized portfolio entry for one non-quote asset."""
        current_price = await self._get_latest_price(asset_symbol, log_failures=True)
        if current_price is None:
            logger.warning(
                f"Using zero price fallback for {asset_symbol} in portfolio snapshot"
            )
            current_price = 0.0

        market_value = total_qty * current_price
        return {
            "quantity": total_qty,
            "avg_price": current_price,
            "cost_basis": market_value,
            "market_value": market_value,
            "unrealized_pnl": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, market_value

    async def _build_live_positions_state(
        self,
        raw_balance: dict[str, Any],
        cash: float,
        quote_asset: str,
    ) -> tuple[dict[str, dict[str, Any]], float]:
        """Convert exchange balances into the shared portfolio-state shape."""
        positions_data: dict[str, dict[str, Any]] = {}
        total_value = cash

        for asset, bal in raw_balance.items():
            if asset in ("free", "used", "total", "info", quote_asset):
                continue

            total_qty = float(bal.get("total", 0.0))
            if total_qty <= 0.000001:
                continue

            asset_symbol = f"{asset}/{quote_asset}"
            position_data, market_value = await self._build_live_position_snapshot(
                asset_symbol,
                total_qty,
            )
            positions_data[asset_symbol] = position_data
            total_value += market_value

        return positions_data, total_value

    async def _get_portfolio_state(self) -> dict[str, Any]:
        """Get the current portfolio state (cash, positions, total equity)."""
        dry_run_executor = self._get_dry_run_executor()
        if self.mode == "dryrun":
            if dry_run_executor:
                return dry_run_executor.get_portfolio()
            return self._build_default_portfolio_state()

        try:
            loop = asyncio.get_running_loop()
            raw_balance = await loop.run_in_executor(
                None,
                self._order_manager._client.fetch_balance,
            )

            quote_asset = self._get_quote_asset()
            cash = float(raw_balance.get(quote_asset, {}).get("free", 0.0))
            positions_data, total_value = await self._build_live_positions_state(
                raw_balance,
                cash,
                quote_asset,
            )

            initial_balance = self._get_initial_capital()
            total_pnl = total_value - initial_balance
            total_pnl_pct = (total_pnl / initial_balance * 100) if initial_balance > 0 else 0.0

            return {
                "cash": cash,
                "positions": positions_data,
                "total_value": total_value,
                "initial_balance": initial_balance,
                "total_pnl": total_pnl,
                "total_pnl_pct": total_pnl_pct,
            }
        except Exception as exc:
            self._runtime_failure(
                "portfolio_state_fetch_failed",
                f"Failed to fetch portfolio state from exchange: {exc}",
                policy=RuntimeFailurePolicy.FALLBACK,
                log_level="error",
            )
            return self._build_default_portfolio_state()

    async def _run_risk_check(
        self,
        symbol: str,
        action: str,
        price: float,
    ) -> tuple[bool, str]:
        """Run the shared risk-engine pre-trade check."""
        if self._risk_engine is None:
            return True, "Risk engine not available"

        portfolio = await self._get_portfolio_state()
        positions = portfolio.get("positions", {})

        quantity_pct = getattr(
            self.config,
            "default_quantity_pct",
            get_default_quantity_pct(),
        )
        available = portfolio["cash"] * quantity_pct
        quantity = available / price if price > 0 else 0

        if quantity <= 0:
            return False, "Insufficient cash for minimum quantity"

        approved, reason = self._risk_engine.pre_trade_checks(
            symbol=symbol,
            side=action.lower(),
            quantity=quantity,
            price=price,
            current_equity=portfolio["total_value"],
            start_equity=self.config.trading.initial_capital,
            positions=positions,
        )

        return approved, reason

    def _get_symbol_exit_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Return pending SL/TP orders for a single symbol."""
        return [order for order in self._pending_sl_tp if order["symbol"] == symbol]

    async def _finalize_triggered_exit(
        self,
        symbol: str,
        trade_result: dict[str, Any],
        alert_sender: Any,
        *,
        action: str,
    ) -> None:
        """Clear local position state and emit the standard close alert."""
        closed_position = dict(self._positions.get(symbol, {}))
        if symbol in self._positions:
            del self._positions[symbol]
        if closed_position:
            trade_result["closed_position"] = closed_position

        self._clear_exit_orders(symbol)
        self._on_position_closed(symbol, trade_result, closed_position or None)
        await alert_sender(
            symbol,
            trade_result,
            {"action": action, "confidence": 100, "rounds": 0},
        )

    async def _handle_dry_run_exit_orders(
        self,
        symbol: str,
        current_price: float,
        symbol_orders: list[dict[str, Any]],
        alert_sender: Any,
        dry_run_executor: Any,
    ) -> None:
        """Evaluate pending SL/TP orders against the dry-run executor."""
        try:
            triggered = dry_run_executor.simulate_sl_tp(
                symbol_orders,
                {symbol: current_price},
            )
        except Exception as exc:
            self._runtime_failure(
                "dry_run_exit_evaluation_failed",
                f"Error evaluating dry-run SL/TP for {symbol}: {exc}",
                policy=RuntimeFailurePolicy.FALLBACK,
                log_level="error",
            )
            return

        for trade_result in triggered:
            triggered_by = trade_result.get("triggered_by", "sl/tp")
            logger.info(
                f"🚨 [{symbol}] {triggered_by.upper()} triggered! "
                f"Closed position @ {current_price}"
            )
            await self._finalize_triggered_exit(
                symbol,
                trade_result,
                alert_sender,
                action="SELL",
            )

    def _get_triggered_live_exit_orders(
        self,
        symbol_orders: list[dict[str, Any]],
        current_price: float,
    ) -> list[dict[str, Any]]:
        """Return live SL/TP orders whose trigger price has been reached."""
        triggered_orders: list[dict[str, Any]] = []
        for order in symbol_orders:
            if is_exit_order_triggered(
                order.get("direction", ""),
                current_price,
                order["stop_price"],
            ):
                triggered_orders.append(order)
        return triggered_orders

    def _calculate_exit_pnl(
        self,
        symbol: str,
        side_to_execute: str,
        exit_price: float,
        quantity: float,
    ) -> tuple[float, float]:
        """Calculate realized PnL for a closing spot trade."""
        open_position = self._positions.get(symbol)
        if not open_position:
            return 0.0, 0.0

        entry_price = float(open_position.get("entry_price", 0.0))
        if entry_price <= 0:
            return 0.0, 0.0

        return calculate_realized_pnl(
            entry_price,
            exit_price,
            quantity,
            side_to_execute=side_to_execute,
        )

    async def _execute_live_exit_order(
        self,
        symbol: str,
        current_price: float,
        trigger_order: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Execute a real/testnet SL/TP close order and normalize the result."""
        if self._order_manager is None:
            self._runtime_failure(
                "live_exit_order_unavailable",
                f"Cannot execute live SL/TP order for {symbol}: no order manager",
                policy=RuntimeFailurePolicy.RETURN_STATUS,
                log_level="error",
            )
            return None

        side_to_execute = trigger_order["side"]
        amount_to_execute = trigger_order["quantity"]
        portfolio = await self._get_portfolio_state()

        logger.info(
            f"🚨 Sending real market {side_to_execute.upper()} "
            f"SL/TP close order to exchange for {symbol}..."
        )
        try:
            loop = asyncio.get_running_loop()
            order = await loop.run_in_executor(
                None,
                lambda: self._order_manager.create_market_order(
                    symbol=symbol,
                    side=side_to_execute,
                    amount=amount_to_execute,
                ),
            )
        except Exception as exc:
            self._runtime_failure(
                "live_exit_order_failed",
                f"Failed to execute real SL/TP close order on exchange: {exc}",
                policy=RuntimeFailurePolicy.RETURN_STATUS,
                log_level="error",
            )
            return None

        avg_fill_price = float(order.get("average", current_price) or current_price)
        filled_qty = float(order.get("filled", amount_to_execute))
        pnl, pnl_pct = self._calculate_exit_pnl(
            symbol,
            side_to_execute,
            avg_fill_price,
            filled_qty,
        )

        result = {
            "trade_id": order.get("id"),
            "symbol": symbol,
            "side": side_to_execute,
            "quantity": filled_qty,
            "price": avg_fill_price,
            "revenue": filled_qty * avg_fill_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "cash_total": portfolio["cash"]
            + (
                filled_qty * avg_fill_price
                if side_to_execute == "sell"
                else -filled_qty * avg_fill_price
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "triggered_by": trigger_order.get("type"),
            "order_info": order,
        }
        self._runtime_success(
            "live_exit_order_executed",
            f"Executed live exit order for {symbol}",
        )
        return result

    async def _handle_pending_exit_orders(
        self,
        symbol: str,
        current_price: float,
        alert_sender: Any,
    ) -> None:
        """Evaluate and execute any pending stop-loss / take-profit orders."""
        if not self._pending_sl_tp:
            return

        symbol_orders = self._get_symbol_exit_orders(symbol)
        if not symbol_orders:
            return

        dry_run_executor = self._get_dry_run_executor()
        if self.mode == "dryrun" and dry_run_executor:
            await self._handle_dry_run_exit_orders(
                symbol,
                current_price,
                symbol_orders,
                alert_sender,
                dry_run_executor,
            )
            return

        for trg_order in self._get_triggered_live_exit_orders(symbol_orders, current_price):
            logger.info(
                f"🚨 [LIVE] {trg_order['type'].upper()} trigger price reached: "
                f"${current_price:.2f} (stop was ${trg_order['stop_price']:.2f})"
            )
            trg_result = await self._execute_live_exit_order(
                symbol,
                current_price,
                trg_order,
            )
            if trg_result is None:
                continue

            await self._finalize_triggered_exit(
                symbol,
                trg_result,
                alert_sender,
                action=trg_order["side"].upper(),
            )
            break

    def _calculate_buy_quantity(self, portfolio: dict[str, Any], price: float) -> float:
        """Calculate buy quantity from available cash."""
        quantity_pct = getattr(
            self.config,
            "default_quantity_pct",
            get_default_quantity_pct(),
        )
        available = portfolio["cash"] * quantity_pct
        return available / price if price > 0 else 0.0

    def _apply_amount_precision(self, symbol: str, quantity: float) -> float:
        """Apply exchange precision rules when available."""
        if self.mode == "dryrun":
            return quantity

        try:
            precise_quantity = self._order_manager._precision_amount(symbol, quantity)
            return float(precise_quantity)
        except Exception as exc:
            self._runtime_failure(
                "amount_precision_failed",
                f"[{symbol}] Failed to apply amount precision: {exc}",
                policy=RuntimeFailurePolicy.FALLBACK,
                log_level="warning",
            )
            return 0.0

    def _get_open_spot_position(
        self,
        symbol: str,
        portfolio: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return the currently open spot position for a symbol, if any."""
        return self._positions.get(symbol) or portfolio.get("positions", {}).get(symbol)

    def _clear_exit_orders(self, symbol: str) -> None:
        """Remove any pending exit orders for a symbol."""
        self._pending_sl_tp = [o for o in self._pending_sl_tp if o["symbol"] != symbol]

    def _on_position_closed(
        self,
        symbol: str,
        trade_result: dict[str, Any],
        closed_position: dict[str, Any] | None = None,
    ) -> None:
        """Hook for subclasses that need to react to position close events."""
        pass

    def _enrich_trade_result(
        self,
        trade_result: dict[str, Any],
        debate_result: dict[str, Any],
        *,
        include_exit_targets: bool = False,
    ) -> dict[str, Any]:
        """Attach shared metadata to a trade result."""
        trade_result["strategy"] = self._get_strategy_name()
        trade_result["ai_confidence"] = debate_result.get("confidence", 50)
        if include_exit_targets:
            trade_result["stop_loss"] = debate_result.get("stop_loss")
            trade_result["take_profit"] = debate_result.get("take_profit")
        return trade_result

    async def _execute_trade_with_status(
        self,
        symbol: str,
        action: str,
        price: float,
        debate_result: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, RuntimeStatus]:
        """Execute a trade using shared buy/sell helpers and return a typed status."""
        normalized_action = action.upper()
        portfolio = await self._get_portfolio_state()

        try:
            if normalized_action == "BUY":
                trade_result = await self._execute_buy(
                    symbol,
                    price,
                    portfolio,
                    debate_result,
                )
            elif normalized_action == "SELL":
                trade_result = await self._execute_sell(
                    symbol,
                    price,
                    portfolio,
                    debate_result,
                )
            else:
                logger.debug(f"[{symbol}] No execution needed for {normalized_action}")
                return None, self._runtime_failure(
                    "trade_not_executed",
                    f"[{symbol}] No execution needed for {normalized_action}",
                    policy=RuntimeFailurePolicy.FALLBACK,
                    log_level="debug",
                )
        except Exception as exc:
            return None, self._runtime_failure(
                "trade_execution_failed",
                f"[{symbol}] Trade execution failed: {exc}",
                policy=RuntimeFailurePolicy.RETURN_STATUS,
                log_level="error",
            )

        if trade_result is None:
            return None, self._runtime_failure(
                "trade_not_executed",
                f"[{symbol}] {normalized_action} produced no executable trade",
                policy=RuntimeFailurePolicy.FALLBACK,
                log_level="debug",
            )

        self._trade_count += 1
        return trade_result, self._runtime_success(
            "trade_executed",
            f"Executed {normalized_action} for {symbol}",
        )

    async def _execute_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        debate_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        trade_result, _status = await self._execute_trade_with_status(
            symbol,
            action,
            price,
            debate_result,
        )
        return trade_result

    async def _execute_buy(
        self,
        symbol: str,
        price: float,
        portfolio: dict[str, Any],
        debate_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Execute a buy order via dry-run or exchange routing."""
        quantity = self._calculate_buy_quantity(portfolio, price)
        quantity = self._apply_amount_precision(symbol, quantity)

        if quantity <= 0:
            logger.warning(
                f"[{symbol}] Insufficient funds for BUY (quantity={quantity:.6f})"
            )
            return None

        timestamp = datetime.now(timezone.utc).isoformat()
        if self.mode == "dryrun":
            return self._execute_dry_run_buy(
                symbol,
                quantity,
                price,
                timestamp,
                debate_result,
            )

        return await self._execute_exchange_buy(
            symbol,
            quantity,
            price,
            portfolio,
            timestamp,
            debate_result,
        )

    async def _execute_sell(
        self,
        symbol: str,
        price: float,
        portfolio: dict[str, Any],
        debate_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Execute a sell order via dry-run or exchange routing."""
        open_position = self._get_open_spot_position(symbol, portfolio)
        if open_position is None:
            logger.warning(
                f"[{symbol}] SELL signal ignored: no open spot position to close"
            )
            return None

        sell_quantity = float(open_position.get("quantity", 0.0))
        if sell_quantity <= 0:
            logger.warning(f"[{symbol}] SELL signal ignored: empty open quantity")
            return None

        timestamp = datetime.now(timezone.utc).isoformat()
        if self.mode == "dryrun":
            return self._execute_dry_run_sell(
                symbol,
                sell_quantity,
                price,
                timestamp,
                debate_result,
            )

        return await self._execute_exchange_sell(
            symbol,
            sell_quantity,
            price,
            portfolio,
            timestamp,
            debate_result,
        )

    def _execute_dry_run_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: str,
        debate_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Execute a dry-run buy and register local position state."""
        dry_run_executor = self._get_dry_run_executor()
        if dry_run_executor is None:
            logger.error(f"[{symbol}] No dry-run executor available")
            return None

        result = dry_run_executor.simulate_buy(
            symbol=symbol,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
        )
        result["timestamp"] = timestamp

        self._positions[symbol] = {
            "side": "LONG",
            "quantity": quantity,
            "entry_price": price,
            "entry_time": timestamp,
        }

        self._register_exit_orders(
            symbol=symbol,
            quantity=quantity,
            stop_loss=debate_result.get("stop_loss"),
            take_profit=debate_result.get("take_profit"),
        )

        return self._enrich_trade_result(
            result,
            debate_result,
            include_exit_targets=True,
        )

    def _execute_dry_run_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: str,
        debate_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Execute a dry-run sell and clear local position state."""
        dry_run_executor = self._get_dry_run_executor()
        if dry_run_executor is None:
            logger.error(f"[{symbol}] No dry-run executor available")
            return None

        result = dry_run_executor.simulate_sell(
            symbol=symbol,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
        )
        result["timestamp"] = timestamp

        closed_position = dict(self._positions.get(symbol, {}))
        if symbol in self._positions:
            del self._positions[symbol]
        self._clear_exit_orders(symbol)

        if closed_position:
            result["closed_position"] = closed_position
        self._on_position_closed(symbol, result, closed_position or None)

        return self._enrich_trade_result(result, debate_result)

    async def _execute_exchange_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        portfolio: dict[str, Any],
        timestamp: str,
        debate_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a real/testnet market buy."""
        logger.info(f"🚨 Sending real market BUY order to exchange for {symbol}...")
        loop = asyncio.get_running_loop()
        order = await loop.run_in_executor(
            None,
            lambda: self._order_manager.create_market_order(
                symbol=symbol,
                side="buy",
                amount=quantity,
            ),
        )

        filled_qty = float(order.get("filled", quantity))
        avg_fill_price = float(order.get("average", price) or price)

        result = {
            "trade_id": order.get("id"),
            "symbol": symbol,
            "side": "buy",
            "quantity": filled_qty,
            "price": avg_fill_price,
            "cost": filled_qty * avg_fill_price,
            "cash_remaining": portfolio["cash"] - (filled_qty * avg_fill_price),
            "timestamp": timestamp,
            "order_info": order,
        }

        self._positions[symbol] = {
            "side": "LONG",
            "quantity": filled_qty,
            "entry_price": avg_fill_price,
            "entry_time": timestamp,
        }

        self._register_exit_orders(
            symbol=symbol,
            quantity=filled_qty,
            stop_loss=debate_result.get("stop_loss"),
            take_profit=debate_result.get("take_profit"),
        )

        return self._enrich_trade_result(
            result,
            debate_result,
            include_exit_targets=True,
        )

    async def _execute_exchange_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        portfolio: dict[str, Any],
        timestamp: str,
        debate_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a real/testnet market sell."""
        logger.info(f"🚨 Sending real market SELL order to exchange for {symbol}...")
        loop = asyncio.get_running_loop()
        order = await loop.run_in_executor(
            None,
            lambda: self._order_manager.create_market_order(
                symbol=symbol,
                side="sell",
                amount=quantity,
            ),
        )

        filled_qty = float(order.get("filled", quantity))
        avg_fill_price = float(order.get("average", price) or price)

        closed_position = dict(self._positions.get(symbol, {}))
        pnl, pnl_pct = self._calculate_exit_pnl(
            symbol,
            "sell",
            avg_fill_price,
            filled_qty,
        )
        if symbol in self._positions:
            del self._positions[symbol]

        self._clear_exit_orders(symbol)

        result = {
            "trade_id": order.get("id"),
            "symbol": symbol,
            "side": "sell",
            "quantity": filled_qty,
            "price": avg_fill_price,
            "revenue": filled_qty * avg_fill_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "cash_total": portfolio["cash"] + (filled_qty * avg_fill_price),
            "timestamp": timestamp,
            "order_info": order,
        }

        if closed_position:
            result["closed_position"] = closed_position
        self._on_position_closed(symbol, result, closed_position or None)

        return self._enrich_trade_result(result, debate_result)
