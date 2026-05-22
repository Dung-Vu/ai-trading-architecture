"""AlertFormatter — Static methods for building formatted alert messages."""

from __future__ import annotations

from typing import Any


class AlertFormatter:
    """Builds HTML-formatted messages for Telegram alerts and reports."""

    @staticmethod
    def format_trade_alert(
        side: str,
        symbol: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
        strategy: str = "SMA Cross",
        ai_confidence: float | None = None,
        mode: str | None = None,
    ) -> str:
        """Return an HTML-formatted trade notification.

        Parameters
        ----------
        side : str
            "BUY" or "SELL"
        symbol : str
            Trading pair, e.g. "BTC/USDT"
        quantity : float
            Order quantity
        price : float
            Execution price
        pnl : float | None
            Realized P&L (optional)
        strategy : str
            Strategy name that triggered the trade
        """
        emoji = "🟢" if side.upper() == "BUY" else "🔴"

        msg = (
            f"{emoji} <b>TRADE EXECUTED</b>\n"
            f"<b>Symbol:</b> <code>{symbol}</code>\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Quantity:</b> {quantity}\n"
            f"<b>Price:</b> ${price:,.2f}\n"
            f"<b>Strategy:</b> <code>{strategy}</code>"
        )

        if ai_confidence is not None:
            msg += f"\n<b>AI Confidence:</b> <code>{ai_confidence:.1f}%</code>"

        if mode is not None:
            msg += f"\n<b>Mode:</b> <code>{mode}</code>"

        if pnl is not None:
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            msg += f"\n{pnl_emoji} <b>P&L:</b> <code>${pnl:+,.2f}</code>"

        return msg

    @staticmethod
    def _normalize_positions(positions: Any) -> list[dict[str, Any]]:
        """Normalize both list and dict representations of positions into list of dicts."""
        if not positions:
            return []
        normalized = []
        if isinstance(positions, dict):
            for sym, pos_data in positions.items():
                if isinstance(pos_data, dict):
                    # For format: { "BTC/USDT": { "quantity": 0.01, "value": 500 } }
                    pos_dict = {"symbol": sym}
                    for k, v in pos_data.items():
                        pos_dict[k] = v
                    normalized.append(pos_dict)
                else:
                    # For format: { "BTC/USDT": 5000 }
                    normalized.append({"symbol": sym, "value": pos_data})
        elif isinstance(positions, list):
            for p in positions:
                if isinstance(p, dict):
                    normalized.append(p)
                elif isinstance(p, str):
                    normalized.append({"symbol": p})
        return normalized

    @staticmethod
    def _format_position_line(pos: dict[str, Any]) -> str:
        """Render a single position line without inventing missing entry prices."""
        symbol = pos.get("symbol", "?")
        side = pos.get("side")
        quantity = pos.get("quantity")
        entry_price = pos.get("entry_price")
        value = pos.get("value")
        pnl = pos.get("pnl", 0.0)
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"

        line = f"\n  • <code>{symbol}</code>"
        if side:
            line += f" {side}"

        if quantity is not None and entry_price is not None:
            line += f" {quantity} @ ${entry_price:,.2f}"
        elif quantity is not None:
            line += f" qty={quantity}"

        if value is not None and entry_price is None:
            line += f" value=<code>${float(value):,.2f}</code>"

        line += f" {pnl_emoji} <code>${pnl:+,.2f}</code>"
        return line

    @staticmethod
    def format_status(
        mode: str,
        positions: list[dict[str, Any]] | dict[str, Any],
        daily_pnl: float,
        total_value: float,
        win_rate: float,
    ) -> str:
        """Return an HTML-formatted trading status message.

        Parameters
        ----------
        mode : str
            "dryrun" | "live"
        positions : list[dict] | dict
            Each dict or value: {symbol, side, quantity, entry_price, pnl}
        daily_pnl : float
            Today's P&L
        total_value : float
            Portfolio total value
        win_rate : float
            Win rate as percentage (0–100)
        """
        mode_emoji = "🔵" if mode == "dryrun" else "🟢"

        msg = (
            f"📊 <b>TRADING STATUS</b>\n"
            f"<b>Mode:</b> {mode_emoji} <code>{mode.upper()}</code>\n"
            f"<b>Portfolio Value:</b> <code>${total_value:,.2f}</code>\n"
            f"<b>Daily P&L:</b> {'🟢' if daily_pnl >= 0 else '🔴'} <code>${daily_pnl:+,.2f}</code>\n"
            f"<b>Win Rate:</b> <code>{win_rate:.1f}%</code>"
        )

        positions_list = AlertFormatter._normalize_positions(positions)
        if positions_list:
            msg += f"\n\n<b>Open Positions ({len(positions_list)})</b>"
            for pos in positions_list:
                msg += AlertFormatter._format_position_line(pos)
        else:
            msg += "\n\n📭 <b>No open positions</b>"

        return msg

    @staticmethod
    def format_daily_report(
        date: str,
        total_pnl: float,
        win_rate: float,
        total_trades: int,
        sharpe: float | None = None,
        max_dd: float | None = None,
        positions: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> str:
        """Return an HTML-formatted daily performance report.

        Parameters
        ----------
        date : str
            Report date
        total_pnl : float
            Total P&L for the day
        win_rate : float
            Win rate percentage
        total_trades : int
            Number of trades executed
        sharpe : float | None
            Sharpe ratio
        max_dd : float | None
            Maximum drawdown percentage
        positions : list[dict] | dict | None
            Currently open positions
        """
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"

        msg = (
            f"📈 <b>DAILY REPORT — {date}</b>\n"
            f"<b>Total P&L:</b> {pnl_emoji} <code>${total_pnl:+,.2f}</code>\n"
            f"<b>Win Rate:</b> <code>{win_rate:.1f}%</code>\n"
            f"<b>Total Trades:</b> <code>{total_trades}</code>"
        )

        if sharpe is not None:
            msg += f"\n<b>Sharpe Ratio:</b> <code>{sharpe:.2f}</code>"
        if max_dd is not None:
            dd_color = "🔴" if max_dd > 5 else "🟡"
            msg += f"\n{dd_color} <b>Max Drawdown:</b> <code>{max_dd:.2f}%</code>"

        positions_list = AlertFormatter._normalize_positions(positions)
        if positions_list:
            msg += f"\n\n<b>Open Positions ({len(positions_list)})</b>"
            for pos in positions_list:
                msg += AlertFormatter._format_position_line(pos)

        return msg

    @staticmethod
    def format_error(error_msg: str, context: str = "") -> str:
        """Return an HTML-formatted error alert.

        Parameters
        ----------
        error_msg : str
            Error message or traceback excerpt
        context : str
            Additional context about where the error occurred
        """
        msg = f"🚨 <b>ERROR ALERT</b>\n"
        if context:
            msg += f"<b>Context:</b> <code>{context}</code>\n"
        msg += f"<b>Error:</b>\n<code>{error_msg}</code>"
        return msg

    @staticmethod
    def format_kill_switch(reason: str, timestamp: str) -> str:
        """Return an HTML-formatted kill-switch activation message.

        Parameters
        ----------
        reason : str
            Reason the kill switch was triggered
        timestamp : str
            ISO-format timestamp of activation
        """
        return (
            f"🛑 <b>KILL SWITCH ACTIVATED</b>\n"
            f"<b>Reason:</b> {reason}\n"
            f"<b>Time:</b> <code>{timestamp}</code>\n"
            f"\n⚠️ All trading has been halted. Use /start_bot to re-arm."
        )
