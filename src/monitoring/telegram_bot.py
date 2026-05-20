"""TelegramBot — Telegram bot interface for trading notifications and commands."""

from __future__ import annotations

import logging
from datetime import UTC
from typing import TYPE_CHECKING, Any

from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

if TYPE_CHECKING:
    from .trading_logger import TradingLogger


logger = logging.getLogger(__name__)


class TelegramBot:
    """Async Telegram bot for trading alerts and interactive commands.

    Uses python-telegram-bot v21+ ApplicationBuilder pattern.
    All handlers are async. ParseMode is HTML.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        trading_logger: TradingLogger | None = None,
    ) -> None:
        """Initialize the Telegram bot.

        Parameters
        ----------
        bot_token : str
            Telegram Bot API token (from @BotFather)
        chat_id : str
            Target chat ID for alerts
        trading_logger : TradingLogger | None
            Optional logger for recording command usage
        """
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._logger = trading_logger
        self._application: Application | None = None

        # Trading state shared with the rest of the system
        self._mode: str = "dryrun"
        self._positions: list[dict[str, Any]] = []
        self._daily_pnl: float = 0.0
        self._total_value: float = 0.0
        self._win_rate: float = 0.0
        self._is_running: bool = False

    async def _reply(self, update: Update, message: str) -> None:
        """Reply to a command update when Telegram included a message."""
        if update.message is None:
            logger.warning("Received Telegram command update without a message.")
            return
        await update.message.reply_text(message, parse_mode=ParseMode.HTML)

    # ------------------------------------------------------------------
    # Public state setters (called by other modules)
    # ------------------------------------------------------------------

    def set_trading_state(
        self,
        mode: str = "dryrun",
        positions: list[dict[str, Any]] | None = None,
        daily_pnl: float = 0.0,
        total_value: float = 0.0,
        win_rate: float = 0.0,
    ) -> None:
        """Update internal trading state for /status responses.

        Parameters
        ----------
        mode : str
            "dryrun" or "live"
        positions : list[dict]
            Currently open positions
        daily_pnl : float
            Today's realised + unrealised P&L
        total_value : float
            Total portfolio value
        win_rate : float
            Win-rate percentage
        """
        self._mode = mode
        self._positions = positions or []
        self._daily_pnl = daily_pnl
        self._total_value = total_value
        self._win_rate = win_rate

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_polling(self) -> None:
        """Start the bot polling loop (blocks until stop)."""
        self._application = ApplicationBuilder().token(self._bot_token).build()

        # Register command handlers
        self._application.add_handler(CommandHandler("start", self._cmd_start))
        self._application.add_handler(CommandHandler("status", self._cmd_status))
        self._application.add_handler(CommandHandler("balance", self._cmd_balance))
        self._application.add_handler(CommandHandler("stop", self._cmd_stop))
        self._application.add_handler(CommandHandler("start_bot", self._cmd_start_bot))
        self._application.add_handler(CommandHandler("pnl", self._cmd_pnl))

        logger.info("Starting Telegram bot polling…")
        self._is_running = True
        self._application.run_polling(drop_pending_updates=True)

    async def stop_polling(self) -> None:
        """Gracefully stop the bot."""
        if self._application:
            await self._application.shutdown()
            self._is_running = False
            logger.info("Telegram bot stopped.")

    # ------------------------------------------------------------------
    # Command handlers (all async)
    # ------------------------------------------------------------------

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start — welcome message with command list."""
        msg = (
            "🤖 <b>AI Trading Bot Online!</b>\n"
            "\n<b>Available commands:</b>\n"
            "  /status — Current trading status\n"
            "  /balance — Portfolio balance\n"
            "  /stop — Kill switch (halt trading)\n"
            "  /start_bot — Re-arm after kill switch\n"
            "  /pnl — Today's P&L summary"
        )
        await self._reply(update, msg)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status — show trading status."""
        from .alert_formatter import AlertFormatter

        msg = AlertFormatter.format_status(
            mode=self._mode,
            positions=self._positions,
            daily_pnl=self._daily_pnl,
            total_value=self._total_value,
            win_rate=self._win_rate,
        )
        await self._reply(update, msg)

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /balance — show portfolio balance."""
        msg = (
            "💰 <b>PORTFOLIO BALANCE</b>\n"
            f"<b>Total Value:</b> <code>${self._total_value:,.2f}</code>\n"
            f"<b>Daily P&L:</b> {'🟢' if self._daily_pnl >= 0 else '🔴'} <code>${self._daily_pnl:+,.2f}</code>"
        )
        await self._reply(update, msg)

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stop — trigger kill switch."""
        from datetime import datetime

        from .alert_formatter import AlertFormatter

        ts = datetime.now(UTC).isoformat()
        msg = AlertFormatter.format_kill_switch(
            reason="Manual stop via /stop command",
            timestamp=ts,
        )
        await self._reply(update, msg)
        self._is_running = False

        # Log the kill-switch event
        if self._logger:
            self._logger.log_info(f"Kill switch activated via Telegram at {ts}")

    async def _cmd_start_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start_bot — re-arm after kill switch."""
        self._is_running = True
        msg = (
            "✅ <b>BOT RE-ARMED</b>\n"
            "Trading has been resumed. Monitor /status for updates."
        )
        await self._reply(update, msg)

        if self._logger:
            self._logger.log_info("Bot re-armed via Telegram /start_bot")

    async def _cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pnl — show today's P&L summary."""
        pnl_emoji = "🟢" if self._daily_pnl >= 0 else "🔴"
        msg = (
            f"📊 <b>TODAY'S P&L</b>\n"
            f"{pnl_emoji} <b>P&L:</b> <code>${self._daily_pnl:+,.2f}</code>\n"
            f"<b>Win Rate:</b> <code>{self._win_rate:.1f}%</code>"
        )
        await self._reply(update, msg)

    # ------------------------------------------------------------------
    # Alert-sending methods (called by other modules)
    # ------------------------------------------------------------------

    async def send_alert(self, message: str, parse_mode: str = "HTML") -> None:
        """Send a free-form alert message to the configured chat.

        Parameters
        ----------
        message : str
            Alert text (HTML or plain text)
        parse_mode : str
            "HTML" (default) or "Markdown"
        """
        pm = ParseMode.HTML if parse_mode == "HTML" else ParseMode.MARKDOWN
        if not self._application:
            logger.info("Telegram polling is not running; sending one-shot alert.")
            bot = Bot(token=self._bot_token)
            await bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode=pm,
            )
            return

        if not self._application:
            logger.warning("Cannot send alert — bot is not running.")
            return

        pm = ParseMode.HTML if parse_mode == "HTML" else ParseMode.MARKDOWN
        await self._application.bot.send_message(
            chat_id=self._chat_id,
            text=message,
            parse_mode=pm,
        )

    async def send_trade_alert(
        self,
        side: str,
        symbol: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
    ) -> None:
        """Send a formatted trade-execution alert.

        Parameters
        ----------
        side : str
            "buy" or "sell"
        symbol : str
            Trading pair
        quantity : float
            Order quantity
        price : float
            Execution price
        pnl : float | None
            Realized P&L (optional)
        """
        from .alert_formatter import AlertFormatter

        msg = AlertFormatter.format_trade_alert(
            side=side,
            symbol=symbol,
            quantity=quantity,
            price=price,
            pnl=pnl,
        )
        await self.send_alert(msg)

    async def send_daily_report(self, portfolio_data: dict[str, Any]) -> None:
        """Send a formatted daily performance report.

        Parameters
        ----------
        portfolio_data : dict
            Keys: date, total_pnl, win_rate, total_trades,
                  sharpe (optional), max_dd (optional), positions (optional)
        """
        from .alert_formatter import AlertFormatter

        msg = AlertFormatter.format_daily_report(
            date=portfolio_data.get("date", "Today"),
            total_pnl=portfolio_data.get("total_pnl", 0.0),
            win_rate=portfolio_data.get("win_rate", 0.0),
            total_trades=portfolio_data.get("total_trades", 0),
            sharpe=portfolio_data.get("sharpe"),
            max_dd=portfolio_data.get("max_dd"),
            positions=portfolio_data.get("positions"),
        )
        await self.send_alert(msg)
