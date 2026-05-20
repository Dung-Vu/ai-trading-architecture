"""TradingLogger — Structured logging for trades, decisions, and errors.

Uses loguru with file rotation (daily), error-only files, and JSONL trade logs.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class TradingLogger:
    """Configures and manages structured logging for the trading system."""

    def __init__(self, log_dir: str = "logs/", level: str = "INFO") -> None:
        """Initialise the logger (does not configure sinks — call setup() first).

        Parameters
        ----------
        log_dir : str
            Directory for log files
        level : str
            Minimum log level (e.g. "DEBUG", "INFO", "WARNING")
        """
        self._log_dir = Path(log_dir)
        self._level = level
        self._setup_done = False

        # Separate loguru logger instance for trade JSONL
        self._trade_logger = logger.bind(kind="trade")

    def setup(self) -> None:
        """Configure loguru sinks: console, daily rotating files, error file, JSONL."""
        if self._setup_done:
            logger.warning("TradingLogger.setup() called more than once — skipping.")
            return

        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Remove default sink
        logger.remove()

        # 1. Console output (colored, INFO+)
        logger.add(
            sys.stderr,
            level=self._level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
        )

        # 2. Daily rotating log file (all levels, 30 days retention)
        logger.add(
            self._log_dir / "trading_{time:YYYY-MM-DD}.log",
            level=self._level,
            rotation="00:00",
            retention="30 days",
            compression="zip",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            enqueue=True,
        )

        # 3. Error-only file (ERROR and above)
        logger.add(
            self._log_dir / "error_{time:YYYY-MM-DD}.log",
            level="ERROR",
            rotation="00:00",
            retention="30 days",
            compression="zip",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )

        self._setup_done = True
        logger.info(f"TradingLogger configured — log_dir={self._log_dir}, level={self._level}")

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_trade(self, trade_data: dict[str, Any]) -> None:
        """Log a trade execution to both the structured logger and a JSONL file.

        Parameters
        ----------
        trade_data : dict
            Required keys:
                - timestamp (str): ISO timestamp
                - symbol (str): Trading pair
                - side (str): "BUY" or "SELL"
                - quantity (float): Order quantity
                - price (float): Execution price
            Optional keys:
                - pnl (float): Realized P&L
                - strategy (str): Strategy name
                - mode (str): "dryrun" or "live"
        """
        ts = trade_data.get("timestamp", datetime.now(timezone.utc).isoformat())
        symbol = trade_data.get("symbol", "?")
        side = trade_data.get("side", "?")
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0)
        pnl = trade_data.get("pnl")
        strategy = trade_data.get("strategy", "unknown")
        mode = trade_data.get("mode", "dryrun")

        pnl_str = f"${pnl:+,.2f}" if pnl is not None else "N/A"

        logger.info(
            f"TRADE | {mode} | {side} {quantity} {symbol} @ ${price:,.2f} "
            f"| P&L: {pnl_str} | Strategy: {strategy}"
        )

        # Write to JSONL
        jsonl_path = self._log_dir / "trades.jsonl"
        record = {
            "timestamp": ts,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "pnl": pnl,
            "strategy": strategy,
            "mode": mode,
        }
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_decision(self, decision_data: dict[str, Any]) -> None:
        """Log an AI/strategy decision.

        Parameters
        ----------
        decision_data : dict
            Required keys:
                - timestamp (str): ISO timestamp
                - symbol (str): Trading pair
                - action (str): "BUY", "SELL", or "HOLD"
            Optional keys:
                - reason (str): Decision rationale
                - confidence (float): Confidence score 0–1
                - indicators (dict): Technical indicator values
        """
        ts = decision_data.get("timestamp", datetime.now(timezone.utc).isoformat())
        symbol = decision_data.get("symbol", "?")
        action = decision_data.get("action", "?")
        reason = decision_data.get("reason", "")
        confidence = decision_data.get("confidence")
        indicators = decision_data.get("indicators", {})

        conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        ind_str = ", ".join(f"{k}={v}" for k, v in indicators.items()) if indicators else ""

        msg = (
            f"DECISION | {symbol} → {action} | "
            f"Confidence: {conf_str} | Reason: {reason}"
        )
        if ind_str:
            msg += f" | Indicators: {ind_str}"

        logger.info(msg)

    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an error with full traceback.

        Parameters
        ----------
        error : Exception
            The exception instance
        context : str
            Additional context about where the error occurred
        """
        ctx_str = f" [{context}]" if context else ""
        logger.opt(exception=error).error(f"ERROR{ctx_str}: {error}")

    def log_info(self, message: str) -> None:
        """Log a general informational message.

        Parameters
        ----------
        message : str
            Message to log
        """
        logger.info(message)
