"""Trade fetching and performance metric helpers for AutoTuner."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from src.strategy.metrics import MetricsCalculator


class AutoTuneMetricsMixin:
    """Metrics and sample-data helpers used by ``AutoTuner``."""

    async def _fetch_recent_trades(self, days: int = 7) -> list[dict[str, Any]]:
        """Fetch trades from memory for the last N days."""
        if self.trade_memory is None:
            logger.warning("No trade memory available for optimization")
            return []

        try:
            if hasattr(self.trade_memory, "get_trade_history"):
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=days)

                import inspect

                if inspect.iscoroutinefunction(self.trade_memory.get_trade_history):
                    trades = await self.trade_memory.get_trade_history(
                        limit=500, start_date=start_date, end_date=end_date
                    )
                else:
                    trades = self.trade_memory.get_trade_history(
                        limit=500, start_date=start_date, end_date=end_date
                    )
                return trades
        except Exception as exc:
            logger.warning(f"Failed to fetch trades from memory: {exc}")

        return self._generate_sample_trades(days)

    def _generate_sample_trades(self, days: int = 7) -> list[dict[str, Any]]:
        """Generate sample trade data for testing/demo purposes."""
        import random

        random.seed(42)

        trades = []
        now = datetime.now(timezone.utc)
        num_trades = min(days * 3, 30)

        for _ in range(num_trades):
            pnl = random.gauss(50, 150)
            ts = now - timedelta(hours=random.randint(0, days * 24))

            trades.append(
                {
                    "timestamp": ts.isoformat(),
                    "symbol": random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT"]),
                    "side": random.choice(["BUY", "SELL"]),
                    "price": random.uniform(30000, 100000),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl / 1000 * 100, 2),
                    "strategy": self.strategy_name,
                    "indicators": {
                        "rsi": random.uniform(20, 80),
                        "sma_fast": random.uniform(30000, 100000),
                        "sma_slow": random.uniform(30000, 100000),
                    },
                    "market_conditions": {
                        "trend": random.choice(["uptrend", "downtrend", "sideways"]),
                        "volume_high": random.choice([True, False]),
                    },
                    "debate_result": {
                        "action": random.choice(["BUY", "SELL"]),
                        "confidence": random.uniform(50, 95),
                        "reasoning": "Sample reasoning for optimization",
                    },
                    "outcome_notes": "",
                }
            )

        return trades

    def _calculate_metrics(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate performance metrics from a trade list."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
            }

        pnls = [t.get("pnl", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls)
        win_rate = len(wins) / len(pnls)
        sharpe = MetricsCalculator.calc_trade_pnl_sharpe_ratio(
            pnls, risk_free_rate=0.0
        )
        max_dd = MetricsCalculator.calc_trade_pnl_max_drawdown(pnls)

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trads": len(losses),
            "win_rate": win_rate,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4),
            "profit_factor": round(profit_factor, 3),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
        }
