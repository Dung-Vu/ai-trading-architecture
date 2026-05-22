"""Performance analytics and pattern detection for TradeMemory."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from src.strategy.metrics import MetricsCalculator
from .schema import PerformanceSummary


class TradeMemoryAnalyticsMixin:
    async def get_performance_summary(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        strategy: str | None = None,
    ) -> PerformanceSummary:
        """
        Compute aggregated performance metrics.

        Args:
            start_date: Start of period.
            end_date: End of period.
            strategy: Filter by strategy name.

        Returns:
            PerformanceSummary with all computed metrics.
        """
        self._require_connected()
        if self._pg_pool is None:
            raise RuntimeError("PostgreSQL pool is None")

        # Only analyze closing trades (SELL side with realized PnL)
        conditions = ["side = 'SELL'", "pnl != 0"]
        params: list[Any] = []
        param_idx = 1

        if start_date:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(start_date)
            param_idx += 1

        if end_date:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(end_date)
            param_idx += 1

        if strategy:
            conditions.append(f"strategy = ${param_idx}")
            params.append(strategy)
            param_idx += 1

        where = " AND ".join(conditions)

        async with self._pg_pool.acquire() as conn:
            # Get aggregate stats
            stats = await conn.fetchrow(
                f"""
                SELECT
                    COUNT(*) as total_trades,
                    COUNT(*) FILTER (WHERE pnl > 0) as winning_trades,
                    COUNT(*) FILTER (WHERE pnl <= 0) as losing_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    COALESCE(MAX(pnl), 0) as best_trade,
                    COALESCE(MIN(pnl) FILTER (WHERE pnl < 0), 0) as worst_trade,
                    COALESCE(SUM(pnl) FILTER (WHERE pnl > 0), 0) as gross_profit,
                    COALESCE(ABS(SUM(pnl)) FILTER (WHERE pnl < 0), 0) as gross_loss
                FROM trades WHERE {where}
                """,
                *params,
            )

            if stats is None or stats["total_trades"] == 0:
                return PerformanceSummary()

            # Get all PnLs for Sharpe calculation
            pnl_rows = await conn.fetch(
                f"SELECT pnl FROM trades WHERE {where} ORDER BY timestamp", *params
            )
            pnls = [float(row["pnl"]) for row in pnl_rows]

            max_dd = round(MetricsCalculator.calc_trade_pnl_max_drawdown(pnls) * 100, 2)
            sharpe = round(
                MetricsCalculator.calc_trade_pnl_sharpe_ratio(
                    pnls,
                    risk_free_rate=0.0,
                    periods_per_year=250,
                ),
                3,
            )

            gross_profit = float(stats["gross_profit"])
            gross_loss = float(stats["gross_loss"])
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

            total = int(stats["total_trades"])
            wins = int(stats["winning_trades"])

            return PerformanceSummary(
                total_trades=total,
                winning_trades=wins,
                losing_trades=int(stats["losing_trades"]),
                win_rate=(wins / total * 100) if total > 0 else 0.0,
                total_pnl=float(stats["total_pnl"]),
                avg_pnl=float(stats["avg_pnl"]),
                best_trade=float(stats["best_trade"]),
                worst_trade=float(stats["worst_trade"]),
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                profit_factor=profit_factor,
            )

    async def get_strategy_performance(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> dict[str, PerformanceSummary]:
        """
        Get performance broken down by strategy.

        Args:
            start_date: Start of period.
            end_date: End of period.

        Returns:
            Dict mapping strategy name → PerformanceSummary.
        """
        self._require_connected()
        if self._pg_pool is None:
            raise RuntimeError("PostgreSQL pool is None")

        conditions = ["side = 'SELL'", "pnl != 0"]
        params: list[Any] = []
        param_idx = 1

        if start_date:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(start_date)
            param_idx += 1

        if end_date:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(end_date)
            param_idx += 1

        where = " AND ".join(conditions)

        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT strategy FROM trades WHERE {where}
                """,
                *params,
            )

        strategies = [row["strategy"] for row in rows]
        result: dict[str, PerformanceSummary] = {}

        for strat in strategies:
            result[strat] = await self.get_performance_summary(
                start_date=start_date, end_date=end_date, strategy=strat
            )

        return result

    async def get_trade_patterns(
        self, min_samples: int = 5
    ) -> dict[str, Any]:
        """
        Identify common trading patterns from historical data.

        Detects patterns like:
        - Symbol + side combinations with high/low win rates
        - Strategy performance by symbol
        - Time-of-day performance
        - Confidence vs. outcome correlation

        Args:
            min_samples: Minimum trades required for a pattern to be reported.

        Returns:
            Dict with pattern categories and their insights.
        """
        self._require_connected()
        if self._pg_pool is None:
            raise RuntimeError("PostgreSQL pool is None")

        patterns: dict[str, Any] = {
            "symbol_side_patterns": [],
            "strategy_symbol_patterns": [],
            "time_of_day_patterns": [],
            "confidence_correlation": {},
            "risk_action_outcomes": [],
        }

        async with self._pg_pool.acquire() as conn:
            # Pattern 1: Symbol + Side win rates
            rows = await conn.fetch(
                """
                SELECT symbol, side,
                       COUNT(*) as total,
                       COUNT(*) FILTER (WHERE pnl > 0) as wins,
                       COALESCE(SUM(pnl), 0) as total_pnl,
                       COALESCE(AVG(pnl), 0) as avg_pnl
                FROM trades
                WHERE side = 'SELL' AND pnl != 0
                GROUP BY symbol, side
                HAVING COUNT(*) >= $1
                ORDER BY total DESC
                """,
                min_samples,
            )

            for row in rows:
                win_rate = (int(row["wins"]) / int(row["total"]) * 100) if row["total"] > 0 else 0
                patterns["symbol_side_patterns"].append({
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "total_trades": int(row["total"]),
                    "win_rate": round(win_rate, 1),
                    "total_pnl": round(float(row["total_pnl"]), 2),
                    "avg_pnl": round(float(row["avg_pnl"]), 2),
                })

            # Pattern 2: Strategy performance by symbol
            strat_rows = await conn.fetch(
                """
                SELECT strategy, symbol,
                       COUNT(*) as total,
                       COUNT(*) FILTER (WHERE pnl > 0) as wins,
                       COALESCE(AVG(pnl), 0) as avg_pnl
                FROM trades
                WHERE side = 'SELL' AND pnl != 0
                GROUP BY strategy, symbol
                HAVING COUNT(*) >= $1
                ORDER BY total DESC
                """,
                min_samples,
            )

            for row in strat_rows:
                win_rate = (int(row["wins"]) / int(row["total"]) * 100) if row["total"] > 0 else 0
                patterns["strategy_symbol_patterns"].append({
                    "strategy": row["strategy"],
                    "symbol": row["symbol"],
                    "total_trades": int(row["total"]),
                    "win_rate": round(win_rate, 1),
                    "avg_pnl": round(float(row["avg_pnl"]), 2),
                })

            # Pattern 3: Time of day performance (UTC hour)
            time_rows = await conn.fetch(
                """
                SELECT EXTRACT(HOUR FROM timestamp) as hour,
                       COUNT(*) as total,
                       COUNT(*) FILTER (WHERE pnl > 0) as wins,
                       COALESCE(AVG(pnl), 0) as avg_pnl
                FROM trades
                WHERE side = 'SELL' AND pnl != 0
                GROUP BY hour
                HAVING COUNT(*) >= $1
                ORDER BY avg_pnl DESC
                """,
                max(min_samples, 3),
            )

            for row in time_rows:
                win_rate = (int(row["wins"]) / int(row["total"]) * 100) if row["total"] > 0 else 0
                patterns["time_of_day_patterns"].append({
                    "hour_utc": int(row["hour"]),
                    "total_trades": int(row["total"]),
                    "win_rate": round(win_rate, 1),
                    "avg_pnl": round(float(row["avg_pnl"]), 2),
                })

            # Pattern 4: AI confidence correlation
            conf_rows = await conn.fetch(
                """
                SELECT ai_confidence, pnl
                FROM trades
                WHERE ai_confidence IS NOT NULL AND side = 'SELL' AND pnl != 0
                ORDER BY timestamp
                """
            )

            if conf_rows:
                high_conf = [float(r["pnl"]) for r in conf_rows if float(r["ai_confidence"] or 0) >= 70]
                low_conf = [float(r["pnl"]) for r in conf_rows if float(r["ai_confidence"] or 0) < 70]

                patterns["confidence_correlation"] = {
                    "high_confidence_avg_pnl": round(
                        sum(high_conf) / len(high_conf) if high_conf else 0, 2
                    ),
                    "low_confidence_avg_pnl": round(
                        sum(low_conf) / len(low_conf) if low_conf else 0, 2
                    ),
                    "high_confidence_count": len(high_conf),
                    "low_confidence_count": len(low_conf),
                }

            # Pattern 5: Risk action outcomes
            # Note: judge_action stored in debate_result JSONB, need different approach
            # For now, use debates table joined approach
            debate_rows = await conn.fetch(
                """
                SELECT judge_action,
                       COUNT(*) as total,
                       COUNT(*) FILTER (WHERE pnl > 0) as wins,
                       COALESCE(AVG(pnl), 0) as avg_pnl
                FROM debates d
                JOIN trades t ON d.symbol = t.symbol
                    AND t.timestamp BETWEEN d.timestamp
                    AND d.timestamp + INTERVAL '1 hour'
                WHERE t.side = 'SELL' AND t.pnl != 0
                GROUP BY d.judge_action
                """
            )

            for row in debate_rows:
                win_rate = (int(row["wins"]) / int(row["total"]) * 100) if row["total"] > 0 else 0
                patterns["risk_action_outcomes"].append({
                    "judge_action": row["judge_action"],
                    "total_trades": int(row["total"]),
                    "win_rate": round(win_rate, 1),
                    "avg_pnl": round(float(row["avg_pnl"]), 2),
                })

        logger.info(f"[TradeMemory] Pattern analysis complete: {sum(len(v) if isinstance(v, list) else 1 for v in patterns.values())} patterns found")
        return patterns

