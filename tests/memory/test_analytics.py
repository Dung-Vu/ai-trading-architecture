import asyncio

import src.memory.analytics as analytics_module


class _FakeMetricsCalculator:
    @staticmethod
    def calc_trade_pnl_max_drawdown(_pnls):
        return 0.1

    @staticmethod
    def calc_trade_pnl_sharpe_ratio(_pnls, risk_free_rate=0.0, periods_per_year=250):
        return 1.234


analytics_module.MetricsCalculator = _FakeMetricsCalculator
TradeMemoryAnalyticsMixin = analytics_module.TradeMemoryAnalyticsMixin


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


class _AnalyticsHarness(TradeMemoryAnalyticsMixin):
    def __init__(self, conn):
        self._pg_pool = _FakePool(conn)

    def _require_connected(self):
        return None


class _SummaryConn:
    def __init__(self):
        self.fetchrow_queries = []

    async def fetchrow(self, query, *params):
        self.fetchrow_queries.append((query, params))
        return {
            "total_trades": 3,
            "winning_trades": 1,
            "losing_trades": 2,
            "total_pnl": -10.0,
            "avg_pnl": -3.3333,
            "best_trade": 15.0,
            "worst_trade": -20.0,
            "gross_profit": 15.0,
            "gross_loss": 25.0,
        }

    async def fetch(self, query, *params):
        assert "ORDER BY timestamp" in query
        return [{"pnl": 15.0}, {"pnl": -5.0}, {"pnl": -20.0}]


class _PatternsConn:
    def __init__(self):
        self.queries = []

    async def fetch(self, query, *params):
        self.queries.append((query, params))

        if "FROM trades" in query and "judge_action" in query:
            raise AssertionError("analytics should not query judge_action from trades")

        if "GROUP BY symbol, side" in query:
            return [
                {
                    "symbol": "BTC/USDT",
                    "side": "SELL",
                    "total": 6,
                    "wins": 4,
                    "total_pnl": 120.0,
                    "avg_pnl": 20.0,
                }
            ]

        if "GROUP BY strategy, symbol" in query:
            return [
                {
                    "strategy": "sma_cross",
                    "symbol": "BTC/USDT",
                    "total": 6,
                    "wins": 4,
                    "avg_pnl": 20.0,
                }
            ]

        if "GROUP BY hour" in query:
            return [{"hour": 9, "total": 5, "wins": 3, "avg_pnl": 12.5}]

        if "SELECT ai_confidence, pnl" in query:
            return [
                {"ai_confidence": 80.0, "pnl": 20.0},
                {"ai_confidence": 65.0, "pnl": -5.0},
            ]

        if "FROM debates d" in query:
            return [{"judge_action": "BUY", "total": 4, "wins": 3, "avg_pnl": 14.0}]

        raise AssertionError(f"Unexpected query: {query}")


def test_get_performance_summary_uses_min_for_worst_trade():
    async def run_test():
        conn = _SummaryConn()
        harness = _AnalyticsHarness(conn)

        summary = await harness.get_performance_summary()

        assert summary.worst_trade == -20.0
        query, _params = conn.fetchrow_queries[0]
        assert "COALESCE(MIN(pnl) FILTER (WHERE pnl < 0), 0) as worst_trade" in query

    asyncio.run(run_test())


def test_get_trade_patterns_joins_debates_for_judge_action():
    async def run_test():
        conn = _PatternsConn()
        harness = _AnalyticsHarness(conn)

        patterns = await harness.get_trade_patterns(min_samples=4)

        assert patterns["risk_action_outcomes"] == [
            {
                "judge_action": "BUY",
                "total_trades": 4,
                "win_rate": 75.0,
                "avg_pnl": 14.0,
            }
        ]
        assert any("FROM debates d" in query for query, _params in conn.queries)
        assert all(
            not ("FROM trades" in query and "judge_action" in query)
            for query, _params in conn.queries
        )

    asyncio.run(run_test())