from pathlib import Path
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.autotune import AutoTuner


def _sample_trades(count: int) -> list[dict]:
    return [
        {
            "symbol": "BTC/USDT",
            "side": "BUY" if idx % 2 == 0 else "SELL",
            "pnl": 50.0 if idx % 2 == 0 else -20.0,
            "indicators": {"rsi": 25 if idx % 2 == 0 else 75},
            "market_conditions": {"volume_high": True},
        }
        for idx in range(count)
    ]


@pytest.mark.asyncio
async def test_weekly_optimization_cycle_runs_optimizer_and_persists_results(tmp_path):
    tuner = AutoTuner(
        trade_memory=None,
        debate_config={"temperature": 0.7},
        strategy_name="sma_cross",
        config_dir=str(tmp_path),
    )
    optimized_path = tmp_path / "optimized" / "sma_cross.json"

    tuner._fetch_recent_trades = AsyncMock(return_value=_sample_trades(20))
    tuner._run_optimizer = MagicMock(return_value={
        "new_params": {"sma_fast": 10, "sma_slow": 30},
        "previous_score": 0.5,
        "expected_score": 1.1,
        "improvement": 0.6,
    })
    tuner._update_debate_prompts = MagicMock()
    tuner._save_optimized_config = MagicMock(return_value=optimized_path)
    tuner._generate_comparison = MagicMock(return_value={"summary": "improved"})

    report = await tuner.weekly_optimization_cycle()

    assert report["status"] == "completed"
    assert report["config_path"] == str(optimized_path)
    assert report["comparison"] == {"summary": "improved"}
    assert report["steps_completed"] == [
        "fetch_trades",
        "calculate_metrics",
        "run_optimizer",
        "update_prompts",
        "save_config",
        "generate_report",
    ]
    tuner._run_optimizer.assert_called_once()
    tuner._update_debate_prompts.assert_called_once()
    tuner._save_optimized_config.assert_called_once()


def test_detect_strategy_decay_triggers_on_win_rate_drop_threshold(tmp_path):
    async def run_test():
        tuner = AutoTuner(
            trade_memory=None,
            debate_config={"temperature": 0.7},
            strategy_name="sma_cross",
            config_dir=str(tmp_path),
        )

        tuner._fetch_recent_trades = AsyncMock(return_value=_sample_trades(30))
        tuner._calculate_metrics = MagicMock(
            side_effect=[
                {"sharpe_ratio": 1.0, "win_rate": 0.30, "max_drawdown": 0.05},
                {"sharpe_ratio": 1.0, "win_rate": 0.80, "max_drawdown": 0.05},
            ]
        )

        assert await tuner.detect_strategy_decay() is True

    asyncio.run(run_test())


def test_calculate_metrics_reports_losing_trades_key(tmp_path):
    tuner = AutoTuner(
        trade_memory=None,
        debate_config={"temperature": 0.7},
        strategy_name="sma_cross",
        config_dir=str(tmp_path),
    )

    metrics = tuner._calculate_metrics(_sample_trades(4))

    assert metrics["winning_trades"] == 2
    assert metrics["losing_trades"] == 2
    assert "losing_trads" not in metrics
