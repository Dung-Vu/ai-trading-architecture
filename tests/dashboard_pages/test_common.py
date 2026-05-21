from unittest.mock import patch

import pandas as pd

from src.dashboard_pages.common import (
    _build_live_equity_curve,
    _flatten_trade_patterns,
    _should_use_mock_dashboard_data,
    load_dashboard_data,
)


def test_load_dashboard_data_returns_mock_when_forced_by_env(monkeypatch):
    marker = {"source": "mock"}
    monkeypatch.setenv("DASHBOARD_USE_MOCK_DATA", "true")

    with patch("src.dashboard_pages.common.load_mock_data", return_value=marker):
        assert load_dashboard_data() == marker


def test_load_dashboard_data_uses_live_loader_when_history_exists(monkeypatch):
    monkeypatch.delenv("DASHBOARD_USE_MOCK_DATA", raising=False)
    live_payload = {
        "trades": [{"timestamp": "2026-05-20T00:00:00Z"}],
        "debates": [{"timestamp": "2026-05-20T00:00:00Z"}],
        "equity_curve": pd.Series([10000.0]),
        "patterns": [],
    }

    with patch(
        "src.dashboard_pages.common._load_live_dashboard_data",
        return_value=live_payload,
    ), patch("src.dashboard_pages.common.load_mock_data") as mock_loader:
        assert load_dashboard_data() == live_payload

    mock_loader.assert_not_called()


def test_load_dashboard_data_falls_back_to_mock_when_live_loader_fails(monkeypatch):
    marker = {"source": "mock"}
    monkeypatch.delenv("DASHBOARD_USE_MOCK_DATA", raising=False)

    with patch(
        "src.dashboard_pages.common._load_live_dashboard_data",
        side_effect=RuntimeError("db offline"),
    ), patch("src.dashboard_pages.common.load_mock_data", return_value=marker):
        assert load_dashboard_data() == marker


def test_build_live_equity_curve_prefers_cash_snapshots_over_raw_pnl():
    curve = _build_live_equity_curve([
        {
            "timestamp": "2026-05-20T00:00:00Z",
            "cash_remaining": 9000.0,
            "pnl": -50.0,
        },
        {
            "timestamp": "2026-05-21T00:00:00Z",
            "cash_total": 10150.0,
            "pnl": 150.0,
        },
        {
            "timestamp": "2026-05-22T00:00:00Z",
            "pnl": -25.0,
        },
    ])

    assert list(curve.values) == [9000.0, 10150.0, 10125.0]
    assert str(curve.index[0].tz) == "UTC"


def test_flatten_trade_patterns_returns_dashboard_shape():
    flattened = _flatten_trade_patterns({
        "symbol_side_patterns": [
            {
                "symbol": "BTC/USDT",
                "side": "SELL",
                "total_trades": 8,
                "win_rate": 62.5,
                "avg_pnl": 145.2,
            }
        ],
        "confidence_correlation": {
            "high_confidence_count": 5,
            "low_confidence_count": 3,
            "high_confidence_avg_pnl": 120.0,
            "low_confidence_avg_pnl": -30.0,
        },
    })

    assert flattened[0]["name"] == "BTC/USDT SELL"
    assert flattened[0]["frequency"] == 8
    assert "Win rate 62.5%" in flattened[0]["description"]
    assert flattened[1]["name"] == "Confidence bands"
    assert flattened[1]["frequency"] == 8


def test_should_use_mock_dashboard_data_parses_common_truthy_values(monkeypatch):
    monkeypatch.setenv("DASHBOARD_USE_MOCK_DATA", "yes")

    assert _should_use_mock_dashboard_data() is True
    assert _should_use_mock_dashboard_data(use_mock=False) is False