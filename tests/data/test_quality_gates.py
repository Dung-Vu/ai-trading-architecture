"""
Unit tests for QualityGates module.
"""

import time
import pytest
from src.data.quality_gates import QualityGates


class TestQualityGates:
    @pytest.fixture
    def gates(self):
        return QualityGates(max_latency_ms=5000, z_score_threshold=3.0, max_spread_pct=1.0)

    def test_check_latency_pass(self, gates):
        recent_ts = time.time() - 1  # 1 second ago
        assert gates.check_latency(recent_ts) is True

    def test_check_latency_fail(self, gates):
        old_ts = time.time() - 10  # 10 seconds ago
        assert gates.check_latency(old_ts) is False

    def test_check_spread_pass(self, gates):
        assert gates.check_spread(100.0, 100.5) is True  # 0.5% spread

    def test_check_spread_fail(self, gates):
        assert gates.check_spread(100.0, 105.0) is False  # 5% spread

    def test_check_price_spike_normal(self, gates):
        prices = [100.0, 100.5, 101.0, 100.8, 100.2, 101.5, 100.9]
        assert gates.check_price_spike(101.2, prices) is True

    def test_check_price_spike_anomaly(self, gates):
        prices = [100.0] * 10
        assert gates.check_price_spike(200.0, prices) is False

    def test_validate_trade_pass(self, gates):
        trade = {"symbol": "BTC-USDT", "price": 50000.0, "side": "buy"}
        recent = [50000.0, 50100.0, 49900.0]
        approved, reason = gates.validate_trade(trade, time.time() - 1, recent)
        assert approved is True

    def test_validate_trade_latency_fail(self, gates):
        trade = {"symbol": "BTC-USDT", "price": 50000.0, "side": "buy"}
        recent = [50000.0, 50100.0]
        approved, reason = gates.validate_trade(trade, time.time() - 10, recent)
        assert approved is False
        assert "latency" in reason.lower()

    def test_validate_trade_spread_fail(self, gates):
        trade = {"symbol": "BTC-USDT", "price": 50000.0, "side": "buy", "bid": 50000.0, "ask": 55000.0}
        recent = [50000.0]
        approved, reason = gates.validate_trade(trade, time.time() - 1, recent)
        assert approved is False
        assert "spread" in reason.lower()
