"""
Unit tests for KillSwitch module.
"""

import pytest
from datetime import datetime
from src.risk.kill_switch import KillSwitch


class TestKillSwitch:
    def test_initial_state(self):
        ks = KillSwitch()
        assert ks.is_active() is False

    def test_trigger(self):
        ks = KillSwitch()
        ks.trigger("Manual trigger")
        assert ks.is_active() is True

        trigger = ks.get_last_trigger()
        assert trigger["reason"] == "Manual trigger"

    def test_disarm(self):
        ks = KillSwitch()
        ks.trigger("Test trigger")
        ks.disarm()
        assert ks.is_active() is False

    def test_trigger_history(self):
        ks = KillSwitch()
        ks.trigger("Trigger 1")
        ks.disarm()
        ks.trigger("Trigger 2")

        history = ks.get_trigger_history()
        assert len(history) == 2
        assert history[0]["reason"] == "Trigger 1"
        assert history[1]["reason"] == "Trigger 2"

    def test_auto_check_triggered(self):
        ks = KillSwitch()
        ks.auto_check(max_drawdown_pct=10.0, current_drawdown=11.0)
        assert ks.is_active() is True

    def test_auto_check_not_triggered(self):
        ks = KillSwitch()
        ks.auto_check(max_drawdown_pct=10.0, current_drawdown=5.0)
        assert ks.is_active() is False
