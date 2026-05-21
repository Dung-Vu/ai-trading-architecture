from types import SimpleNamespace

from src.debate.prompts import build_risk_manager_system_prompt


def test_risk_manager_prompt_uses_configured_limits():
    prompt = build_risk_manager_system_prompt(
        SimpleNamespace(
            max_daily_loss_pct=4.5,
            max_drawdown_pct=12.5,
            max_position_pct=25,
            max_leverage=4,
        )
    )

    assert "4.5% of total portfolio value" in prompt
    assert "12.5% from peak equity" in prompt
    assert "25% of portfolio" in prompt
    assert "4x" in prompt
    assert "3% of total portfolio value" not in prompt
