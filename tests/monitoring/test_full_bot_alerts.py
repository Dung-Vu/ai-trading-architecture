import pytest

from src.main_full import FullTradingBot


class _Config:
    class Trading:
        initial_capital = 10000.0

    class Monitoring:
        telegram_enabled = True
        telegram_bot_token = "dummy-token"
        telegram_chat_id = "123"

    trading = Trading()
    monitoring = Monitoring()


@pytest.mark.asyncio
async def test_full_bot_send_alert_uses_formatter_and_send_alert(monkeypatch):
    sent_alerts: list[str] = []

    class _Bot:
        def __init__(self, bot_token: str, chat_id: str) -> None:
            assert bot_token == "dummy-token"
            assert chat_id == "123"

        async def send_alert(self, message: str) -> None:
            sent_alerts.append(message)

    monkeypatch.setattr("src.monitoring.telegram_bot.TelegramBot", _Bot)

    bot = FullTradingBot(config=_Config(), symbols=["BTC/USDT"])
    await bot._send_alert(
        "BTC/USDT",
        {"side": "BUY", "quantity": 0.01, "price": 67000.0, "strategy": "ai_debate"},
        {"action": "BUY"},
    )

    assert len(sent_alerts) == 1
    assert "BTC/USDT" in sent_alerts[0]
    assert "BUY" in sent_alerts[0]
