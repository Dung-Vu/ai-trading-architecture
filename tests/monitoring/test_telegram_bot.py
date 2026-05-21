from unittest.mock import AsyncMock, patch

import pytest
from telegram.constants import ParseMode

from src.monitoring.telegram_bot import TelegramBot


@pytest.mark.asyncio
@patch("src.monitoring.telegram_bot.Bot")
async def test_send_alert_works_without_polling(mock_bot_class):
    bot_client = AsyncMock()
    mock_bot_class.return_value = bot_client

    bot = TelegramBot(bot_token="token", chat_id="chat")
    await bot.send_alert("hello")

    mock_bot_class.assert_called_once_with(token="token")
    bot_client.send_message.assert_awaited_once_with(
        chat_id="chat",
        text="hello",
        parse_mode=ParseMode.HTML,
    )