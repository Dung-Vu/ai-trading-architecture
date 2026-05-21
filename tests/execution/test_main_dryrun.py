from types import SimpleNamespace
from unittest.mock import patch

from src.main import run_dry_run


def test_run_dry_run_delegates_to_full_trading_bot_loop():
    config = SimpleNamespace(trading=SimpleNamespace(mode="testnet"))

    with patch("src.main.run_trading_bot") as mock_run_trading_bot:
        run_dry_run(config)

    assert config.trading.mode == "dryrun"
    mock_run_trading_bot.assert_called_once_with(config)
