import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import src.main as app_main


def _make_config(mode: str = "dryrun") -> SimpleNamespace:
    return SimpleNamespace(
        trading=SimpleNamespace(mode=mode, symbols=["BTC/USDT"]),
        strategy=SimpleNamespace(name="sma_cross"),
        monitoring=SimpleNamespace(
            log_level="INFO",
            telegram_bot_token="token",
            telegram_chat_id="chat",
        ),
    )


def _make_args(**overrides: object) -> SimpleNamespace:
    args = {
        "mode": "dryrun",
        "strategy": "sma_cross",
        "symbols": ["BTC/USDT"],
        "config": None,
        "log_level": "INFO",
        "initial_capital": 10000.0,
        "backtest": False,
        "backtest_days": 90,
        "data_pipeline": False,
        "monitor": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_parse_args_accepts_runtime_and_backtest_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--mode",
            "testnet",
            "--strategy",
            "bbands",
            "--symbols",
            "BTC/USDT",
            "ETH/USDT",
            "--backtest",
            "--backtest-days",
            "21",
            "--log-level",
            "DEBUG",
            "--initial-capital",
            "25000",
        ],
    )

    args = app_main.parse_args()

    assert args.mode == "testnet"
    assert args.strategy == "bbands"
    assert args.symbols == ["BTC/USDT", "ETH/USDT"]
    assert args.backtest is True
    assert args.backtest_days == 21
    assert args.log_level == "DEBUG"
    assert args.initial_capital == 25000.0


@pytest.mark.parametrize(
    ("args", "config_mode", "expected_target"),
    [
        (_make_args(data_pipeline=True), "dryrun", "run_data_pipeline"),
        (_make_args(backtest=True, backtest_days=45), "dryrun", "run_backtest"),
        (_make_args(monitor=True), "dryrun", "run_monitor"),
        (_make_args(), "dryrun", "run_dry_run"),
        (_make_args(mode="testnet"), "testnet", "run_trading_bot"),
    ],
)
@patch("src.main.setup_logging")
@patch("src.main.load_runtime_config")
@patch("src.main.parse_args")
def test_main_routes_to_expected_entrypoint(
    mock_parse_args,
    mock_load_runtime_config,
    mock_setup_logging,
    args,
    config_mode,
    expected_target,
):
    config = _make_config(mode=config_mode)
    mock_parse_args.return_value = args
    mock_load_runtime_config.return_value = config

    with patch("src.main.run_data_pipeline") as mock_data_pipeline, patch(
        "src.main.run_backtest"
    ) as mock_backtest, patch("src.main.run_monitor") as mock_monitor, patch(
        "src.main.run_dry_run"
    ) as mock_dry_run, patch("src.main.run_trading_bot") as mock_trading_bot:
        app_main.main()

    called_targets = {
        "run_data_pipeline": mock_data_pipeline,
        "run_backtest": mock_backtest,
        "run_monitor": mock_monitor,
        "run_dry_run": mock_dry_run,
        "run_trading_bot": mock_trading_bot,
    }
    for name, mock in called_targets.items():
        if name == expected_target:
            mock.assert_called_once_with(config)
        else:
            mock.assert_not_called()

    if expected_target == "run_backtest":
        assert config.backtest_days == 45

    mock_setup_logging.assert_called_once_with(log_level="INFO")


@patch("src.main.setup_logging")
@patch("src.main.load_runtime_config")
@patch("src.main.parse_args")
def test_main_live_mode_warns_before_starting_trading_bot(
    mock_parse_args,
    mock_load_runtime_config,
    mock_setup_logging,
):
    config = _make_config(mode="live")
    mock_parse_args.return_value = _make_args(mode="live")
    mock_load_runtime_config.return_value = config

    with patch("src.main.run_trading_bot") as mock_run_trading_bot, patch(
        "src.main.logger.warning"
    ) as mock_warning:
        app_main.main()

    mock_run_trading_bot.assert_called_once_with(config)
    assert mock_warning.call_count == 2
    mock_setup_logging.assert_called_once_with(log_level="INFO")