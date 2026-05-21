from types import SimpleNamespace
from unittest.mock import patch

from src.main_full import main, run_backtest
from src.runtime_helpers import summarize_backtest_results
from src.strategy.bbands import BBandsStrategy


class MockBacktestConfig:
    def __init__(self):
        self.trading = SimpleNamespace(
            mode="dryrun",
            initial_capital=5000.0,
            symbols=["ETH/USDT"],
        )
        self.monitoring = SimpleNamespace(log_level="INFO")
        self.strategy = SimpleNamespace(
            name="bbands",
            sma_fast=10,
            sma_slow=30,
            rsi_period=14,
        )
        self.litellm_model = "anthropic/claude-sonnet-4"


@patch("src.main_full.run_backtest")
@patch("src.main_full.setup_logging")
@patch("src.config.load_config")
@patch("src.main_full.parse_args")
def test_main_backtest_preserves_config_strategy(
    mock_parse_args,
    mock_load_config,
    mock_setup_logging,
    mock_run_backtest,
):
    args = SimpleNamespace(
        mode="dryrun",
        strategy=None,
        symbols=None,
        config=None,
        log_level="INFO",
        initial_capital=5000.0,
        interval=60,
        no_memory=False,
        no_news=False,
        no_autotune=False,
        debate_only=False,
        backtest=True,
        backtest_days=30,
    )
    config = MockBacktestConfig()

    mock_parse_args.return_value = args
    mock_load_config.return_value = config

    main()

    passed_config, passed_args = mock_run_backtest.call_args.args
    assert passed_args.backtest_days == 30
    assert passed_config.strategy.name == "bbands"
    mock_setup_logging.assert_called_once_with(
        log_level="INFO",
        app_log_name="full_trading",
        error_log_name="full_error",
    )


@patch("src.strategy.backtest.BacktestRunner")
def test_run_backtest_uses_requested_days_and_strategy(mock_runner_class):
    config = MockBacktestConfig()
    args = SimpleNamespace(backtest_days=21)

    mock_runner = mock_runner_class.return_value
    mock_runner.run.return_value = None

    run_backtest(config, args)

    kwargs = mock_runner_class.call_args.kwargs
    assert kwargs["strategy_class"] is BBandsStrategy
    assert kwargs["parameters"] == {"symbol": "ETH/USDT"}
    assert (kwargs["end_date"] - kwargs["start_date"]).days == 21


def test_summarize_backtest_results_handles_missing_equity_curve():
    metrics = summarize_backtest_results({"trades": [{"pnl": 10.0}]})

    assert metrics["total_trades"] == 1
    assert metrics["start_value"] == 0.0
    assert metrics["end_value"] == 0.0
