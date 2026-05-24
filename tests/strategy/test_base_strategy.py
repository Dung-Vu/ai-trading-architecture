import importlib
import os
from pathlib import Path
import sys
import threading
import types


def _load_base_module(monkeypatch):
    strategy_pkg = types.ModuleType("src.strategy")
    strategy_pkg.__path__ = [str(Path(__file__).parents[2] / "src" / "strategy")]
    monkeypatch.setitem(sys.modules, "src.strategy", strategy_pkg)

    fake_logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "loguru", types.SimpleNamespace(logger=fake_logger))
    monkeypatch.setitem(
        sys.modules,
        "pandas",
        types.SimpleNamespace(DataFrame=object, Series=object),
    )

    asset_type = type("AssetType", (), {"CRYPTO": "crypto"})
    asset_cls = type("Asset", (), {"AssetType": asset_type})
    monkeypatch.setitem(
        sys.modules,
        "lumibot.entities",
        types.SimpleNamespace(
            Asset=asset_cls,
            Order=object,
        ),
    )
    strategy_base = type("Strategy", (), {"__init__": lambda self, *args, **kwargs: None})
    monkeypatch.setitem(
        sys.modules,
        "lumibot.strategies",
        types.SimpleNamespace(Strategy=strategy_base),
    )

    monkeypatch.setitem(
        sys.modules,
        "src.strategy.indicator_utils",
        types.SimpleNamespace(calculate_indicator=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.strategy.mock_runtime",
        types.SimpleNamespace(ensure_strategy_runtime_kwargs=lambda kwargs: kwargs),
    )

    sys.modules.pop("src.strategy.base", None)
    return importlib.import_module("src.strategy.base")


def test_base_strategy_serializes_is_backtesting_env_mutation(monkeypatch):
    module = _load_base_module(monkeypatch)
    first_entered = threading.Event()
    allow_first_exit = threading.Event()
    second_entered = threading.Event()
    call_index = {"value": 0}

    def fake_strategy_init(self, *args, **kwargs):
        del self, args, kwargs
        call_index["value"] += 1
        assert os.environ.get("IS_BACKTESTING") == "true"

        if call_index["value"] == 1:
            first_entered.set()
            assert allow_first_exit.wait(timeout=1.0)
        else:
            second_entered.set()

    module.Strategy.__init__ = fake_strategy_init

    class DummyStrategy(module.BaseStrategy):
        def on_trading_iteration(self) -> None:
            return None

    os.environ.pop("IS_BACKTESTING", None)

    errors = []

    def instantiate_strategy():
        try:
            DummyStrategy()
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=instantiate_strategy)
    second = threading.Thread(target=instantiate_strategy)

    first.start()
    assert first_entered.wait(timeout=1.0)

    second.start()
    assert not second_entered.wait(timeout=0.2)

    allow_first_exit.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not errors
    assert second_entered.is_set()
    assert os.environ.get("IS_BACKTESTING") is None
