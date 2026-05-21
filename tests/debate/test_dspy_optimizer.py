import importlib
import sys
import types
from types import SimpleNamespace

import pytest


def _install_fake_dspy(monkeypatch):
    fake_dspy = types.ModuleType("dspy")
    fake_dspy.compile_calls = []

    class Signature:
        pass

    def input_field(**kwargs):
        return kwargs

    def output_field(**kwargs):
        return kwargs

    class Module:
        pass

    class LM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def configure(**kwargs):
        fake_dspy.configured = kwargs

    class Prediction(SimpleNamespace):
        pass

    class ChainOfThought:
        def __init__(self, signature):
            self.signature = signature
            self.demos = []

        def __call__(self, market_data):
            del market_data
            return Prediction(
                action="BUY",
                confidence="75",
                reason="Optimized by fake DSPy",
                stop_loss="95",
                take_profit="110",
            )

    class Example(SimpleNamespace):
        def with_inputs(self, *_args):
            return self

    class MIPROv2:
        def __init__(
            self,
            metric,
            num_trials,
            max_bootstrapped_demos,
            max_labeled_demos,
        ):
            self.metric = metric
            self.num_trials = num_trials
            self.max_bootstrapped_demos = max_bootstrapped_demos
            self.max_labeled_demos = max_labeled_demos

        def compile(self, program, trainset, num_threads, progress_bar):
            fake_dspy.compile_calls.append({
                "program": program,
                "trainset_size": len(trainset),
                "num_threads": num_threads,
                "progress_bar": progress_bar,
            })
            return program

    fake_dspy.Signature = Signature
    fake_dspy.InputField = input_field
    fake_dspy.OutputField = output_field
    fake_dspy.Module = Module
    fake_dspy.LM = LM
    fake_dspy.configure = configure
    fake_dspy.Prediction = Prediction
    fake_dspy.ChainOfThought = ChainOfThought
    fake_dspy.Example = Example
    fake_dspy.MIPROv2 = MIPROv2

    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)
    sys.modules.pop("src.debate.optimizer", None)
    module = importlib.import_module("src.debate.optimizer")
    return module, fake_dspy


@pytest.mark.asyncio
async def test_dspy_optimizer_runs_miprov2_with_prepared_demos(monkeypatch):
    optimizer_module, fake_dspy = _install_fake_dspy(monkeypatch)
    optimizer = optimizer_module.DSPyOptimizer(
        trade_memory=SimpleNamespace(),
        debate_config=SimpleNamespace(),
        llm_model="openai/gpt-4o-mini",
    )

    optimized_program = await optimizer.optimize(
        demo_trades=[
            {
                "market_data_text": "BTC oversold bounce",
                "action": "BUY",
                "pnl": 120.0,
                "was_profitable": True,
            },
            {
                "market_data_text": "BTC weak momentum",
                "action": "HOLD",
                "pnl": -40.0,
                "was_profitable": False,
            },
            {
                "market_data_text": "ETH breakout continuation",
                "action": "BUY",
                "pnl": 80.0,
                "was_profitable": True,
            },
            {
                "market_data_text": "SOL bearish reversal",
                "action": "SELL",
                "pnl": 60.0,
                "was_profitable": True,
            },
        ],
        metric="pnl_weighted",
        num_trials=3,
        max_bootstrapped_demos=2,
    )

    assert optimized_program is optimizer._program
    assert fake_dspy.configured["lm"].kwargs["model"] == "openai/gpt-4o-mini"
    assert fake_dspy.compile_calls[0]["trainset_size"] == 3
    assert fake_dspy.compile_calls[0]["num_threads"] == 1
    assert fake_dspy.compile_calls[0]["progress_bar"] is True
