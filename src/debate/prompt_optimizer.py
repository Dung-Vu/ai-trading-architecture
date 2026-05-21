"""DSPy prompt optimization workflow for the AI Debate Engine."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.debate.dspy_infra import (
    MarketInput,
    dspy,
    pnl_weighted_metric,
    sharpe_metric,
)
from src.config import get_default_dspy_model
from src.memory.interfaces import TradeMemoryInterface


class DSPyOptimizer:
    """
    DSPy-based prompt optimizer for the AI Debate Engine.

    Uses historical trade data as demonstrations to automatically optimize the
    debate engine prompts for better trading performance.
    """

    def __init__(
        self,
        trade_memory: TradeMemoryInterface,
        debate_config: Any,
        llm_model: str | None = None,
    ) -> None:
        if dspy is None:
            raise ImportError("dspy-ai is required. Install: pip install dspy-ai")

        self._memory = trade_memory
        self._debate_config = debate_config
        self._llm_model = llm_model or get_default_dspy_model()
        self._program: dspy.Module | None = None
        self._optimized_program: dspy.Module | None = None

    def setup_program(self) -> dspy.Module:
        """Create a DSPy program for the debate engine."""
        logger.info(f"[DSPyOptimizer] Setting up DSPy program with {self._llm_model}")

        lm = dspy.LM(
            model=self._llm_model,
            temperature=0.7,
            max_tokens=2048,
        )
        dspy.configure(lm=lm)

        class DebateProgram(dspy.Module):
            """DSPy program for trading debate decisions."""

            def __init__(self) -> None:
                super().__init__()
                self.predictor = dspy.ChainOfThought(MarketInput)

            def forward(self, market_data: str) -> dspy.Prediction:
                return self.predictor(market_data=market_data)

        self._program = DebateProgram()
        logger.info("[DSPyOptimizer] DSPy program created with ChainOfThought predictor")
        return self._program

    async def _prepare_demonstrations(
        self,
        demo_trades: list[dict[str, Any]] | None = None,
        max_demos: int = 50,
    ) -> list[dspy.Example]:
        """Prepare DSPy examples from explicit demos or trade memory history."""
        demonstrations: list[dict[str, Any]] = []

        if demo_trades:
            demonstrations = demo_trades[:max_demos]
        else:
            try:
                history = await self._memory.get_trade_history(limit=max_demos * 4)
                history = list(reversed(history))
                open_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)

                for trade in history:
                    side = str(trade.get("side", "")).upper()
                    symbol = trade.get("symbol", "")

                    if side == "BUY":
                        open_entries[symbol].append(trade)
                        continue

                    if side != "SELL" or trade.get("pnl", 0) == 0:
                        continue

                    pnl = trade.get("pnl", 0)
                    debate_result = _parse_debate_result(trade.get("debate_result") or {})
                    market_text = self._format_market_data_for_demo(
                        trade, debate_result
                    )
                    correct_action = trade.get("side", "SELL") if pnl > 0 else "HOLD"

                    demonstrations.append(
                        {
                            "market_data_text": market_text,
                            "action": correct_action,
                            "pnl": pnl,
                            "was_profitable": trade.get("pnl", 0) > 0,
                        }
                    )

                    if len(demonstrations) >= max_demos:
                        break

                    if open_entries[symbol]:
                        entry_trade = open_entries[symbol].pop(0)
                        entry_debate = _parse_debate_result(
                            entry_trade.get("debate_result") or {}
                        )
                        entry_text = self._format_market_data_for_demo(
                            entry_trade,
                            entry_debate,
                        )
                        demonstrations.append(
                            {
                                "market_data_text": entry_text,
                                "action": "BUY" if pnl > 0 else "HOLD",
                                "pnl": pnl,
                                "was_profitable": pnl > 0,
                            }
                        )

                    if len(demonstrations) >= max_demos:
                        break

            except Exception as exc:
                logger.warning(f"Failed to fetch trade history for demos: {exc}")
                return []

        examples = []
        for demo in demonstrations:
            example = dspy.Example(
                market_data=demo["market_data_text"],
                action=demo["action"],
                confidence="75" if demo["was_profitable"] else "30",
                reason="Optimized by DSPy",
                stop_loss=str(demo.get("stop_loss", 0)),
                take_profit=str(demo.get("take_profit", 0)),
            ).with_inputs("market_data")
            examples.append(example)

        logger.info(f"[DSPyOptimizer] Prepared {len(examples)} demonstrations")
        return examples

    @staticmethod
    def _format_market_data_for_demo(
        trade: dict[str, Any], debate_result: dict[str, Any]
    ) -> str:
        """Format a trade into market data text for DSPy demonstrations."""
        parts = [
            f"Symbol: {trade.get('symbol', 'N/A')}",
            f"Price: ${trade.get('price', 0):,.2f}",
            f"Strategy: {trade.get('strategy', 'N/A')}",
        ]

        if trade.get("ai_confidence"):
            parts.append(f"AI Confidence: {trade['ai_confidence']:.1f}")
        if trade.get("stop_loss"):
            parts.append(f"Stop Loss: ${trade['stop_loss']:,.2f}")
        if trade.get("take_profit"):
            parts.append(f"Take Profit: ${trade['take_profit']:,.2f}")

        if debate_result:
            bull = debate_result.get("bull_arg", "")
            bear = debate_result.get("bear_arg", "")
            if bull:
                parts.append(f"Bull case: {bull[:200]}")
            if bear:
                parts.append(f"Bear case: {bear[:200]}")

        return " | ".join(parts)

    async def optimize(
        self,
        demo_trades: list[dict[str, Any]] | None = None,
        metric: str = "sharpe_ratio",
        num_trials: int = 10,
        max_bootstrapped_demos: int = 5,
    ) -> dspy.Module:
        """Run MIPROv2 optimization on the DSPy program."""
        if self._program is None:
            self.setup_program()

        logger.info(
            f"[DSPyOptimizer] Starting optimization: metric={metric}, trials={num_trials}"
        )
        examples = await self._prepare_demonstrations(demo_trades)

        if len(examples) < 3:
            logger.warning(
                f"[DSPyOptimizer] Only {len(examples)} demonstrations available. "
                "Need at least 3 for meaningful optimization."
            )
            return self._program

        metric_fn = pnl_weighted_metric if metric == "pnl_weighted" else sharpe_metric

        train_size = max(3, int(len(examples) * 0.7))
        trainset = examples[:train_size]
        evalset = examples[train_size:] or trainset
        logger.info(
            f"[DSPyOptimizer] Train set: {len(trainset)}, Eval set: {len(evalset)}"
        )

        try:
            optimizer = dspy.MIPROv2(
                metric=metric_fn,
                num_trials=num_trials,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=10,
            )
            optimized = optimizer.compile(
                self._program,
                trainset=trainset,
                num_threads=1,
                progress_bar=True,
            )
            self._optimized_program = optimized
            logger.info("[DSPyOptimizer] Optimization complete")
        except Exception as exc:
            logger.warning(f"[DSPyOptimizer] MIPROv2 optimization failed: {exc}")
            logger.info("[DSPyOptimizer] Using unoptimized program as fallback")
            self._optimized_program = self._program

        return self._optimized_program

    def save_optimized_prompts(
        self,
        program: dspy.Module | None = None,
        path: str | None = None,
    ) -> Path:
        """Save optimized prompts to disk."""
        prog = program or self._optimized_program or self._program
        if prog is None:
            raise RuntimeError("No program to save. Call setup_program() first.")

        if path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            dirpath = Path("config/optimized_prompts")
            dirpath.mkdir(parents=True, exist_ok=True)
            filepath = dirpath / f"dspy_prompts_{timestamp}.json"
        else:
            filepath = Path(path)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            program_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "llm_model": self._llm_model,
                "optimized": prog is self._optimized_program,
                "predictor_type": "ChainOfThought",
                "signature": "MarketInput",
                "prompt_state": self._extract_prompt_state(prog),
            }
            filepath.write_text(json.dumps(program_data, indent=2), encoding="utf-8")
            logger.info(f"[DSPyOptimizer] Prompts saved to {filepath}")
        except Exception as exc:
            logger.warning(f"[DSPyOptimizer] Failed to save prompts: {exc}")
            fallback_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "llm_model": self._llm_model,
                "optimized": prog is self._optimized_program,
                "error": str(exc),
            }
            filepath.write_text(json.dumps(fallback_data, indent=2), encoding="utf-8")

        return filepath

    def load_optimized_prompts(self, path: str) -> dspy.Module:
        """Load previously optimized prompts and reinitialize the program."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Prompts file not found: {filepath}")

        data = json.loads(filepath.read_text(encoding="utf-8"))
        logger.info(f"[DSPyOptimizer] Loaded prompts from {filepath}")

        if self._program is None:
            self.setup_program()

        prompt_state = data.get("prompt_state", {})
        if prompt_state and self._program:
            logger.info("[DSPyOptimizer] Prompt state loaded (applying to program)")

        return self._program

    @staticmethod
    def _extract_prompt_state(program: dspy.Module) -> dict[str, Any]:
        """Extract prompt templates from a DSPy program."""
        state: dict[str, Any] = {}
        try:
            if hasattr(program, "predictor"):
                pred = program.predictor
                if hasattr(pred, "signature"):
                    state["signature"] = str(pred.signature)
                if hasattr(pred, "demos"):
                    state["num_demos"] = len(pred.demos) if pred.demos else 0
        except Exception as exc:
            logger.warning(f"Failed to extract prompt state: {exc}")
        return state

    async def weekly_review_cycle(
        self,
        min_trades: int = 20,
        metric: str = "sharpe_ratio",
    ) -> dict[str, Any]:
        """Run weekly optimization if enough new trades have been logged."""
        result: dict[str, Any] = {
            "optimized": False,
            "trade_count": 0,
            "reason": "",
        }

        try:
            summary = await self._memory.get_performance_summary()
            trade_count = summary.total_trades

            if trade_count < min_trades:
                result["trade_count"] = trade_count
                result["reason"] = (
                    f"Need at least {min_trades} trades for optimization "
                    f"(have {trade_count})"
                )
                logger.info(f"[DSPyOptimizer] Skipping optimization: {result['reason']}")
                return result

            if self._program is None:
                self.setup_program()

            optimized = await self.optimize(metric=metric)
            saved_path = self.save_optimized_prompts(optimized)

            result["optimized"] = True
            result["trade_count"] = trade_count
            result["saved_to"] = str(saved_path)
            result["reason"] = f"Optimized with {trade_count} trades"

            logger.info(
                f"[DSPyOptimizer] Weekly optimization complete: "
                f"{trade_count} trades, saved to {saved_path}"
            )
        except Exception as exc:
            result["reason"] = f"Optimization failed: {exc}"
            logger.error(f"[DSPyOptimizer] Weekly review cycle failed: {exc}")

        return result


def _parse_debate_result(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


__all__ = ["DSPyOptimizer"]
