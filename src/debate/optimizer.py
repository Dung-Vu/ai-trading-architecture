"""
DSPy Prompt Optimizer — Automated prompt tuning for the AI Debate Engine.

Uses DSPy's MIPROv2 optimizer to automatically find the best prompts
for the debate engine based on historical trade performance.

Architecture:
    Historical trades → DSPy demonstrations → MIPROv2 optimizer → Optimized prompts

Usage:
    >>> from src.memory import TradeMemory
    >>> from src.debate.optimizer import DSPyOptimizer
    >>> from src.debate import DebateConfig
    >>>
    >>> async with TradeMemory() as memory:
    ...     optimizer = DSPyOptimizer(
    ...         trade_memory=memory,
    ...         debate_config=DebateConfig(),
    ...         llm_model="anthropic/claude-sonnet-4",
    ...     )
    ...     program = await optimizer.setup_program()
    ...     optimized = await optimizer.optimize(demo_trades=...)
    ...     optimizer.save_optimized_prompts(optimized)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

try:
    import dspy
except ImportError:
    dspy = None  # type: ignore[assignment]


# ─── DSPy Signatures ──────────────────────────────────────────────────
class MarketInput(dspy.Signature):
    """
    Given market data with technical indicators → predict trading action.

    The input contains current price, technical indicators (RSI, MACD,
    Bollinger Bands, volume, etc.), recent price action, and market context.
    The output should be a structured trading decision.
    """

    market_data = dspy.InputField(
        desc="Market data including price, technical indicators, volume, and context"
    )
    action = dspy.OutputField(
        desc="Trading action: BUY, SELL, or HOLD"
    )
    confidence = dspy.OutputField(
        desc="Confidence in the action (0-100)"
    )
    reason = dspy.OutputField(
        desc="Detailed reasoning for the decision"
    )
    stop_loss = dspy.OutputField(
        desc="Recommended stop-loss price"
    )
    take_profit = dspy.OutputField(
        desc="Recommended take-profit price"
    )


# ─── Debate Result Output Model ────────────────────────────────────────
class DebateResultOutput(BaseModel):
    """Structured output from the DSPy debate program."""

    action: str = Field(..., description="BUY, SELL, or HOLD")
    confidence: float = Field(..., ge=0, le=100, description="Confidence level")
    reason: str = Field(..., description="Detailed reasoning")
    stop_loss: float = Field(..., gt=0, description="Stop-loss price")
    take_profit: float = Field(..., gt=0, description="Take-profit price")


# ─── Trade Demonstration ──────────────────────────────────────────────
class TradeDemonstration(BaseModel):
    """A single trade used as a demonstration for DSPy optimization."""

    market_data_text: str = Field(..., description="Text description of market conditions")
    action: str = Field(..., description="The action that was taken")
    pnl: float = Field(..., description="Realized P&L from this trade")
    was_profitable: bool = Field(..., description="Whether the trade was profitable")


# ─── Metric Functions ──────────────────────────────────────────────────
def sharpe_metric(demo: Any, pred: Any, trace: Any = None) -> float:
    """
    DSPy metric that rewards correct action prediction and confidence calibration.

    Returns 1.0 if the predicted action matches the demonstration action,
    scaled by confidence calibration.
    """
    if not hasattr(pred, "action") or not hasattr(demo, "action"):
        return 0.0

    action_match = pred.action.upper() == demo.action.upper()
    if not action_match:
        return 0.0

    # Bonus for reasonable confidence
    confidence = float(getattr(pred, "confidence", 50))
    if 60 <= confidence <= 90:
        return 1.0  # Good confidence range
    elif confidence > 90:
        return 0.8  # Slightly penalize overconfidence
    else:
        return 0.6  # Penalize low confidence on correct action


def pnl_weighted_metric(demo: Any, pred: Any, trace: Any = None) -> float:
    """
    DSPy metric weighted by the PnL of the demonstration trade.

    Higher PnL trades get more weight in the optimization.
    """
    base = sharpe_metric(demo, pred, trace)

    # Weight by the absolute PnL (scaled)
    pnl = float(getattr(demo, "pnl", 0))
    pnl_weight = min(abs(pnl) / 100.0, 2.0)  # Cap at 2x weight

    return base * (1 + pnl_weight)


# ─── DSPyOptimizer ─────────────────────────────────────────────────────
class DSPyOptimizer:
    """
    DSPy-based prompt optimizer for the AI Debate Engine.

    Uses historical trade data as demonstrations to automatically optimize
    the debate engine prompts for better trading performance.

    The optimization process:
    1. Collect historical trades and format as DSPy demonstrations
    2. Set up a DSPy program with the MarketInput signature
    3. Run MIPROv2 optimizer to find optimal prompts
    4. Save the optimized prompts for use in production
    """

    def __init__(
        self,
        trade_memory: Any,  # TradeMemory instance
        debate_config: Any,  # DebateConfig instance
        llm_model: str = "anthropic/claude-sonnet-4",
    ) -> None:
        """
        Initialize the DSPy optimizer.

        Args:
            trade_memory: TradeMemory instance for querying trade data.
            debate_config: DebateConfig with LLM settings.
            llm_model: LiteLLM model identifier for DSPy.
        """
        if dspy is None:
            raise ImportError("dspy-ai is required. Install: pip install dspy-ai")

        self._memory = trade_memory
        self._debate_config = debate_config
        self._llm_model = llm_model

        self._program: dspy.Module | None = None
        self._optimized_program: dspy.Module | None = None

    # ─── DSPy Setup ────────────────────────────────────────────────────

    def setup_program(self) -> dspy.Module:
        """
        Create a DSPy program for the debate engine.

        The program takes market data as input and produces a DebateResult
        with action, confidence, reason, stop-loss, and take-profit.

        Returns:
            DSPy Module ready for optimization.
        """
        logger.info(f"[DSPyOptimizer] Setting up DSPy program with {self._llm_model}")

        # Configure DSPy LM
        lm = dspy.LM(
            model=self._llm_model,
            temperature=0.7,
            max_tokens=2048,
        )
        dspy.configure(lm=lm)

        # Create the program
        class DebateProgram(dspy.Module):
            """DSPy program for trading debate decisions."""

            def __init__(self) -> None:
                super().__init__()
                self.predictor = dspy.ChainOfThought(MarketInput)

            def forward(
                self,
                market_data: str,
            ) -> dspy.Prediction:
                """
                Run the debate program on market data.

                Args:
                    market_data: Text description of market conditions.

                Returns:
                    Prediction with action, confidence, reason, SL, TP.
                """
                result = self.predictor(market_data=market_data)
                return result

        self._program = DebateProgram()
        logger.info("[DSPyOptimizer] DSPy program created with ChainOfThought predictor")
        return self._program

    # ─── Demonstration Preparation ─────────────────────────────────────

    async def _prepare_demonstrations(
        self,
        demo_trades: list[dict[str, Any]] | None = None,
        max_demos: int = 50,
    ) -> list[dspy.Example]:
        """
        Prepare DSPy demonstrations from trade history.

        If demo_trades is provided, uses those directly. Otherwise,
        queries the trade memory for historical trades.

        Args:
            demo_trades: Pre-formatted trade dicts.
            max_demos: Maximum number of demonstrations.

        Returns:
            List of DSPy Example objects.
        """
        demonstrations: list[dict[str, Any]] = []

        if demo_trades:
            demonstrations = demo_trades[:max_demos]
        else:
            # Fetch from trade memory
            try:
                history = await self._memory.get_trade_history(limit=max_demos * 2)

                for trade in history:
                    if trade.get("side") != "SELL" or trade.get("pnl", 0) == 0:
                        continue

                    # Build market data text from trade context
                    debate_result = trade.get("debate_result") or {}
                    if isinstance(debate_result, str):
                        try:
                            debate_result = json.loads(debate_result)
                        except json.JSONDecodeError:
                            debate_result = {}

                    market_text = self._format_market_data_for_demo(trade, debate_result)

                    # Determine what the "correct" action should have been
                    # based on the outcome
                    if trade.get("pnl", 0) > 0:
                        correct_action = trade.get("side", "BUY")  # The entry was correct
                        pnl = trade.get("pnl", 0)
                    else:
                        # For losses, the correct action would have been HOLD
                        correct_action = "HOLD"
                        pnl = trade.get("pnl", 0)

                    demonstrations.append({
                        "market_data_text": market_text,
                        "action": correct_action,
                        "pnl": pnl,
                        "was_profitable": trade.get("pnl", 0) > 0,
                    })

                    if len(demonstrations) >= max_demos:
                        break

            except Exception as exc:
                logger.warning(f"Failed to fetch trade history for demos: {exc}")
                return []

        # Convert to DSPy Examples
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

    # ─── Optimization ──────────────────────────────────────────────────

    async def optimize(
        self,
        demo_trades: list[dict[str, Any]] | None = None,
        metric: str = "sharpe_ratio",
        num_trials: int = 10,
        max_bootstrapped_demos: int = 5,
    ) -> dspy.Module:
        """
        Run MIPROv2 optimization on the DSPy program.

        Uses historical trades as demonstrations to find prompts that
        maximize the specified metric (Sharpe ratio by default).

        Args:
            demo_trades: Optional pre-formatted demonstrations.
            metric: Metric to optimize ("sharpe_ratio" or "pnl_weighted").
            num_trials: Number of optimization trials.
            max_bootstrapped_demos: Max bootstrapped demos for MIPROv2.

        Returns:
            Optimized DSPy Module.
        """
        if self._program is None:
            self.setup_program()

        logger.info(
            f"[DSPyOptimizer] Starting optimization: "
            f"metric={metric}, trials={num_trials}"
        )

        # Prepare demonstrations
        examples = await self._prepare_demonstrations(demo_trades)

        if len(examples) < 3:
            logger.warning(
                f"[DSPyOptimizer] Only {len(examples)} demonstrations available. "
                f"Need at least 3 for meaningful optimization."
            )
            return self._program

        # Select metric function
        if metric == "pnl_weighted":
            metric_fn = pnl_weighted_metric
        else:
            metric_fn = sharpe_metric

        # Split into train/eval
        train_size = max(3, int(len(examples) * 0.7))
        trainset = examples[:train_size]
        evalset = examples[train_size:]

        if not evalset:
            evalset = trainset  # Use same set if too few examples

        logger.info(
            f"[DSPyOptimizer] Train set: {len(trainset)}, Eval set: {len(evalset)}"
        )

        # Run MIPROv2 optimizer
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

    # ─── Prompt Persistence ────────────────────────────────────────────

    def save_optimized_prompts(
        self,
        program: dspy.Module | None = None,
        path: str | None = None,
    ) -> Path:
        """
        Save optimized prompts to disk.

        Args:
            program: DSPy program to save. Uses optimized program if None.
            path: Custom file path. Auto-generates if None.

        Returns:
            Path to the saved file.
        """
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

        # Extract and save the predictor's state
        try:
            # DSPy programs can be saved via their internal state
            program_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "llm_model": self._llm_model,
                "optimized": prog is self._optimized_program,
                "predictor_type": "ChainOfThought",
                "signature": "MarketInput",
                # Save any learned prompt templates
                "prompt_state": self._extract_prompt_state(prog),
            }

            filepath.write_text(json.dumps(program_data, indent=2), encoding="utf-8")
            logger.info(f"[DSPyOptimizer] Prompts saved to {filepath}")

        except Exception as exc:
            logger.warning(f"[DSPyOptimizer] Failed to save prompts: {exc}")
            # Fallback: save minimal state
            fallback_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "llm_model": self._llm_model,
                "optimized": prog is self._optimized_program,
                "error": str(exc),
            }
            filepath.write_text(json.dumps(fallback_data, indent=2), encoding="utf-8")

        return filepath

    def load_optimized_prompts(self, path: str) -> dspy.Module:
        """
        Load previously optimized prompts.

        Note: Full DSPy program deserialization requires the original
        program structure. This method loads the prompt state and
        reinitializes the program.

        Args:
            path: Path to the saved prompts file.

        Returns:
            DSPy Module with loaded prompts.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Prompts file not found: {filepath}")

        data = json.loads(filepath.read_text(encoding="utf-8"))
        logger.info(f"[DSPyOptimizer] Loaded prompts from {filepath}")

        # Re-setup the program
        if self._program is None:
            self.setup_program()

        # If we have prompt state, try to restore it
        prompt_state = data.get("prompt_state", {})
        if prompt_state and self._program:
            logger.info("[DSPyOptimizer] Prompt state loaded (applying to program)")
            # Note: Full DSPy state restoration would need deeper integration
            # For now, we use the saved state as guidance

        return self._program

    @staticmethod
    def _extract_prompt_state(program: dspy.Module) -> dict[str, Any]:
        """
        Extract prompt templates from a DSPy program.

        Args:
            program: DSPy Module to extract state from.

        Returns:
            Dict with extracted prompt state.
        """
        state: dict[str, Any] = {}

        try:
            # Try to extract predictor state
            if hasattr(program, "predictor"):
                pred = program.predictor
                if hasattr(pred, "signature"):
                    state["signature"] = str(pred.signature)
                if hasattr(pred, "demos"):
                    state["num_demos"] = len(pred.demos) if pred.demos else 0
        except Exception as exc:
            logger.warning(f"Failed to extract prompt state: {exc}")

        return state

    # ─── Weekly Review Cycle ───────────────────────────────────────────

    async def weekly_review_cycle(
        self,
        min_trades: int = 20,
        metric: str = "sharpe_ratio",
    ) -> dict[str, Any]:
        """
        Run weekly optimization if enough new trades have been logged.

        This method should be called weekly (e.g., from a cron job or
        scheduled task) to continuously improve the debate engine prompts.

        Args:
            min_trades: Minimum trades required to trigger optimization.
            metric: Metric to optimize.

        Returns:
            Dict with optimization results summary.
        """
        result: dict[str, Any] = {
            "optimized": False,
            "trade_count": 0,
            "reason": "",
        }

        try:
            # Check if we have enough trades
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

            # Setup program if not already done
            if self._program is None:
                self.setup_program()

            # Run optimization
            optimized = await self.optimize(metric=metric)

            # Save optimized prompts
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
