"""Per-symbol trading pipeline for the full trading bot."""

from __future__ import annotations

import asyncio
import hashlib
import random
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.config import get_default_simulated_base_prices
from src.shared_utils import trim_mapping_size


class FullTradingPipelineMixin:
    async def _process_symbol(self, symbol: str) -> None:
        """
        Process a single trading symbol through the full unified pipeline.

        Pipeline:
        1. Fetch market data (prices, indicators)
        2. Fetch news sentiment
        3. Run strategy → generate signal
        4. Query knowledge graph for similar patterns
        5. Query Mem0 for similar past trades
        6. Run debate (with news sentiment + memory context)
        7. Risk Manager pre-trade check
        8. Execute order
        9. Log to memory (PostgreSQL + Mem0)
        10. Update knowledge graph
        11. Send Telegram alert
        """
        logger.debug(f"🔄 Processing {symbol} (loop #{self._loop_count})")

        market_context = await self._prepare_symbol_market_context(symbol)
        if market_context is None:
            return

        market_data, current_price, indicators = market_context
        signal_action = await self._get_actionable_signal(
            symbol,
            market_data,
            current_price,
        )
        if signal_action is None:
            return

        debate_result = await self._run_symbol_debate(symbol, market_data, indicators)
        if debate_result is None:
            return

        await self._execute_symbol_trade_pipeline(
            symbol,
            current_price,
            indicators,
            market_data,
            debate_result,
        )

    async def _prepare_symbol_market_context(
        self,
        symbol: str,
    ) -> tuple[dict[str, Any], float, dict[str, Any]] | None:
        """Fetch market inputs for one symbol and process pending exit orders."""
        market_data = await self._build_market_data(symbol)
        if market_data is None:
            logger.debug(f"  No market data for {symbol}, skipping")
            return None

        current_price = market_data.get("price", 0)
        indicators = market_data.get("indicators", {})
        await self._handle_pending_exit_orders(
            symbol,
            current_price,
            self._send_alert,
        )
        return market_data, current_price, indicators

    async def _get_actionable_signal(
        self,
        symbol: str,
        market_data: dict[str, Any],
        current_price: float,
    ) -> str | None:
        """Run the strategy and stop early when no trade should be considered."""
        signal_action = self._run_strategy(symbol, market_data)
        if signal_action == "HOLD":
            logger.debug(f"  [{symbol}] Strategy says HOLD")
            return None

        logger.info(
            f"  [{symbol}] Strategy signal: {signal_action} "
            f"(price=${current_price:,.2f})"
        )
        return signal_action

    def _build_symbol_memory_context(
        self,
        symbol: str,
        indicators: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str]:
        """Gather the reusable memory context for one debate round."""
        kg_context = self._query_knowledge_graph(indicators)
        mem0_context = self._query_mem0_memory(symbol, indicators)
        return kg_context, mem0_context

    async def _run_symbol_debate(
        self,
        symbol: str,
        market_data: dict[str, Any],
        indicators: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fetch debate inputs and return only actionable debate decisions."""
        sentiment = await self._fetch_sentiment(symbol)
        kg_context, mem0_context = self._build_symbol_memory_context(symbol, indicators)
        debate_result = await self._run_debate(
            symbol,
            market_data,
            sentiment,
            kg_context,
            mem0_context,
        )
        if debate_result is None:
            logger.warning(f"  [{symbol}] Debate engine unavailable")
            return None

        final_action = debate_result.get("action", "HOLD")
        if final_action == "HOLD":
            logger.info(f"  [{symbol}] Debate says HOLD")
            return None

        return debate_result

    async def _execute_symbol_trade_pipeline(
        self,
        symbol: str,
        current_price: float,
        indicators: dict[str, Any],
        market_data: dict[str, Any],
        debate_result: dict[str, Any],
    ) -> None:
        """Apply risk checks, execute the trade, and record side effects."""
        final_action = debate_result.get("action", "HOLD")
        approved, reason = await self._run_risk_check(
            symbol,
            final_action,
            current_price,
        )
        if not approved:
            logger.info(f"  [{symbol}] Risk rejected: {reason}")
            return

        trade_result = await self._execute_trade(
            symbol,
            final_action,
            current_price,
            debate_result,
        )
        if trade_result is None:
            return

        await self._log_trade(symbol, trade_result, debate_result, market_data)
        self._update_knowledge_graph(indicators, final_action, trade_result)
        await self._send_alert(symbol, trade_result, debate_result)

    # ─── Pipeline Steps ────────────────────────────────────────────────

    async def _build_market_data(self, symbol: str) -> dict[str, Any] | None:
        """Build market data dict with price and indicators."""
        price = await self._get_latest_price(symbol)
        if price is None:
            if self.mode != "dryrun":
                logger.debug(f"  No live price available for {symbol}, skipping")
                return None
            price = self._simulate_price(symbol)
            if price is None:
                return None

        indicators = self._build_simulated_indicators(symbol, price)
        market_conditions = self._derive_market_conditions(indicators)

        return {
            "symbol": symbol,
            "price": price,
            "indicators": indicators,
            "market_conditions": market_conditions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _simulate_price(self, symbol: str) -> float | None:
        """Simulate a realistic price for dry-run mode."""
        cached = self._simulated_price_cache.get(symbol)
        if cached is not None:
            return cached

        base_prices = getattr(
            self.config,
            "simulated_base_prices",
            get_default_simulated_base_prices(),
        )

        base = base_prices.get(symbol)
        if base is None:
            return None

        seed_material = f"{self._simulation_seed}:{symbol}".encode("utf-8")
        seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
        rng = random.Random(seed)
        price = round(base * (1 + rng.uniform(-0.03, 0.03)), 2)
        self._simulated_price_cache[symbol] = price
        return price

    def _build_simulated_indicators(
        self,
        symbol: str,
        price: float,
    ) -> dict[str, Any]:
        """Generate deterministic session-scoped indicators for a symbol/price pair."""
        cache_key = (symbol, round(price, 8))
        cached = self._indicator_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        seed_material = f"{self._indicator_seed}:{symbol}:{cache_key[1]}".encode("utf-8")
        seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
        rng = random.Random(seed)

        indicators = {
            "price": price,
            "rsi": round(rng.uniform(20, 80), 1),
            "sma_fast": round(price * (1 + rng.uniform(-0.02, 0.02)), 2),
            "sma_slow": round(price * (1 + rng.uniform(-0.05, 0.05)), 2),
            "volume": round(rng.uniform(0.5, 3.0), 2),
            "volume_high": rng.choice([True, False]),
        }

        self._indicator_cache[cache_key] = dict(indicators)
        trim_mapping_size(self._indicator_cache, 256)

        return indicators

    @staticmethod
    def _derive_market_conditions(indicators: dict[str, Any]) -> dict[str, Any]:
        """Derive stable market-condition labels from the indicator snapshot."""
        sma_fast = float(indicators.get("sma_fast", indicators.get("price", 0.0)))
        sma_slow = float(indicators.get("sma_slow", indicators.get("price", 0.0)))
        price = float(indicators.get("price", 0.0))
        volume_high = bool(indicators.get("volume_high", False))

        if sma_fast > sma_slow:
            trend = "uptrend"
        elif sma_fast < sma_slow:
            trend = "downtrend"
        else:
            trend = "sideways"

        divergence = abs(sma_fast - sma_slow) / price if price > 0 else 0.0
        if divergence >= 0.03:
            volatility = "high"
        elif divergence >= 0.01:
            volatility = "medium"
        else:
            volatility = "low"

        return {
            "trend": trend,
            "volatility": volatility,
            "volume_high": volume_high,
        }

    async def _fetch_sentiment(self, symbol: str) -> dict[str, Any]:
        """Fetch news sentiment for a symbol."""
        if not self._news_pipeline:
            return {"score": 0.0, "positive": 0, "negative": 0}

        try:
            ticker = symbol.split("/")[0]
            # News fetching is synchronous — run in executor
            loop = asyncio.get_running_loop()
            news_items = await loop.run_in_executor(
                None, self._news_pipeline.fetch_crypto_news, ticker, 24
            )

            sentiment = await loop.run_in_executor(
                None, self._news_pipeline.analyze_sentiment, news_items
            )

            logger.info(
                f"  [{symbol}] Sentiment: score={sentiment['overall_score']:.2f}, "
                f"+{sentiment['positive_count']}/-{sentiment['negative_count']}"
            )
            return sentiment
        except Exception as exc:
            logger.warning(f"  [{symbol}] Sentiment fetch failed: {exc}")
            return {"score": 0.0, "positive": 0, "negative": 0}

    def _query_knowledge_graph(self, indicators: dict) -> list[dict]:
        """Query knowledge graph for matching patterns."""
        if self._knowledge_graph is None:
            return []

        # Build condition string from current indicators
        rsi = indicators.get("rsi", 50)
        volume_high = indicators.get("volume_high", False)

        condition_parts = []
        if rsi < 30:
            condition_parts.append("RSI oversold")
        elif rsi > 70:
            condition_parts.append("RSI overbought")

        if volume_high:
            condition_parts.append("volume high")

        if not condition_parts:
            return []

        condition = " AND ".join(condition_parts)
        return self._knowledge_graph.query_pattern(condition)

    def _query_mem0_memory(self, symbol: str, indicators: dict) -> str:
        """Query Mem0 for similar past trades."""
        if not self._mem0_memory:
            return ""

        rsi = indicators.get("rsi", 50)
        query = f"{symbol} RSI {rsi:.0f}"

        if rsi < 30:
            query += " oversold bounce"
        elif rsi > 70:
            query += " overbought rejection"

        try:
            similar = self._mem0_memory.query_similar_trades(query, limit=3)
            if similar:
                summary_parts = []
                for s in similar:
                    outcome = s.get("outcome", "pending")
                    pnl = s.get("pnl", 0)
                    summary_parts.append(
                        f"Past: {outcome} (${pnl:+.2f})"
                    )
                return "; ".join(summary_parts)
        except Exception as exc:
            logger.debug(f"Mem0 query failed: {exc}")

        return ""

    async def _run_debate(
        self,
        symbol: str,
        market_data: dict,
        sentiment: dict,
        kg_context: list[dict],
        mem0_context: str,
    ) -> dict[str, Any] | None:
        """Run debate engine with full context."""
        # Build enriched market data
        enriched = dict(market_data)
        enriched["news_sentiment"] = sentiment
        enriched["knowledge_graph_patterns"] = kg_context
        enriched["memory_context"] = mem0_context

        if self._debate_engine:
            try:
                positions = self._positions.get(symbol, {})
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._debate_engine.run_debate(
                        market_data=enriched,
                        current_positions={symbol: positions},
                        portfolio=self._portfolio,
                        symbol=symbol,
                    ),
                )

                return {
                    "action": result.action,
                    "confidence": result.confidence,
                    "reasoning": result.reason,
                    "stop_loss": result.stop_loss,
                    "take_profit": result.take_profit,
                    "risk_decision": result.risk_decision,
                    "bull_argument": result.bull_argument,
                    "bear_argument": result.bear_argument,
                    "devil_argument": result.devil_argument,
                    "rounds": len(result.rounds),
                }
            except Exception as exc:
                logger.warning(f"Debate engine error: {exc}")
                return None

        # Fallback: use strategy signal directly
        return {
            "action": market_data.get("signal", "HOLD"),
            "confidence": 60.0,
            "reasoning": f"Strategy {self.strategy_name} signal (no debate engine)",
            "stop_loss": 0,
            "take_profit": 0,
            "risk_decision": "APPROVE",
            "bull_argument": "",
            "bear_argument": "",
            "devil_argument": "",
            "rounds": 0,
        }

    async def _log_trade(
        self,
        symbol: str,
        trade_result: dict,
        debate_result: dict,
        market_data: dict,
    ) -> None:
        """Log trade to all memory systems."""
        trade_record = {
            "symbol": symbol,
            "side": trade_result["side"],
            "quantity": trade_result["quantity"],
            "price": trade_result["price"],
            "strategy": self.strategy_name,
            "mode": self.mode,
            "indicators": market_data.get("indicators", {}),
            "market_conditions": market_data.get("market_conditions", {}),
            "stop_loss": debate_result.get("stop_loss"),
            "take_profit": debate_result.get("take_profit"),
            "ai_confidence": debate_result.get("confidence", 0),
        }

        # PostgreSQL memory
        if self._trade_memory:
            try:
                await self._trade_memory.log_trade(trade_record)
            except Exception as exc:
                logger.warning(f"Failed to log trade to PostgreSQL: {exc}")

        # Mem0 semantic memory
        if self._mem0_memory:
            try:
                self._mem0_memory.add_trade_memory(trade_record, debate_result)
            except Exception as exc:
                logger.warning(f"Failed to log trade to Mem0: {exc}")

        # Log debate
        debate_record = {
            "symbol": symbol,
            "bull_arg": debate_result.get("bull_argument", "")[:500],
            "bear_arg": debate_result.get("bear_argument", "")[:500],
            "devil_arg": debate_result.get("devil_argument", "")[:500],
            "judge_action": debate_result.get("action", "HOLD"),
            "judge_confidence": debate_result.get("confidence", 50),
            "risk_action": debate_result.get("risk_decision", "APPROVE"),
            "risk_reasoning": "",
            "rounds": debate_result.get("rounds", 0),
        }

        if self._trade_memory:
            try:
                await self._trade_memory.log_debate(debate_record)
            except Exception as exc:
                logger.warning(f"Failed to log debate to PostgreSQL: {exc}")

    def _update_knowledge_graph(
        self,
        indicators: dict,
        action: str,
        trade_result: dict,
    ) -> None:
        """Add pattern to knowledge graph."""
        if self._knowledge_graph is None:
            return

        if action.upper() != "BUY":
            return

        # Build condition string
        rsi = indicators.get("rsi", 50)
        volume_high = indicators.get("volume_high", False)

        conditions = []
        if rsi < 30:
            conditions.append("RSI < 30")
        elif rsi > 70:
            conditions.append("RSI > 70")

        if volume_high:
            conditions.append("volume > avg")

        if conditions:
            condition_str = " AND ".join(conditions)
            pattern_id = self._knowledge_graph.add_pattern(
                condition=condition_str,
                action=action,
                outcome="pending",  # Will be updated when trade closes
                confidence=0.5,
            )

            position = self._positions.get(trade_result.get("symbol", ""))
            if position is not None:
                position["knowledge_graph_pattern_id"] = pattern_id
                position["knowledge_graph_condition"] = condition_str

    def _on_position_closed(
        self,
        symbol: str,
        trade_result: dict[str, Any],
        closed_position: dict[str, Any] | None = None,
    ) -> None:
        """Update knowledge-graph outcomes when a tracked position closes."""
        del symbol

        if self._knowledge_graph is None or not closed_position:
            return

        pattern_id = closed_position.get("knowledge_graph_pattern_id")
        if not pattern_id:
            return

        pnl = float(trade_result.get("pnl", 0.0) or 0.0)
        if pnl > 0:
            outcome = "win"
        elif pnl < 0:
            outcome = "loss"
        else:
            outcome = "break_even"

        self._knowledge_graph.update_pattern_by_id(
            pattern_id,
            outcome=outcome,
            pnl=pnl,
        )

    async def _send_alert(
        self,
        symbol: str,
        trade_result: dict,
        debate_result: dict,
    ) -> None:
        """Send Telegram alert for trade execution."""
        if self._telegram_bot is None:
            return

        try:
            await self._telegram_bot.send_trade_alert(
                side=trade_result.get("side", "BUY"),
                symbol=symbol,
                quantity=trade_result.get("quantity", 0),
                price=trade_result.get("price", 0),
                pnl=trade_result.get("pnl"),
                strategy=self.strategy_name,
                ai_confidence=debate_result.get("confidence"),
                mode=self.mode,
            )
            logger.info(f"  [{symbol}] Telegram alert sent")
        except Exception as exc:
            logger.warning(f"Failed to send alert: {exc}")
