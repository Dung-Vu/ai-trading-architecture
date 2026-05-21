"""Periodic maintenance checks for the full trading bot."""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger


class FullTradingMaintenanceMixin:
    async def _check_drawdown_kill_switch(self) -> None:
        """Auto-trigger the kill switch if portfolio drawdown breaches the limit."""
        if not self._kill_switch or not self._risk_engine:
            return

        portfolio = await self._get_portfolio_state()
        self._risk_engine.snapshot_equity(
            current_equity=portfolio["total_value"],
            start_equity=self.config.trading.initial_capital,
        )

        status = self._risk_engine.get_status()
        if self._kill_switch.auto_check(
            max_drawdown_pct=self._risk_engine.max_drawdown_pct,
            current_drawdown=status.current_drawdown_pct,
        ):
            self._running = False
            self._shutdown_event.set()

    # ─── Maintenance Tasks ─────────────────────────────────────────────

    async def _check_weekly_review(self) -> None:
        """Run weekly review if enough time has passed."""
        if not self._trade_memory or not self._weekly_reviewer:
            return

        now = datetime.now(timezone.utc)
        last = self._last_weekly_review

        if last is None or (now - last).days >= 7:
            try:
                report = await self._weekly_reviewer.generate_report()
                self._weekly_reviewer.save_report(report)

                logger.info("📊 Weekly Review:")
                logger.info("  Weekly review report generated and saved")

                self._last_weekly_review = now
            except Exception as exc:
                logger.warning(f"Weekly review failed: {exc}")

    async def _check_autotune(self) -> None:
        """Run auto-tune weekly optimization cycle."""
        if not self._auto_tuner:
            return

        now = datetime.now(timezone.utc)

        # Run every 7 days
        if self._last_autotune is None or (now - self._last_autotune).days >= 7:
            try:
                # Check if decay detected
                needs_optimization = await self._auto_tuner.detect_strategy_decay()

                if needs_optimization:
                    logger.info("🔧 Strategy decay detected — running optimization...")
                    report = await self._auto_tuner.weekly_optimization_cycle()
                    logger.info(f"  Optimization complete: {report.get('status', 'unknown')}")
                else:
                    # Get recommendations regardless
                    recommendations = await self._auto_tuner.get_optimization_recommendations()
                    for rec in recommendations:
                        logger.info(f"  💡 {rec}")

                self._last_autotune = now
                await self._save_state()
            except Exception as exc:
                logger.warning(f"Auto-tune check failed: {exc}")

    async def _check_memory_cleanup(self) -> None:
        """Clean up old Mem0 memories periodically."""
        if not self._mem0_memory:
            return

        now = datetime.now(timezone.utc)

        # Run cleanup every 30 days
        if self._last_memory_cleanup is None or (now - self._last_memory_cleanup).days >= 30:
            try:
                removed = self._mem0_memory.clear_old_memories(days=90)
                logger.info(
                    f"🧹 Memory cleanup: removed {removed} old memories (>90 days)"
                )
                self._last_memory_cleanup = now
            except Exception as exc:
                logger.warning(f"Memory cleanup failed: {exc}")
