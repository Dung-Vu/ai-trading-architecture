"""Kill switch for emergency trading halt."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from loguru import logger


class KillSwitchState(Enum):
    """Kill switch states."""

    DISARMED = "disarmed"
    ARMED = "armed"
    TRIGGERED = "triggered"


@dataclass
class TriggerRecord:
    """Record of a kill switch trigger event."""

    reason: str
    timestamp: str
    state_before: KillSwitchState


class KillSwitch:
    """Emergency kill switch to halt all trading activity.

    The kill switch has three states:
    - DISARMED: Inactive, trading allowed
    - ARMED: Ready to trigger on conditions
    - TRIGGERED: All trading halted, requires manual disarm
    """

    def __init__(self) -> None:
        """Initialize kill switch in disarmed state."""
        self._state = KillSwitchState.DISARMED
        self._triggers: list[TriggerRecord] = []
        logger.info("KillSwitch initialized in DISARMED state")

    @property
    def state(self) -> KillSwitchState:
        """Current kill switch state."""
        return self._state

    def arm(self) -> None:
        """Arm the kill switch (ready to auto-trigger).

        Armed state means the kill switch will automatically
        trigger if conditions are met.
        """
        previous = self._state
        self._state = KillSwitchState.ARMED
        logger.info(f"KillSwitch armed (was {previous.value})")

    def trigger(self, reason: str) -> None:
        """Activate the kill switch, halting all trading.

        Args:
            reason: Description of why the kill switch was triggered.
        """
        previous = self._state
        record = TriggerRecord(
            reason=reason,
            timestamp=datetime.now(UTC).isoformat(),
            state_before=previous,
        )
        self._triggers.append(record)
        self._state = KillSwitchState.TRIGGERED

        logger.critical(
            f"KillSwitch TRIGGERED: {reason} "
            f"(was {previous.value}, total triggers: {len(self._triggers)})"
        )

    def disarm(self, confirmation: str = "") -> None:
        """Deactivate the kill switch.

        Args:
            confirmation: Manual confirmation text (must be 'DISARM').

        Raises:
            ValueError: If confirmation is not provided.
        """
        if confirmation != "DISARM":
            raise ValueError(
                "Kill switch disarm requires confirmation. "
                "Pass confirmation='DISARM' to proceed."
            )

        previous = self._state
        self._state = KillSwitchState.DISARMED
        logger.warning(
            f"KillSwitch DISARMED (was {previous.value})"
        )

    def is_active(self) -> bool:
        """Check if kill switch is currently active.

        Returns:
            True if kill switch is triggered (trading halted).
        """
        return self._state == KillSwitchState.TRIGGERED

    def get_last_trigger(self) -> dict | None:
        """Get the most recent trigger event.

        Returns:
            Dictionary with reason and timestamp, or None if never triggered.
        """
        if not self._triggers:
            return None

        last = self._triggers[-1]
        return {
            "reason": last.reason,
            "timestamp": last.timestamp,
            "state_before": last.state_before.value,
        }

    def get_trigger_history(self) -> list[dict]:
        """Get full trigger history.

        Returns:
            List of all trigger records.
        """
        return [
            {
                "reason": t.reason,
                "timestamp": t.timestamp,
                "state_before": t.state_before.value,
            }
            for t in self._triggers
        ]

    def auto_check(
        self,
        max_drawdown_pct: float,
        current_drawdown: float,
    ) -> bool:
        """Auto-trigger kill switch if drawdown exceeds limit.

        Only triggers if kill switch is armed and condition is met.

        Args:
            max_drawdown_pct: Maximum allowed drawdown fraction.
            current_drawdown: Current drawdown fraction.

        Returns:
            True if kill switch was triggered.
        """
        if self._state != KillSwitchState.ARMED:
            return False

        if current_drawdown >= max_drawdown_pct:
            self.trigger(
                f"Auto-triggered: drawdown {current_drawdown:.2%} "
                f"exceeds limit {max_drawdown_pct:.2%}"
            )
            return True

        return False

    def reset(self) -> None:
        """Reset kill switch to disarmed state and clear history."""
        self._state = KillSwitchState.DISARMED
        self._triggers.clear()
        logger.info("KillSwitch reset to DISARMED state, history cleared")
