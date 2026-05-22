"""Typed runtime status helpers for bot integration flows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RuntimeFailurePolicy(str, Enum):
    """How a runtime integration failure should be handled by callers."""

    RAISE = "raise"
    FALLBACK = "fallback"
    RETURN_STATUS = "return_status"


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """Typed result for runtime integrations that may fail without crashing the bot."""

    ok: bool
    code: str
    message: str = ""
    policy: RuntimeFailurePolicy = RuntimeFailurePolicy.RETURN_STATUS

    def __bool__(self) -> bool:
        return self.ok

    @classmethod
    def success(cls, code: str = "ok", message: str = "") -> "RuntimeStatus":
        return cls(ok=True, code=code, message=message)

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        *,
        policy: RuntimeFailurePolicy = RuntimeFailurePolicy.RETURN_STATUS,
    ) -> "RuntimeStatus":
        return cls(ok=False, code=code, message=message, policy=policy)