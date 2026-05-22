"""Public risk facade.

Read this package root for the pre-trade approval boundary.
`build_risk_engine(config)` is the canonical constructor for bot wiring.
"""

from __future__ import annotations

from typing import Any


def build_risk_engine(config: Any):
    """Build a RiskEngine from the standard app config object."""
    from .risk_engine import RiskEngine

    risk = getattr(config, "risk", config)
    return RiskEngine(
        max_daily_loss_pct=getattr(risk, "max_daily_loss_pct", 3.0) / 100,
        max_drawdown_pct=getattr(risk, "max_drawdown_pct", 10.0) / 100,
        max_position_pct=getattr(risk, "max_position_pct", 20.0) / 100,
        max_leverage=getattr(risk, "max_leverage", 3),
    )


def __getattr__(name):
    """Lazy imports to avoid requiring all dependencies at package level."""
    if name == "RiskEngine":
        from .risk_engine import RiskEngine
        return RiskEngine
    if name == "KillSwitch":
        from .kill_switch import KillSwitch
        return KillSwitch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RiskEngine",
    "KillSwitch",
    "build_risk_engine",
]
