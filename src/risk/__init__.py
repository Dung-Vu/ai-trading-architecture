"""Risk management module."""


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
]
