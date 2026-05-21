"""Reusable indicator calculation helpers for trading strategies."""

from __future__ import annotations

from typing import Any

import pandas as pd


INDICATOR_ALIASES = {
    "stochastic": "stoch",
    "williams_r": "williams",
}


def calculate_indicator(
    bars_df: pd.DataFrame,
    indicator_name: str,
    **params: Any,
) -> pd.Series | tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate a technical indicator from an OHLCV DataFrame."""
    import ta.momentum
    import ta.trend
    import ta.volatility
    import ta.volume

    close = bars_df["close"]
    high = bars_df.get("high", close)
    low = bars_df.get("low", close)
    volume = bars_df.get("volume", pd.Series(0, index=close.index))

    def period(default: int) -> int:
        return int(params.get("window", params.get("period", default)))

    def bollinger_bands() -> ta.volatility.BollingerBands:
        std_dev = params.get("std_dev", params.get("window_dev", 2.0))
        return ta.volatility.BollingerBands(
            close,
            window=period(20),
            window_dev=float(std_dev),
        )

    normalized_name = INDICATOR_ALIASES.get(
        indicator_name.lower(),
        indicator_name.lower(),
    )

    match normalized_name:
        case "sma":
            return ta.trend.SMAIndicator(close, window=period(20)).sma_indicator()
        case "ema":
            return ta.trend.EMAIndicator(close, window=period(20)).ema_indicator()
        case "macd":
            fast = params.get("window_fast", params.get("fast", 12))
            slow = params.get("window_slow", params.get("slow", 26))
            macd_obj = ta.trend.MACD(
                close,
                window_slow=int(slow),
                window_fast=int(fast),
            )
            signal_col = params.get("signal_col", "macd")
            if signal_col == "macd_signal":
                return macd_obj.macd_signal()
            if signal_col == "macd_diff":
                return macd_obj.macd_diff()
            return macd_obj.macd()
        case "adx":
            return ta.trend.ADXIndicator(high, low, close, window=period(14)).adx()
        case "rsi":
            return ta.momentum.RSIIndicator(close, window=period(14)).rsi()
        case "stoch":
            return ta.momentum.StochasticOscillator(
                high,
                low,
                close,
                window=period(14),
                smooth_window=int(params.get("smooth_window", 3)),
            ).stoch()
        case "williams":
            return ta.momentum.WilliamsRIndicator(
                high,
                low,
                close,
                lbp=int(params.get("period", 14)),
            ).williams_r()
        case "bb_upper":
            return bollinger_bands().bollinger_hband()
        case "bb_middle":
            return bollinger_bands().bollinger_mavg()
        case "bb_lower":
            return bollinger_bands().bollinger_lband()
        case "bb":
            bands = bollinger_bands()
            return (
                bands.bollinger_hband(),
                bands.bollinger_mavg(),
                bands.bollinger_lband(),
            )
        case "atr":
            return ta.volatility.AverageTrueRange(
                high,
                low,
                close,
                window=period(14),
            ).average_true_range()
        case "obv":
            return ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        case "vwap":
            return ta.volume.VolumeWeightedAveragePrice(
                high,
                low,
                close,
                volume,
            ).volume_weighted_average_price()
        case "volume_sma":
            return ta.trend.SMAIndicator(volume, window=period(20)).sma_indicator()
        case _:
            raise ValueError(f"Unknown indicator: {indicator_name!r}")