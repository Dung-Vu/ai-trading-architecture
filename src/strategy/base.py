"""Base strategy class for crypto trading."""

from abc import abstractmethod

import pandas as pd
from loguru import logger
from lumibot.entities import Asset
from lumibot.strategies import Strategy


class MockFilledPositions(list):
    def get_list(self):
        return self


class MockDataSource:
    SOURCE = "MOCK"
    def __init__(self):
        self.datetime_start = None
        self.datetime_end = None
        self._data_store = {}

    def get_datetime(self):
        from datetime import datetime
        return datetime.utcnow()


class MockBroker:
    IS_BACKTESTING_BROKER = True
    def __init__(self):
        self.name = "mock"
        self.data_source = MockDataSource()
        self._filled_positions = MockFilledPositions()
        self.quote_assets = set()
        self.market = "24/7"

    def _add_subscriber(self, subscriber):
        pass


class BaseStrategy(Strategy):
    """Abstract base class for all trading strategies.

    Inherits from Lumibot's Strategy and provides common utilities:
    - Indicator calculation helper
    - Risk/drawdown monitoring
    - Trade logging
    - Crypto market configuration (24/7)

    Subclasses must implement ``on_trading_iteration()``.
    """

    parameters = {
        "symbol": "BTC",
        "quote_asset": "USDT",
        "initial_capital": 100_000,
        "risk_pct": 0.02,
        "max_drawdown": 0.15,
    }

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the strategy and bypass broker checks if running out of Lumibot loop."""
        import os

        # Use MockBroker if no broker is provided to avoid raising ValueError inside Lumibot
        if "broker" not in kwargs or kwargs["broker"] is None:
            kwargs["broker"] = MockBroker()
        if "data_source" not in kwargs or kwargs["data_source"] is None:
            kwargs["data_source"] = kwargs["broker"].data_source

        prev_is_backtesting = os.environ.get("IS_BACKTESTING")
        os.environ["IS_BACKTESTING"] = "true"
        try:
            super().__init__(*args, **kwargs)
        finally:
            if prev_is_backtesting is not None:
                os.environ["IS_BACKTESTING"] = prev_is_backtesting
            else:
                os.environ.pop("IS_BACKTESTING", None)

    def generate_signal(
        self, price: float, indicators: dict, market_data: dict
    ) -> str:
        """Stub for manual signal generation outside the Lumibot lifecycle."""
        return "HOLD"

    def initialize(self) -> None:
        """Set up the strategy for crypto (24/7) markets and initialise state.

        This is called once before any trading iteration.  It configures
        the market type, sets the polling interval, and seeds ``self.vars``
        with persistent state variables.
        """
        # Crypto markets never close
        self.set_market("24/7")
        self.sleeptime = "1M"  # 1-minute candles between iterations

        # Persistent state via self.vars (required by Lumibot)
        self.vars.symbol = self.parameters["symbol"]
        self.vars.quote_asset = self.parameters["quote_asset"]
        self.vars.initial_capital = self.parameters["initial_capital"]
        self.vars.risk_pct = self.parameters["risk_pct"]
        self.vars.max_drawdown = self.parameters.get("max_drawdown", 0.15)

        # Trading state
        self.vars.entry_price = None
        self.vars.trade_count = 0
        self.vars.peak_value = self.parameters["initial_capital"]

        # Build Asset objects
        self.vars.base_asset = Asset(
            self.vars.symbol, asset_type=Asset.AssetType.CRYPTO
        )
        self.vars.quote = Asset(
            self.vars.quote_asset, asset_type=Asset.AssetType.CRYPTO
        )

        logger.info(
            "BaseStrategy initialized: symbol={}/{}, capital={}",
            self.vars.symbol,
            self.vars.quote_asset,
            self.vars.initial_capital,
        )

    # ------------------------------------------------------------------ #
    #  Indicator helper
    # ------------------------------------------------------------------ #

    def _get_indicator(
        self, bars_df: pd.DataFrame, indicator_name: str, **params
    ) -> pd.Series:
        """Calculate a technical indicator from a price DataFrame.

        Uses the ``ta`` library under the hood.  Supported indicator
        families: ``sma``, ``ema``, ``rsi``, ``bb_*``, ``macd``, ``atr``,
        ``adx``, ``obv``, ``vwap``, ``stoch``.

        Parameters
        ----------
        bars_df :
            DataFrame with at least a ``close`` column (and ``volume`` /
            ``high`` / ``low`` where required).
        indicator_name :
            Name of the indicator (see supported list above).
        **params :
            Keyword arguments forwarded to the underlying ta function.

        Returns
        -------
        pd.Series
            The computed indicator values, same length as ``bars_df``.

        Raises
        ------
        ValueError
            If *indicator_name* is not recognised.
        """
        import ta.momentum
        import ta.trend
        import ta.volatility
        import ta.volume

        close = bars_df["close"]
        high = bars_df.get("high", close)
        low = bars_df.get("low", close)
        volume = bars_df.get("volume", pd.Series(0, index=close.index))

        name = indicator_name.lower()

        # -- Trend indicators --
        if name == "sma":
            window = params.get("window", params.get("period", 20))
            return ta.trend.SMAIndicator(close, window=int(window)).sma_indicator()

        if name == "ema":
            window = params.get("window", params.get("period", 20))
            return ta.trend.EMAIndicator(close, window=int(window)).ema_indicator()

        if name == "macd":
            fast = params.get("window_fast", params.get("fast", 12))
            slow = params.get("window_slow", params.get("slow", 26))
            macd_obj = ta.trend.MACD(
                close,
                window_slow=int(slow),
                window_fast=int(fast),
            )
            signal_col = params.get("signal_col", "macd")
            if signal_col == "macd":
                return macd_obj.macd()
            if signal_col == "macd_signal":
                return macd_obj.macd_signal()
            return macd_obj.macd_diff()

        if name == "adx":
            period = params.get("window", params.get("period", 14))
            return ta.trend.ADXIndicator(
                high, low, close, window=int(period)
            ).adx()

        # -- Momentum indicators --
        if name == "rsi":
            period = params.get("window", params.get("period", 14))
            return ta.momentum.RSIIndicator(close, window=int(period)).rsi()

        if name == "stoch" or name == "stochastic":
            fast = params.get("window", params.get("period", 14))
            slow = params.get("smooth_window", 3)
            return ta.momentum.StochasticOscillator(
                high, low, close, window=int(fast), smooth_window=int(slow)
            ).stoch()

        if name == "williams" or name == "williams_r":
            period = params.get("period", 14)
            return ta.momentum.WilliamsRIndicator(
                high, low, close, lbp=int(period)
            ).williams_r()

        # -- Volatility indicators --
        if name in ("bb_upper", "bb_middle", "bb_lower", "bb"):
            window = params.get("window", params.get("period", 20))
            std_dev = params.get("std_dev", params.get("window_dev", 2.0))
            bb = ta.volatility.BollingerBands(
                close, window=int(window), window_dev=float(std_dev)
            )
            if name == "bb_upper":
                return bb.bollinger_hband()
            if name == "bb_lower":
                return bb.bollinger_lband()
            if name == "bb":
                return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
            return bb.bollinger_mavg()

        if name == "atr":
            period = params.get("window", params.get("period", 14))
            return ta.volatility.AverageTrueRange(
                high, low, close, window=int(period)
            ).average_true_range()

        # -- Volume indicators --
        if name == "obv":
            return ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        if name == "vwap":
            return ta.volume.VolumeWeightedAveragePrice(
                high, low, close, volume
            ).volume_weighted_average_price()

        if name == "volume_sma":
            window = params.get("window", params.get("period", 20))
            return ta.trend.SMAIndicator(volume, window=int(window)).sma_indicator()

        raise ValueError(f"Unknown indicator: {indicator_name!r}")

    # ------------------------------------------------------------------ #
    #  Risk management
    # ------------------------------------------------------------------ #

    def _check_risk(self) -> bool:
        """Check whether the portfolio has breached the max drawdown limit.

        Updates the peak portfolio value and compares current value to the
        peak.  If the drawdown exceeds ``max_drawdown``, logs a warning
        and returns ``False`` to signal that no new trades should be
        opened.

        Returns
        -------
        bool
            ``True`` if risk limits are respected, ``False`` otherwise.
        """
        current_value = self.portfolio_value
        if current_value > self.vars.peak_value:
            self.vars.peak_value = current_value

        drawdown = (
            (self.vars.peak_value - current_value) / self.vars.peak_value
            if self.vars.peak_value > 0
            else 0.0
        )

        if drawdown >= self.vars.max_drawdown:
            logger.warning(
                "Max drawdown breached: {:.2%} >= {:.2%} — halting new trades",
                drawdown,
                self.vars.max_drawdown,
            )
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Trade logging
    # ------------------------------------------------------------------ #

    def log_trade(
        self, side: str, symbol: str, quantity: float, price: float
    ) -> None:
        """Log a trade to the logger and update internal counters.

        Parameters
        ----------
        side :
            ``"buy"`` or ``"sell"``.
        symbol :
            Trading symbol (e.g. ``"BTC"``).
        quantity :
            Number of units traded.
        price :
            Execution price.
        """
        self.vars.trade_count += 1
        notional = quantity * price
        logger.info(
            "TRADE #{} | {} {} {} @ ${:,.2f} (notional ${:,.2f})",
            self.vars.trade_count,
            side.upper(),
            quantity,
            symbol,
            price,
            notional,
        )

    # ------------------------------------------------------------------ #
    #  Abstract hook
    # ------------------------------------------------------------------ #

    @abstractmethod
    def on_trading_iteration(self) -> None:
        """Main trading logic — must be overridden by subclasses."""
        raise NotImplementedError
