"""SMA crossover + RSI filter strategy."""

from loguru import logger

from .base import BaseStrategy


class SMACrossStrategy(BaseStrategy):
    """Dual SMA crossover strategy with RSI confirmation.

    Entry signal
    ------------
    * Fast SMA crosses **above** slow SMA (bullish crossover)
    * AND RSI is below the overbought threshold (room to run)

    Exit signal
    -----------
    * Fast SMA crosses **below** slow SMA (bearish crossover)
    * OR RSI exceeds the overbought threshold (take profits)

    Parameters
    ----------
    sma_fast : int
        Period for the fast moving average (default 20).
    sma_slow : int
        Period for the slow moving average (default 50).
    rsi_period : int
        RSI lookback window (default 14).
    rsi_overbought : float
        RSI level considered overbought (default 70).
    rsi_oversold : float
        RSI level considered oversold (default 30).
    """

    parameters = {
        **BaseStrategy.parameters,
        "sma_fast": 20,
        "sma_slow": 50,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
    }

    def initialize(self) -> None:
        """Initialise SMA/RSI parameters on top of the base config."""
        super().initialize()

        self.vars.sma_fast = self.parameters["sma_fast"]
        self.vars.sma_slow = self.parameters["sma_slow"]
        self.vars.rsi_period = self.parameters["rsi_period"]
        self.vars.rsi_overbought = self.parameters["rsi_overbought"]
        self.vars.rsi_oversold = self.parameters["rsi_oversold"]

        # Track previous indicator values for crossover detection
        self.vars.prev_sma_fast = None
        self.vars.prev_sma_slow = None

        logger.info(
            "SMACrossStrategy params: fast={}, slow={}, rsi_period={}, "
            "overbought={}, oversold={}",
            self.vars.sma_fast,
            self.vars.sma_slow,
            self.vars.rsi_period,
            self.vars.rsi_overbought,
            self.vars.rsi_oversold,
        )

    def on_trading_iteration(self) -> None:
        """Execute one trading iteration: evaluate signals and trade."""
        symbol = self.vars.symbol
        base_asset = self.vars.base_asset
        quote = self.vars.quote

        # ---- Get historical prices ----
        bars = self.get_historical_prices(
            base_asset, 100, "minute", quote=quote
        )
        if bars is None or bars.df is None or bars.df.empty:
            logger.warning("No historical data for {}", symbol)
            return

        df = bars.df

        # Need enough rows for slow SMA + RSI
        min_rows = max(self.vars.sma_slow, self.vars.rsi_period) + 2
        if len(df) < min_rows:
            logger.warning(
                "Insufficient data: {} rows < {}", len(df), min_rows
            )
            return

        # ---- Compute indicators ----
        sma_fast_series = self._get_indicator(
            df, "sma", window=self.vars.sma_fast
        )
        sma_slow_series = self._get_indicator(
            df, "sma", window=self.vars.sma_slow
        )
        rsi_series = self._get_indicator(
            df, "rsi", window=self.vars.rsi_period
        )

        # Latest values
        sma_fast = float(sma_fast_series.iloc[-1])
        sma_slow = float(sma_slow_series.iloc[-1])
        rsi = float(rsi_series.iloc[-1])

        # Previous values (for crossover detection)
        prev_fast = float(sma_fast_series.iloc[-2])
        prev_slow = float(sma_slow_series.iloc[-2])

        price = self.get_last_price(base_asset, quote=quote)
        if price is None:
            logger.warning("Cannot get last price for {}", symbol)
            return

        # Store for next iteration
        self.vars.prev_sma_fast = sma_fast
        self.vars.prev_sma_slow = sma_slow

        # ---- Position check ----
        position = self.get_position(base_asset)
        has_position = position is not None and position.quantity != 0

        # ---- Signal logic ----
        bullish_cross = (prev_fast <= prev_slow) and (sma_fast > sma_slow)
        bearish_cross = (prev_fast >= prev_slow) and (sma_fast < sma_slow)
        rsi_overbought = rsi > self.vars.rsi_overbought
        rsi_safe = rsi < self.vars.rsi_overbought

        dt = self.get_datetime()

        logger.debug(
            "[{}] {} price={:.2f} sma_fast={:.2f} sma_slow={:.2f} rsi={:.1f}",
            dt,
            symbol,
            price,
            sma_fast,
            sma_slow,
            rsi,
        )

        # -- ENTRY: bullish cross + RSI not overbought --
        if bullish_cross and rsi_safe and not has_position:
            if not self._check_risk():
                return

            # Size position using risk_pct of portfolio
            allocation = self.portfolio_value * self.vars.risk_pct
            quantity = allocation / price

            order = self.create_order(
                asset=base_asset,
                quantity=quantity,
                side="buy",
                quote=quote,
            )
            self.submit_order(order)
            self.vars.entry_price = price
            self.log_trade("buy", symbol, quantity, price)
            logger.info(
                "BUY signal: bullish SMA cross + RSI {:.1f} < {}",
                rsi,
                self.vars.rsi_overbought,
            )

        # -- EXIT: bearish cross OR RSI overbought --
        elif (bearish_cross or rsi_overbought) and has_position:
            qty = abs(position.quantity)

            order = self.create_order(
                asset=base_asset,
                quantity=qty,
                side="sell",
                quote=quote,
            )
            self.submit_order(order)
            self.log_trade("sell", symbol, qty, price)
            reason = "bearish SMA cross" if bearish_cross else "RSI overbought"
            logger.info(
                "SELL signal: {} (RSI={:.1f})", reason, rsi
            )
            self.vars.entry_price = None

    def generate_signal(
        self, price: float, indicators: dict, market_data: dict
    ) -> str:
        """Generate dual SMA crossover strategy with RSI filter signal."""
        rsi = indicators.get("rsi")
        sma_fast = indicators.get("sma_fast")
        sma_slow = indicators.get("sma_slow")

        if rsi is None or sma_fast is None or sma_slow is None:
            return "HOLD"

        # Crossover logic needs history
        prev_fast = getattr(self, "_prev_sma_fast", None)
        prev_slow = getattr(self, "_prev_sma_slow", None)

        self._prev_sma_fast = sma_fast
        self._prev_sma_slow = sma_slow

        if prev_fast is None or prev_slow is None:
            return "HOLD"

        bullish_cross = (prev_fast <= prev_slow) and (sma_fast > sma_slow)
        bearish_cross = (prev_fast >= prev_slow) and (sma_fast < sma_slow)

        rsi_overbought = rsi > self.parameters.get("rsi_overbought", 70)
        rsi_safe = rsi < self.parameters.get("rsi_overbought", 70)

        if bullish_cross and rsi_safe:
            return "BUY"
        elif bearish_cross or rsi_overbought:
            return "SELL"

        return "HOLD"
