"""Bollinger Bands mean-reversion strategy with volume confirmation."""

from loguru import logger
from lumibot.entities import Asset

from .base import BaseStrategy


class BBandsStrategy(BaseStrategy):
    """Bollinger Bands mean-reversion strategy.

    Entry signal
    ------------
    * Price touches or closes below the **lower** Bollinger Band
    * AND current volume > ``volume_factor`` × average volume
      (confirms the move is significant, not just low-liquidity noise)

    Exit signals
    ------------
    * Price touches or closes above the **upper** Bollinger Band
      (mean-reversion target reached)
    * OR a hard stop-loss triggers: price drops ≥ 2 % below entry price

    Parameters
    ----------
    bb_period : int
        Lookback period for the Bollinger Bands SMA (default 20).
    bb_std_dev : float
        Number of standard deviations for the bands (default 2.0).
    volume_factor : float
        Volume multiplier above average required to confirm entry
        (default 1.5).
    """

    STOP_LOSS_PCT = 0.02  # 2 % hard stop

    parameters = {
        **BaseStrategy.parameters,
        "bb_period": 20,
        "bb_std_dev": 2.0,
        "volume_factor": 1.5,
    }

    def initialize(self) -> None:
        """Initialise BB and volume parameters."""
        super().initialize()

        self.vars.bb_period = self.parameters["bb_period"]
        self.vars.bb_std_dev = self.parameters["bb_std_dev"]
        self.vars.volume_factor = self.parameters["volume_factor"]

        logger.info(
            "BBandsStrategy params: period={}, std_dev={}, vol_factor={}",
            self.vars.bb_period,
            self.vars.bb_std_dev,
            self.vars.volume_factor,
        )

    def on_trading_iteration(self) -> None:
        """Execute one trading iteration."""
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

        min_rows = self.vars.bb_period + 5
        if len(df) < min_rows:
            logger.warning(
                "Insufficient data: {} rows < {}", len(df), min_rows
            )
            return

        # ---- Compute Bollinger Bands ----
        bb_upper, bb_middle, bb_lower = self._get_indicator(
            df, "bb", window=self.vars.bb_period, std_dev=self.vars.bb_std_dev
        )

        # ---- Volume SMA ----
        vol_sma = self._get_indicator(df, "volume_sma", window=self.vars.bb_period)

        # Latest values
        price = self.get_last_price(base_asset, quote=quote)
        if price is None:
            logger.warning("Cannot get last price for {}", symbol)
            return

        upper = float(bb_upper.iloc[-1])
        middle = float(bb_middle.iloc[-1])
        lower = float(bb_lower.iloc[-1])
        avg_vol = float(vol_sma.iloc[-1])
        current_vol = float(df["volume"].iloc[-1])

        dt = self.get_datetime()
        logger.debug(
            "[{}] {} price={:.2f} BB=[{:.2f}, {:.2f}, {:.2f}] "
            "vol={:.0f} avg_vol={:.0f}",
            dt,
            symbol,
            price,
            lower,
            middle,
            upper,
            current_vol,
            avg_vol,
        )

        # ---- Position check ----
        position = self.get_position(base_asset)
        has_position = position is not None and position.quantity != 0

        # ---- ENTRY: price touches lower band + volume spike ----
        volume_spike = current_vol > (self.vars.volume_factor * avg_vol)
        touches_lower = price <= lower

        if touches_lower and volume_spike and not has_position:
            if not self._check_risk():
                return

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
                "BUY signal: price {:.2f} <= lower {:.2f}, "
                "vol {:.0f} > {:.0f}×avg",
                price,
                lower,
                current_vol,
                self.vars.volume_factor,
            )

        # ---- EXIT conditions ----
        if has_position:
            stop_price = self.vars.entry_price * (1 - self.STOP_LOSS_PCT)
            touches_upper = price >= upper

            qty = abs(position.quantity)

            if touches_upper:
                order = self.create_order(
                    asset=base_asset,
                    quantity=qty,
                    side="sell",
                    quote=quote,
                )
                self.submit_order(order)
                self.log_trade("sell", symbol, qty, price)
                logger.info(
                    "SELL signal: price {:.2f} >= upper {:.2f} "
                    "(mean-reversion target)",
                    price,
                    upper,
                )
                self.vars.entry_price = None

            elif price <= stop_price:
                order = self.create_order(
                    asset=base_asset,
                    quantity=qty,
                    side="sell",
                    quote=quote,
                )
                self.submit_order(order)
                pnl_pct = (price - self.vars.entry_price) / self.vars.entry_price
                self.log_trade("sell", symbol, qty, price)
                logger.warning(
                    "STOP-LOSS triggered: price {:.2f} <= stop {:.2f} "
                    "({:.2%} loss)",
                    price,
                    stop_price,
                    pnl_pct,
                )
                self.vars.entry_price = None

    def generate_signal(
        self, price: float, indicators: dict, market_data: dict
    ) -> str:
        """Generate Bollinger Bands signal (BUY, SELL, or HOLD)."""
        bb_upper = indicators.get("bb_upper", price * 1.02)
        bb_lower = indicators.get("bb_lower", price * 0.98)
        volume_high = indicators.get("volume_high", False)

        if price <= bb_lower and volume_high:
            return "BUY"
        elif price >= bb_upper:
            return "SELL"

        return "HOLD"
