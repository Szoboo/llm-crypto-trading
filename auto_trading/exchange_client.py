from __future__ import annotations



import os

from datetime import datetime, timedelta, timezone

from typing import Any



import ccxt

import pandas as pd



from .config import TradingConfig





_TIMEFRAME_SECONDS = {

    "1m": 60,

    "3m": 180,

    "5m": 300,

    "15m": 900,

    "30m": 1800,

    "1h": 3600,

    "4h": 14400,

    "1d": 86400,

}





class ExchangeClient:

    """Thin Upbit/ccxt adapter for spot trading in the configured quote market."""



    def __init__(self, config: TradingConfig):

        self.config = config

        exchange_cls = getattr(ccxt, config.exchange_id)



        api_key = os.getenv("UPBIT_ACCESS_KEY") or os.getenv("EXCHANGE_API_KEY")

        api_secret = os.getenv("UPBIT_SECRET_KEY") or os.getenv("EXCHANGE_API_SECRET")



        params: dict[str, Any] = {

            "apiKey": api_key,

            "secret": api_secret,

            "enableRateLimit": True,

            "options": {

                "defaultType": config.market_type,

                "createMarketBuyOrderRequiresPrice": False,

            },

        }

        self.exchange = exchange_cls(params)



        if config.use_testnet and hasattr(self.exchange, "set_sandbox_mode"):

            self.exchange.set_sandbox_mode(True)



        self.markets_loaded = False



    def load_markets(self) -> None:

        if not self.markets_loaded:

            self.exchange.load_markets()

            self.markets_loaded = True



    def _assert_allowed_market(self) -> None:

        if self.config.exchange_id != "upbit" or self.config.market_type != "spot":

            raise ValueError("This bot is restricted to Upbit spot trading.")

        if self.config.symbol not in {"ETH/KRW", "ETH/USDT"}:

            raise ValueError("This bot supports Upbit spot ETH/KRW or ETH/USDT only.")



    def fetch_ohlcv_df(self, limit: int) -> pd.DataFrame:

        self.load_markets()

        ohlcv = self.exchange.fetch_ohlcv(

            self.config.symbol,

            timeframe=self.config.timeframe,

            limit=limit,

        )

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        df.set_index("timestamp", inplace=True)

        return df.sort_index()



    def fetch_closed_ohlcv_df(self, limit: int) -> pd.DataFrame:

        """Return only fully closed candles."""

        df = self.fetch_ohlcv_df(limit=limit + 3)

        if df.empty:

            return df



        seconds = _TIMEFRAME_SECONDS.get(self.config.timeframe)

        if seconds is None:

            return df.iloc[:-1].tail(limit) if len(df) > 1 else df



        latest_start = df.index[-1].to_pydatetime()

        if latest_start.tzinfo is None:

            latest_start = latest_start.replace(tzinfo=timezone.utc)

        latest_close = latest_start + timedelta(seconds=seconds)

        now = datetime.now(timezone.utc)



        if latest_close > now:

            df = df.iloc[:-1]

        return df.tail(limit)



    def fetch_last_price(self) -> float:

        ticker = self.exchange.fetch_ticker(self.config.symbol)

        price = ticker.get("last") or ticker.get("close")

        if price is None:

            raise RuntimeError(f"No last price returned for {self.config.symbol}")

        return float(price)



    def fetch_quote_balance(self) -> float:

        if self.config.dry_run:

            return float(self.config.dry_run_balance_quote)

        balance = self.exchange.fetch_balance()

        quote = self.config.quote_currency

        free = balance.get(quote, {}).get("free") if isinstance(balance.get(quote), dict) else None

        if free is None:

            free = balance.get("free", {}).get(quote, 0)

        return float(free or 0)



    def fetch_base_balance(self) -> float:

        if self.config.dry_run:

            return float(os.getenv("AUTO_TRADING_DRY_RUN_BASE_BALANCE", "0"))

        balance = self.exchange.fetch_balance()

        base = self.config.base_currency

        free = balance.get(base, {}).get("free") if isinstance(balance.get(base), dict) else None

        if free is None:

            free = balance.get("free", {}).get(base, 0)

        return float(free or 0)



    def amount_to_precision(self, amount: float) -> float:

        self.load_markets()

        return float(self.exchange.amount_to_precision(self.config.symbol, amount))



    def price_to_precision(self, price: float) -> float:

        self.load_markets()

        return float(self.exchange.price_to_precision(self.config.symbol, price))



    def quote_to_precision(self, notional_quote: float) -> float:

        if self.config.quote_currency == "KRW":

            return float(int(notional_quote))

        try:

            self.load_markets()

            if hasattr(self.exchange, "cost_to_precision"):

                return float(self.exchange.cost_to_precision(self.config.symbol, notional_quote))

        except Exception:

            pass

        return float(round(float(notional_quote), 8))



    def create_market_buy_quote(self, notional_quote: float, quote_balance: float | None = None) -> dict[str, Any]:

        self._assert_allowed_market()

        notional_quote = self.quote_to_precision(notional_quote)

        if notional_quote < self.config.min_order_quote:

            raise ValueError(

                f"{self.config.quote_currency} order amount {notional_quote:g} is below minimum "

                f"{self.config.min_order_quote:g}"

            )

        if quote_balance is not None and quote_balance < notional_quote:

            raise ValueError(

                f"Insufficient {self.config.quote_currency} balance: "

                f"{quote_balance:g} < {notional_quote:g}"

            )



        if self.config.dry_run or not self.config.live_enabled:

            result = {

                "id": "DRY_RUN_BUY",

                "symbol": self.config.symbol,

                "type": "market",

                "side": "buy",

                "notional_quote": notional_quote,

                "quote_currency": self.config.quote_currency,

                "status": "simulated",

            }

            if self.config.quote_currency == "KRW":

                result["notional_krw"] = notional_quote

            return result



        # Upbit market-buy uses the amount argument as quote-currency cost when

        # createMarketBuyOrderRequiresPrice is disabled in CCXT options.

        return self.exchange.create_order(self.config.symbol, "market", "buy", notional_quote, None)



    def create_market_buy_krw(self, notional_krw: float, quote_balance_krw: float | None = None) -> dict[str, Any]:

        """Backward-compatible wrapper. Prefer create_market_buy_quote in new code."""

        return self.create_market_buy_quote(notional_krw, quote_balance=quote_balance_krw)



    def create_market_sell_base(self, amount: float) -> dict[str, Any]:

        self._assert_allowed_market()

        amount = self.amount_to_precision(amount)

        if amount <= 0:

            raise ValueError("Sell amount became zero after exchange precision rounding")



        if self.config.dry_run or not self.config.live_enabled:

            return {

                "id": "DRY_RUN_SELL",

                "symbol": self.config.symbol,

                "type": "market",

                "side": "sell",

                "amount": amount,

                "base_currency": self.config.base_currency,

                "status": "simulated",

            }



        return self.exchange.create_order(self.config.symbol, "market", "sell", amount, None)

