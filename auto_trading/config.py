"""Configuration for the Upbit long-only automated trading runtime.



Policy in this version:

- Upbit spot market only

- ETH/KRW and ETH/USDT supported

- 4-hour closed-candle decision cycle

- long entries only; no short selling

- optional sell orders are exits of an existing long spot position, not shorts



Backward compatibility note:

The original code used *_krw field names. Those dataclass fields remain so that

older modules/tests do not break, but their value now means "quote currency

amount". Use the *_quote properties in new code.

"""

from __future__ import annotations



import os

from dataclasses import dataclass

from dotenv import load_dotenv





SUPPORTED_SYMBOLS = {"ETH/KRW", "ETH/USDT"}

SUPPORTED_QUOTES = {"KRW", "USDT"}





@dataclass(frozen=True)

class TradingConfig:

    exchange_id: str = "upbit"

    market_type: str = "spot"

    symbol: str = "ETH/KRW"

    timeframe: str = "4h"

    quote_currency: str = "KRW"

    base_currency: str = "ETH"



    dry_run: bool = True

    use_testnet: bool = False

    live_confirmation: str = "NO"



    # Legacy field names. Values are interpreted as quote-currency amounts.

    base_order_krw: float = 10000.0

    max_order_krw: float = 30000.0

    min_order_krw: float = 5000.0

    max_position_krw: float = 50000.0

    dry_run_balance_quote: float = 1000000.0



    risk_per_trade: float = 0.01

    max_account_allocation: float = 0.20

    min_confidence: float = 0.58

    probability_threshold: float = 0.10



    atr_stop_multiplier: float = 1.5

    take_profit_r_multiple: float = 2.0

    allow_short: bool = False

    allow_exit_sell: bool = True

    allow_pyramiding: bool = False

    exit_on_downside_signal: bool = False



    lookback_candles: int = 120

    poll_seconds: int = 60

    use_candle_scheduler: bool = True

    scheduler_delay_seconds: int = 30

    state_file: str = "auto_trading_state.json"

    trade_log_file: str = "auto_trading_trades.csv"

    runtime_log_file: str = "auto_trading_runtime.log"



    google_model: str = "gemini-2.5-flash"

    use_llm: bool = True

    gcp_project_id: str = ""

    gcp_location: str = "us-central1"



    @property

    def live_enabled(self) -> bool:

        return (not self.dry_run) and self.live_confirmation == "YES"



    @property

    def base_order_quote(self) -> float:

        return self.base_order_krw



    @property

    def max_order_quote(self) -> float:

        return self.max_order_krw



    @property

    def min_order_quote(self) -> float:

        return self.min_order_krw



    @property

    def max_position_quote(self) -> float:

        return self.max_position_krw





def _env_bool(name: str, default: bool) -> bool:

    raw = os.getenv(name)

    if raw is None:

        return default

    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}





def _env_float(name: str, default: float) -> float:

    raw = os.getenv(name)

    if raw is None or raw == "":

        return default

    return float(raw)





def _env_float_any(names: list[str], default: float) -> float:

    for name in names:

        raw = os.getenv(name)

        if raw is not None and raw != "":

            return float(raw)

    return default





def _env_int(name: str, default: int) -> int:

    raw = os.getenv(name)

    if raw is None or raw == "":

        return default

    return int(raw)





def _split_symbol(symbol: str) -> tuple[str, str]:

    if "/" not in symbol:

        raise ValueError("AUTO_TRADING_SYMBOL must use BASE/QUOTE format, e.g. ETH/KRW or ETH/USDT.")

    base, quote = symbol.split("/", 1)

    return base.upper(), quote.upper()





def _quote_defaults(quote_currency: str) -> dict[str, float | str]:

    if quote_currency == "USDT":

        return {

            "base_order": 10.0,

            "max_order": 30.0,

            "min_order": 0.5,

            "max_position": 50.0,

            "dry_run_balance": 1000.0,

            "state_file": "auto_trading_state_usdt.json",

            "trade_log_file": "auto_trading_trades_usdt.csv",

            "runtime_log_file": "auto_trading_runtime_usdt.log",

        }

    return {

        "base_order": 10000.0,

        "max_order": 30000.0,

        "min_order": 5000.0,

        "max_position": 50000.0,

        "dry_run_balance": 1000000.0,

        "state_file": "auto_trading_state.json",

        "trade_log_file": "auto_trading_trades.csv",

        "runtime_log_file": "auto_trading_runtime.log",

    }





def validate_config(config: TradingConfig) -> None:

    """Fail fast on unsafe or inconsistent runtime configuration."""

    if config.exchange_id != "upbit":

        raise ValueError("This capstone bot version is restricted to Upbit only.")

    if config.market_type != "spot":

        raise ValueError("This capstone bot version supports spot trading only.")

    if config.symbol not in SUPPORTED_SYMBOLS:

        raise ValueError("This capstone bot version supports ETH/KRW or ETH/USDT only.")



    symbol_base, symbol_quote = _split_symbol(config.symbol)

    if config.base_currency != symbol_base or config.quote_currency != symbol_quote:

        raise ValueError(

            f"Currency configuration must match symbol. symbol={config.symbol}, "

            f"base={config.base_currency}, quote={config.quote_currency}"

        )

    if config.base_currency != "ETH" or config.quote_currency not in SUPPORTED_QUOTES:

        raise ValueError("Currency configuration must be base=ETH and quote=KRW or USDT.")

    if config.allow_short:

        raise ValueError("Short trading is not allowed in this project.")



    minimum_allowed = 0.5 if config.quote_currency == "USDT" else 5000.0

    if config.min_order_quote < minimum_allowed:

        raise ValueError(

            f"Upbit {config.quote_currency} minimum order should be at least "

            f"{minimum_allowed:g} {config.quote_currency}."

        )

    if config.base_order_quote < config.min_order_quote:

        raise ValueError("AUTO_TRADING_BASE_ORDER_QUOTE must be >= AUTO_TRADING_MIN_ORDER_QUOTE.")

    if config.max_order_quote < config.min_order_quote:

        raise ValueError("AUTO_TRADING_MAX_ORDER_QUOTE must be >= AUTO_TRADING_MIN_ORDER_QUOTE.")

    if config.max_position_quote < config.min_order_quote:

        raise ValueError("AUTO_TRADING_MAX_POSITION_QUOTE must be >= AUTO_TRADING_MIN_ORDER_QUOTE.")

    if config.dry_run_balance_quote <= 0:

        raise ValueError("AUTO_TRADING_DRY_RUN_BALANCE_QUOTE must be positive.")

    if not 0 < config.risk_per_trade <= 0.05:

        raise ValueError("AUTO_TRADING_RISK_PER_TRADE should be > 0 and <= 0.05.")

    if not 0 < config.max_account_allocation <= 1:

        raise ValueError("AUTO_TRADING_MAX_ACCOUNT_ALLOCATION should be > 0 and <= 1.")

    if not 0 <= config.probability_threshold <= 1:

        raise ValueError("AUTO_TRADING_PROBABILITY_THRESHOLD must be between 0 and 1.")

    if not 0 <= config.min_confidence <= 1:

        raise ValueError("AUTO_TRADING_MIN_CONFIDENCE must be between 0 and 1.")

    if config.atr_stop_multiplier <= 0:

        raise ValueError("AUTO_TRADING_ATR_STOP_MULTIPLIER must be positive.")

    if config.take_profit_r_multiple <= 0:

        raise ValueError("AUTO_TRADING_TAKE_PROFIT_R_MULTIPLE must be positive.")

    if config.poll_seconds < 10:

        raise ValueError("AUTO_TRADING_POLL_SECONDS should be at least 10 seconds.")

    if config.timeframe != "4h":

        raise ValueError("This capstone bot version is intentionally fixed to 4h candles.")

    if config.live_enabled:

        if not os.getenv("UPBIT_ACCESS_KEY"):

            raise ValueError("UPBIT_ACCESS_KEY is required for live trading.")

        if not os.getenv("UPBIT_SECRET_KEY"):

            raise ValueError("UPBIT_SECRET_KEY is required for live trading.")





def load_config() -> TradingConfig:

    load_dotenv()



    symbol = os.getenv("AUTO_TRADING_SYMBOL", "ETH/KRW").upper()

    symbol_base, symbol_quote = _split_symbol(symbol)

    base_currency = os.getenv("AUTO_TRADING_BASE_CURRENCY", symbol_base).upper()

    quote_currency = os.getenv("AUTO_TRADING_QUOTE_CURRENCY", symbol_quote).upper()

    defaults = _quote_defaults(quote_currency)



    config = TradingConfig(

        exchange_id=os.getenv("AUTO_TRADING_EXCHANGE", "upbit"),

        market_type=os.getenv("AUTO_TRADING_MARKET_TYPE", "spot"),

        symbol=symbol,

        timeframe=os.getenv("AUTO_TRADING_TIMEFRAME", "4h"),

        quote_currency=quote_currency,

        base_currency=base_currency,

        dry_run=_env_bool("AUTO_TRADING_DRY_RUN", True),

        use_testnet=_env_bool("AUTO_TRADING_USE_TESTNET", False),

        live_confirmation=os.getenv("AUTO_TRADING_CONFIRM_LIVE", "NO"),

        base_order_krw=_env_float_any(

            ["AUTO_TRADING_BASE_ORDER_QUOTE", "AUTO_TRADING_BASE_ORDER_KRW"],

            float(defaults["base_order"]),

        ),

        max_order_krw=_env_float_any(

            ["AUTO_TRADING_MAX_ORDER_QUOTE", "AUTO_TRADING_MAX_ORDER_KRW"],

            float(defaults["max_order"]),

        ),

        min_order_krw=_env_float_any(

            ["AUTO_TRADING_MIN_ORDER_QUOTE", "AUTO_TRADING_MIN_ORDER_KRW"],

            float(defaults["min_order"]),

        ),

        max_position_krw=_env_float_any(

            ["AUTO_TRADING_MAX_POSITION_QUOTE", "AUTO_TRADING_MAX_POSITION_KRW"],

            float(defaults["max_position"]),

        ),

        dry_run_balance_quote=_env_float_any(

            ["AUTO_TRADING_DRY_RUN_BALANCE_QUOTE", f"AUTO_TRADING_DRY_RUN_BALANCE_{quote_currency}"],

            float(defaults["dry_run_balance"]),

        ),

        risk_per_trade=_env_float("AUTO_TRADING_RISK_PER_TRADE", 0.01),

        max_account_allocation=_env_float("AUTO_TRADING_MAX_ACCOUNT_ALLOCATION", 0.20),

        min_confidence=_env_float("AUTO_TRADING_MIN_CONFIDENCE", 0.58),

        probability_threshold=_env_float("AUTO_TRADING_PROBABILITY_THRESHOLD", 0.10),

        atr_stop_multiplier=_env_float("AUTO_TRADING_ATR_STOP_MULTIPLIER", 1.5),

        take_profit_r_multiple=_env_float("AUTO_TRADING_TAKE_PROFIT_R_MULTIPLE", 2.0),

        allow_short=False,

        allow_exit_sell=_env_bool("AUTO_TRADING_ALLOW_EXIT_SELL", True),

        allow_pyramiding=_env_bool("AUTO_TRADING_ALLOW_PYRAMIDING", False),

        exit_on_downside_signal=_env_bool("AUTO_TRADING_EXIT_ON_DOWNSIDE_SIGNAL", False),

        lookback_candles=_env_int("AUTO_TRADING_LOOKBACK_CANDLES", 120),

        poll_seconds=_env_int("AUTO_TRADING_POLL_SECONDS", 60),

        use_candle_scheduler=_env_bool("AUTO_TRADING_USE_CANDLE_SCHEDULER", True),

        scheduler_delay_seconds=_env_int("AUTO_TRADING_SCHEDULER_DELAY_SECONDS", 30),

        state_file=os.getenv("AUTO_TRADING_STATE_FILE", str(defaults["state_file"])),

        trade_log_file=os.getenv("AUTO_TRADING_TRADE_LOG_FILE", str(defaults["trade_log_file"])),

        runtime_log_file=os.getenv("AUTO_TRADING_RUNTIME_LOG_FILE", str(defaults["runtime_log_file"])),

        google_model=os.getenv("AUTO_TRADING_GOOGLE_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),

        use_llm=_env_bool("AUTO_TRADING_USE_LLM", True),

        gcp_project_id=os.getenv("AUTO_TRADING_GCP_PROJECT_ID") or os.getenv("GCP_PROJECT_ID", ""),

        gcp_location=os.getenv("AUTO_TRADING_GCP_LOCATION") or os.getenv("GCP_LOCATION", "us-central1"),

    )

    validate_config(config)

    return config

