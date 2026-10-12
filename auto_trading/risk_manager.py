from __future__ import annotations



from dataclasses import dataclass



from .config import TradingConfig

from .signal_engine import TradeSignal





@dataclass(frozen=True)

class OrderPlan:

    action: str  # BUY, HOLD. SELL is not used unless optional exit mode is enabled elsewhere.

    amount: float  # estimated ETH amount for logs only

    notional_quote: float

    stop_loss: float | None

    take_profit: float | None

    reason: str



    @property

    def notional_krw(self) -> float:

        """Backward-compatible alias for older code/tests."""

        return self.notional_quote





def _round_notional(config: TradingConfig, notional: float) -> float:

    if config.quote_currency == "KRW":

        return float(int(notional))

    return float(round(notional, 8))





def build_order_plan(config: TradingConfig, signal: TradeSignal, quote_balance_krw: float) -> OrderPlan:

    """Build a long-entry order plan in the configured quote currency.



    The argument name quote_balance_krw is kept for backward compatibility. In

    USDT mode it is interpreted as quote_balance_usdt.

    """

    quote_balance = float(quote_balance_krw)

    if signal.decision != "LONG":

        return OrderPlan("HOLD", 0.0, 0.0, None, None, signal.reason)



    price = signal.price

    stop_distance = max(signal.atr * config.atr_stop_multiplier, price * 0.005)

    account_risk_quote = quote_balance * config.risk_per_trade

    risk_based_notional = account_risk_quote * price / stop_distance

    allocation_cap = quote_balance * config.max_account_allocation



    notional = min(

        config.base_order_quote,

        config.max_order_quote,

        allocation_cap,

        risk_based_notional,

    )



    if notional < config.min_order_quote:

        return OrderPlan(

            "HOLD",

            0.0,

            0.0,

            None,

            None,

            f"order amount {notional:g} {config.quote_currency} is below min_order {config.min_order_quote:g}",

        )



    notional = _round_notional(config, notional)

    amount = notional / price

    stop_loss = price - stop_distance

    take_profit = price + (stop_distance * config.take_profit_r_multiple)



    return OrderPlan(

        action="BUY",

        amount=float(amount),

        notional_quote=float(notional),

        stop_loss=float(stop_loss),

        take_profit=float(take_profit),

        reason=signal.reason,

    )

