from __future__ import annotations



from dataclasses import dataclass

from datetime import datetime, timezone

from typing import Any



from .config import TradingConfig

from .risk_manager import OrderPlan

from .signal_engine import TradeSignal

from .state import BotState





@dataclass(frozen=True)

class PositionSnapshot:

    # Legacy field names. Values are interpreted in the configured quote currency.

    quote_balance_krw: float

    base_balance_eth: float

    last_price_krw: float

    current_position_krw: float



    @property

    def quote_balance(self) -> float:

        return self.quote_balance_krw



    @property

    def base_balance(self) -> float:

        return self.base_balance_eth



    @property

    def last_price(self) -> float:

        return self.last_price_krw



    @property

    def current_position_value(self) -> float:

        return self.current_position_krw





@dataclass(frozen=True)

class ExitPlan:

    action: str  # SELL or HOLD

    amount_eth: float

    reason: str





def build_position_snapshot(quote_balance: float, base_balance: float, last_price: float) -> PositionSnapshot:

    return PositionSnapshot(

        quote_balance_krw=float(quote_balance),

        base_balance_eth=float(base_balance),

        last_price_krw=float(last_price),

        current_position_krw=float(base_balance) * float(last_price),

    )





def calculate_unrealized_pnl(state: BotState, position: PositionSnapshot) -> tuple[float, float]:

    """Return unrealized PnL in quote currency and percent for the bot-managed spot long."""

    if not state.position_open or not state.entry_price_krw or not state.entry_amount_eth:

        return 0.0, 0.0

    entry_value = float(state.entry_price_krw) * float(state.entry_amount_eth)

    current_value = float(position.last_price) * float(state.entry_amount_eth)

    pnl_quote = current_value - entry_value

    pnl_pct = pnl_quote / entry_value if entry_value > 0 else 0.0

    return float(pnl_quote), float(pnl_pct)





def apply_position_limits(config: TradingConfig, state: BotState, plan: OrderPlan, position: PositionSnapshot) -> OrderPlan:

    if plan.action != "BUY":

        return plan



    if state.position_open and not config.allow_pyramiding:

        return OrderPlan(

            action="HOLD",

            amount=0.0,

            notional_quote=0.0,

            stop_loss=None,

            take_profit=None,

            reason="existing long position is open; pyramiding is disabled",

        )



    remaining_capacity = config.max_position_quote - position.current_position_value

    if remaining_capacity < config.min_order_quote:

        return OrderPlan(

            action="HOLD",

            amount=0.0,

            notional_quote=0.0,

            stop_loss=None,

            take_profit=None,

            reason=(

                f"position limit reached: current_position={position.current_position_value:g} "

                f"{config.quote_currency}, max_position={config.max_position_quote:g} {config.quote_currency}"

            ),

        )



    if plan.notional_quote > remaining_capacity:

        adjusted_notional = float(int(remaining_capacity)) if config.quote_currency == "KRW" else float(round(remaining_capacity, 8))

        if adjusted_notional < config.min_order_quote:

            return OrderPlan("HOLD", 0.0, 0.0, None, None, "remaining position capacity is below minimum order")

        return OrderPlan(

            action="BUY",

            amount=adjusted_notional / max(position.last_price, 1e-12),

            notional_quote=adjusted_notional,

            stop_loss=plan.stop_loss,

            take_profit=plan.take_profit,

            reason=plan.reason + f"; order capped by max_position to {adjusted_notional:g} {config.quote_currency}",

        )



    return plan





def build_exit_plan(config: TradingConfig, state: BotState, signal: TradeSignal, position: PositionSnapshot) -> ExitPlan:

    if not config.allow_exit_sell:

        return ExitPlan("HOLD", 0.0, "exit sell disabled")

    if position.base_balance_eth <= 0:

        return ExitPlan("HOLD", 0.0, "no ETH balance to exit")

    if not state.position_open:

        return ExitPlan("HOLD", 0.0, "no bot-managed open position")



    price = position.last_price

    if state.stop_loss_krw is not None and price <= state.stop_loss_krw:

        return ExitPlan("SELL", position.base_balance_eth, f"stop loss triggered: price={price:g} <= {state.stop_loss_krw:g}")

    if state.take_profit_krw is not None and price >= state.take_profit_krw:

        return ExitPlan("SELL", position.base_balance_eth, f"take profit triggered: price={price:g} >= {state.take_profit_krw:g}")

    if config.exit_on_downside_signal and signal.raw_direction == "DOWN":

        return ExitPlan("SELL", position.base_balance_eth, "downside signal exit enabled and raw_direction=DOWN")



    return ExitPlan("HOLD", 0.0, "position hold")





def record_entry(state: BotState, plan: OrderPlan, order: dict[str, Any] | None, price_krw: float) -> None:

    # BotState keeps legacy *_krw names. Values are in the configured quote currency.

    state.position_open = True

    state.entry_price_krw = float(price_krw)

    state.entry_amount_eth = float(plan.amount)

    state.entry_notional_krw = float(plan.notional_quote)

    state.stop_loss_krw = plan.stop_loss

    state.take_profit_krw = plan.take_profit

    state.position_opened_at = datetime.now(timezone.utc).isoformat()

    state.last_order_id = order.get("id") if isinstance(order, dict) else None

    state.last_exit_order_id = None

    state.last_exit_reason = None





def record_exit(state: BotState, order: dict[str, Any] | None, reason: str) -> None:

    state.position_open = False

    state.entry_price_krw = None

    state.entry_amount_eth = None

    state.entry_notional_krw = None

    state.stop_loss_krw = None

    state.take_profit_krw = None

    state.position_opened_at = None

    state.last_exit_order_id = order.get("id") if isinstance(order, dict) else None

    state.last_exit_reason = reason

