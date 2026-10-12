from auto_trading.config import TradingConfig

from auto_trading.risk_manager import build_order_plan

from auto_trading.signal_engine import TradeSignal





def _signal(decision="LONG", price=4_000_000.0, atr=50_000.0):

    return TradeSignal(

        decision=decision,

        raw_direction="UP" if decision == "LONG" else "NEUTRAL",

        up_probability=0.65,

        down_probability=0.35,

        confidence=0.7,

        reason="test",

        price=price,

        atr=atr,

        rsi=50.0,

        macd=1.0,

        support=price * 0.98,

        resistance=price * 1.02,

        llm_used=False,

        llm_failed=False,

        llm_error="",

        llm_raw_response="",

        signal_source="indicator_only",

    )





def test_non_long_signal_holds():

    config = TradingConfig()

    plan = build_order_plan(config, _signal(decision="HOLD"), quote_balance_krw=1_000_000)

    assert plan.action == "HOLD"

    assert plan.notional_krw == 0





def test_order_below_minimum_holds():

    config = TradingConfig(base_order_krw=5_000, max_order_krw=5_000, min_order_krw=5_000, risk_per_trade=0.0001)

    plan = build_order_plan(config, _signal(), quote_balance_krw=10_000)

    assert plan.action == "HOLD"

