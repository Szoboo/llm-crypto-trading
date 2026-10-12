from auto_trading.config import TradingConfig

from auto_trading.position_manager import apply_position_limits, build_exit_plan, build_position_snapshot

from auto_trading.risk_manager import OrderPlan

from auto_trading.signal_engine import TradeSignal

from auto_trading.state import BotState





def _signal(raw_direction="UP"):

    return TradeSignal(

        decision="LONG" if raw_direction == "UP" else "HOLD",

        raw_direction=raw_direction,

        up_probability=0.65,

        down_probability=0.35,

        confidence=0.7,

        reason="test",

        price=4_000_000.0,

        atr=50_000.0,

        rsi=50.0,

        macd=1.0,

        support=3_900_000.0,

        resistance=4_100_000.0,

        llm_used=False,

        llm_failed=False,

        llm_error="",

        llm_raw_response="",

        signal_source="indicator_only",

    )





def test_existing_position_blocks_buy_when_pyramiding_disabled():

    config = TradingConfig(allow_pyramiding=False)

    state = BotState(position_open=True)

    position = build_position_snapshot(100_000, 0.01, 4_000_000)

    plan = OrderPlan("BUY", 0.0025, 10_000, 3_900_000, 4_200_000, "test")

    result = apply_position_limits(config, state, plan, position)

    assert result.action == "HOLD"





def test_position_limit_blocks_buy():

    config = TradingConfig(max_position_krw=50_000)

    state = BotState(position_open=False)

    position = build_position_snapshot(100_000, 0.02, 4_000_000)

    plan = OrderPlan("BUY", 0.0025, 10_000, 3_900_000, 4_200_000, "test")

    result = apply_position_limits(config, state, plan, position)

    assert result.action == "HOLD"





def test_stop_loss_exit_is_sell_not_short():

    config = TradingConfig(allow_exit_sell=True)

    state = BotState(position_open=True, stop_loss_krw=3_900_000, take_profit_krw=4_300_000)

    position = build_position_snapshot(100_000, 0.01, 3_850_000)

    exit_plan = build_exit_plan(config, state, _signal(), position)

    assert exit_plan.action == "SELL"

    assert exit_plan.amount_eth == 0.01

