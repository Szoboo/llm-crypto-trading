from auto_trading.config import TradingConfig, validate_config

from auto_trading.risk_manager import build_order_plan

from auto_trading.signal_engine import TradeSignal





def _signal(price=3000.0, atr=50.0):

    return TradeSignal(

        decision="LONG",

        raw_direction="UP",

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





def test_usdt_config_validates():

    config = TradingConfig(

        symbol="ETH/USDT",

        quote_currency="USDT",

        base_currency="ETH",

        base_order_krw=10.0,

        max_order_krw=30.0,

        min_order_krw=0.5,

        max_position_krw=50.0,

        dry_run_balance_quote=1000.0,

    )

    validate_config(config)

    assert config.base_order_quote == 10.0

    assert config.min_order_quote == 0.5





def test_usdt_order_plan_uses_quote_amount():

    config = TradingConfig(

        symbol="ETH/USDT",

        quote_currency="USDT",

        base_currency="ETH",

        base_order_krw=10.0,

        max_order_krw=30.0,

        min_order_krw=0.5,

        max_position_krw=50.0,

        dry_run_balance_quote=1000.0,

    )

    plan = build_order_plan(config, _signal(), quote_balance_krw=1000.0)

    assert plan.action == "BUY"

    assert plan.notional_quote == 10.0

    assert plan.notional_krw == 10.0  # compatibility alias, not KRW in USDT mode

