import pandas as pd



from auto_trading.config import TradingConfig

from auto_trading.llm_outcome_logger import append_prediction_row, update_pending_outcomes

from auto_trading.signal_engine import TradeSignal





def _signal():

    return TradeSignal(

        decision="LONG",

        raw_direction="UP",

        up_probability=0.62,

        down_probability=0.38,

        confidence=0.6,

        reason="test",

        price=100.0,

        atr=2.0,

        rsi=55.0,

        macd=1.0,

        support=95.0,

        resistance=110.0,

        llm_used=True,

        llm_failed=False,

        llm_error="",

        llm_raw_response="{}",

        signal_source="indicator_llm_blend",

        llm_up_probability=0.7,

        llm_down_probability=0.3,

        llm_confidence=0.8,

        llm_decision="UP",

        llm_market_regime="bullish_rebound",

        llm_risk_level="medium",

        llm_main_reason="test",

        llm_invalid_condition="close below support",

    )





def test_append_and_fill_outcome(tmp_path):

    path = tmp_path / "outcomes.csv"

    config = TradingConfig()

    append_prediction_row(str(path), config, _signal(), "2026-05-20T04:00:00+00:00", "2026-05-20T08:01:00+00:00")



    df = pd.DataFrame(

        {"close": [100.0, 103.0]},

        index=pd.to_datetime(["2026-05-20T04:00:00+00:00", "2026-05-20T08:00:00+00:00"]),

    )

    updated = update_pending_outcomes(str(path), df)

    assert updated == 1

    out = pd.read_csv(path)

    assert out.loc[0, "outcome_status"] == "filled"

    assert out.loc[0, "actual_direction"] == "UP"

    assert int(out.loc[0, "llm_hit"]) == 1

