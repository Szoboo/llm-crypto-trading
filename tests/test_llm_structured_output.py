from auto_trading.signal_engine import _parse_llm_response





def test_parse_json_llm_response():

    result = _parse_llm_response(

        '{"market_regime":"bullish_rebound","up_probability":0.7,"down_probability":0.3,'

        '"confidence":0.8,"decision":"UP","risk_level":"medium",'

        '"main_reason":"test reason","invalid_condition":"close below support"}'

    )

    assert result.used is True

    assert result.failed is False

    assert round(result.up_probability, 2) == 0.70

    assert round(result.down_probability, 2) == 0.30

    assert result.decision == "UP"

    assert result.market_regime == "bullish_rebound"

    assert result.parse_error == ""





def test_legacy_parser_fallback():

    result = _parse_llm_response(

        "UP_probability: 0.65\nDOWN_probability: 0.35\nConfidence: 0.7\nDecision: UP\nReason: legacy"

    )

    assert round(result.up_probability, 2) == 0.65

    assert result.decision == "UP"

    assert "legacy parser" in result.parse_error

