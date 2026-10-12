from __future__ import annotations



import json

import os

import re

from dataclasses import dataclass



import pandas as pd



from .config import TradingConfig

from .indicators import calculate_technical_indicators, trend_slope





@dataclass(frozen=True)

class TradeSignal:

    decision: str  # LONG, HOLD. SHORT is intentionally not emitted.

    raw_direction: str  # UP, DOWN, NEUTRAL. DOWN becomes HOLD by policy.

    up_probability: float

    down_probability: float

    confidence: float

    reason: str

    price: float

    atr: float

    rsi: float

    macd: float

    support: float

    resistance: float

    llm_used: bool

    llm_failed: bool

    llm_error: str

    llm_raw_response: str

    signal_source: str  # indicator_only, indicator_llm_blend, indicator_only_llm_disabled, indicator_only_llm_failed



    # Structured LLM fields. Defaults keep old tests/call sites compatible.

    llm_parse_error: str = ""

    llm_up_probability: float = 0.5

    llm_down_probability: float = 0.5

    llm_confidence: float = 0.0

    llm_decision: str = "NEUTRAL"

    llm_market_regime: str = "unknown"

    llm_risk_level: str = "medium"

    llm_main_reason: str = ""

    llm_invalid_condition: str = ""





@dataclass(frozen=True)

class LLMResult:

    used: bool

    failed: bool

    up_probability: float

    down_probability: float

    confidence: float

    raw_response: str

    error: str

    parse_error: str = ""

    decision: str = "NEUTRAL"

    market_regime: str = "unknown"

    risk_level: str = "medium"

    main_reason: str = ""

    invalid_condition: str = ""





def _to_probability(value, default: float) -> float:

    try:

        val = float(value)

    except Exception:

        return default

    if val > 1:

        val = val / 100.0

    return min(max(val, 0.0), 1.0)





def _normalize_probs(up: float, down: float) -> tuple[float, float]:

    up = min(max(float(up), 0.0), 1.0)

    down = min(max(float(down), 0.0), 1.0)

    total = up + down

    if total > 0:

        return up / total, down / total

    return 0.5, 0.5





def _extract_json_object(text: str) -> str:

    stripped = text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)

    if fenced:

        return fenced.group(1)

    start = stripped.find("{")

    end = stripped.rfind("}")

    if start >= 0 and end > start:

        return stripped[start : end + 1]

    raise ValueError("No JSON object found in LLM response")





def _parse_llm_json(text: str) -> LLMResult:

    data = json.loads(_extract_json_object(text))

    up = _to_probability(data.get("up_probability"), 0.5)

    down = _to_probability(data.get("down_probability"), 0.5)

    up, down = _normalize_probs(up, down)

    confidence = _to_probability(data.get("confidence"), abs(up - down))

    decision = str(data.get("decision", "NEUTRAL")).upper().strip()

    if decision not in {"UP", "DOWN", "NEUTRAL"}:

        decision = "NEUTRAL"



    risk_level = str(data.get("risk_level", "medium")).lower().strip()

    if risk_level not in {"low", "medium", "high"}:

        risk_level = "medium"



    return LLMResult(

        used=True,

        failed=False,

        up_probability=up,

        down_probability=down,

        confidence=confidence,

        raw_response=text,

        error="",

        parse_error="",

        decision=decision,

        market_regime=str(data.get("market_regime", "unknown"))[:120],

        risk_level=risk_level,

        main_reason=str(data.get("main_reason", data.get("reason", "")))[:500],

        invalid_condition=str(data.get("invalid_condition", ""))[:300],

    )





def _parse_llm_probs(text: str) -> tuple[float, float, float]:

    def find_number(key: str, default: float) -> float:

        match = re.search(rf"{key}\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)

        if not match:

            return default

        return _to_probability(match.group(1), default)



    up = find_number("UP_probability", 0.5)

    down = find_number("DOWN_probability", 0.5)

    up, down = _normalize_probs(up, down)

    confidence = find_number("Confidence", abs(up - down))

    return up, down, confidence





def _parse_legacy_llm_response(text: str, parse_error: str = "") -> LLMResult:

    up, down, confidence = _parse_llm_probs(text)

    decision_match = re.search(r"Decision\s*[:=]\s*(UP|DOWN|NEUTRAL)", text, re.IGNORECASE)

    reason_match = re.search(r"Reason\s*[:=]\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    decision = decision_match.group(1).upper() if decision_match else "NEUTRAL"

    reason = reason_match.group(1).strip() if reason_match else ""

    return LLMResult(

        used=True,

        failed=False,

        up_probability=up,

        down_probability=down,

        confidence=confidence,

        raw_response=text,

        error="",

        parse_error=parse_error or "LLM response was not valid JSON; legacy probability parser was used",

        decision=decision,

        market_regime="legacy_text_response",

        risk_level="medium",

        main_reason=reason[:500],

        invalid_condition="",

    )





def _parse_llm_response(text: str) -> LLMResult:

    try:

        return _parse_llm_json(text)

    except Exception as exc:

        return _parse_legacy_llm_response(text, f"LLM response was not valid JSON; legacy parser used: {exc}")





def _indicator_probabilities(df: pd.DataFrame) -> tuple[float, float, str]:

    last = df.iloc[-1]

    prev = df.iloc[-2]

    score = 0.0

    reasons: list[str] = []



    if last["RSI_14"] < 30:

        score += 1.2

        reasons.append("RSI oversold: long rebound candidate")

    elif last["RSI_14"] > 70:

        score -= 1.2

        reasons.append("RSI overbought: block new long")



    if last["MACD"] > last["MACD_SIGNAL"] and last["MACD"] > prev["MACD"]:

        score += 1.0

        reasons.append("MACD bullish")

    elif last["MACD"] < last["MACD_SIGNAL"] and last["MACD"] < prev["MACD"]:

        score -= 1.0

        reasons.append("MACD bearish: no short, wait")



    if last["SMA_12"] > last["SMA_48"]:

        score += 0.8

        reasons.append("short MA above long MA")

    elif last["SMA_12"] < last["SMA_48"]:

        score -= 0.8

        reasons.append("short MA below long MA: no new long")



    slope = trend_slope(df["close"], 20)

    if slope > 0:

        score += 0.5

        reasons.append("20-candle slope up")

    elif slope < 0:

        score -= 0.5

        reasons.append("20-candle slope down: no short, wait")



    up = min(max(0.5 + score / 7.0, 0.05), 0.95)

    down = 1.0 - up

    return up, down, ", ".join(reasons) if reasons else "neutral indicators"





def _response_to_text(response) -> str:

    if response is None:

        return ""

    content = getattr(response, "content", None)

    if isinstance(content, str):

        return content

    if content is not None:

        return str(content)

    text = getattr(response, "text", None)

    if isinstance(text, str):

        return text

    return str(response)





def _vertex_project(config: TradingConfig) -> str:

    return (

        getattr(config, "gcp_project_id", "")

        or os.getenv("AUTO_TRADING_GCP_PROJECT_ID", "")

        or os.getenv("GCP_PROJECT_ID", "")

    )





def _vertex_location(config: TradingConfig) -> str:

    return (

        getattr(config, "gcp_location", "")

        or os.getenv("AUTO_TRADING_GCP_LOCATION", "")

        or os.getenv("GCP_LOCATION", "")

        or "us-central1"

    )





def _call_llm(config: TradingConfig, df: pd.DataFrame) -> LLMResult:

    if not config.use_llm:

        return LLMResult(False, False, 0.5, 0.5, 0.0, "", "LLM disabled by AUTO_TRADING_USE_LLM=false")



    gcp_project_id = _vertex_project(config)

    gcp_location = _vertex_location(config)

    if not gcp_project_id:

        return LLMResult(

            False,

            True,

            0.5,

            0.5,

            0.0,

            "",

            "GCP_PROJECT_ID or AUTO_TRADING_GCP_PROJECT_ID is missing for Vertex AI Gemini",

        )



    # Preserve the existing Vertex AI / ADC structure. This does not use GOOGLE_API_KEY.

    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project_id

    os.environ["GOOGLE_CLOUD_LOCATION"] = gcp_location



    try:

        from langchain_google_genai import ChatGoogleGenerativeAI

    except Exception as exc:

        return LLMResult(False, True, 0.5, 0.5, 0.0, "", f"langchain_google_genai import failed: {exc}")



    quote = getattr(config, "quote_currency", "KRW")

    symbol = getattr(config, "symbol", "ETH/KRW")

    recent = df.tail(20)[["close", "volume", "RSI_14", "MACD", "MACD_SIGNAL", "ATR_14"]].to_string()

    last = df.iloc[-1]

    prompt = f"""

You are a quantitative {symbol} spot trading analyst for Upbit.

Predict {symbol} direction for the next closed {config.timeframe} candle.



[DATA]

{recent}



Current close: {last['close']:.8f} {quote}

Support: {last['Support_1']:.8f} {quote}

Resistance: {last['Resistance_1']:.8f} {quote}



Project policy:

- Long-only spot trading.

- Never recommend a short position.

- If downside probability is stronger, the action must be HOLD/WAIT, not SHORT.

- RSI > 70 blocks new long entries unless a strong breakout is present.

- RSI < 30 can be a long rebound candidate only if other indicators confirm.





Conservative long-signal policy:

- For long-only trading, false UP signals are more harmful than missed opportunities.

- Choose NEUTRAL unless bullish evidence is clearly dominant.

- Do not assign high UP probability when trend, MACD, support/resistance, and volatility conditions are mixed.

- In bearish_trend, bearish_breakdown, or high_volatility regimes, require stronger evidence before choosing UP.

- Prefer lower UP probability and NEUTRAL decision when evidence is ambiguous.



Return exactly one valid JSON object. Do not include markdown fences or extra text.

Schema:

{{

  "market_regime": "bullish_trend | bearish_trend | range | bullish_rebound | bearish_breakdown | high_volatility | unknown",

  "up_probability": 0.0,

  "down_probability": 0.0,

  "confidence": 0.0,

  "decision": "UP | DOWN | NEUTRAL",

  "risk_level": "low | medium | high",

  "main_reason": "one concise sentence",

  "invalid_condition": "one concise condition that would invalidate the view"

}}

""".strip()



    try:

        llm = ChatGoogleGenerativeAI(model=config.google_model, temperature=0.1)

        response = llm.invoke(prompt)

        text = _response_to_text(response)

        return _parse_llm_response(text)

    except Exception as exc:

        return LLMResult(False, True, 0.5, 0.5, 0.0, "", str(exc))





def generate_signal(config: TradingConfig, raw_ohlcv: pd.DataFrame) -> TradeSignal:

    df = calculate_technical_indicators(raw_ohlcv).dropna().copy()

    if len(df) < 3:

        raise ValueError("Not enough candles after indicator calculation")



    last = df.iloc[-1]

    ind_up, ind_down, ind_reason = _indicator_probabilities(df)

    llm_result = _call_llm(config, df)



    if not llm_result.used:

        up, down = ind_up, ind_down

        confidence = abs(up - down)

        if config.use_llm and llm_result.failed:

            signal_source = "indicator_only_llm_failed"

            reason = f"indicator-only after LLM failure: {ind_reason}; llm_error={llm_result.error}"

        elif not config.use_llm:

            signal_source = "indicator_only_llm_disabled"

            reason = f"indicator-only because LLM is disabled: {ind_reason}"

        else:

            signal_source = "indicator_only"

            reason = f"indicator-only: {ind_reason}"

    else:

        # Keep LLM central but still blend with indicator probabilities for risk control.

        up = (0.60 * ind_up) + (0.40 * llm_result.up_probability)

        down = (0.60 * ind_down) + (0.40 * llm_result.down_probability)

        total = up + down

        up, down = up / total, down / total

        confidence = min(max((abs(up - down) + llm_result.confidence) / 2, 0), 1)

        signal_source = "indicator_llm_blend"

        raw_preview = llm_result.raw_response[:300].replace(chr(10), " ")

        reason = f"blend: {ind_reason}; llm_decision={llm_result.decision}; llm_reason={llm_result.main_reason}; llm={raw_preview}"

        if llm_result.parse_error:

            reason += f"; llm_parse_error={llm_result.parse_error[:160]}"



    rsi = float(last["RSI_14"])

    if rsi > 75:

        up = min(up, 0.45)

        down = max(down, 0.55)

        reason += "; RSI extreme overbought: new long blocked"

    elif rsi < 25:

        up = max(up, 0.55)

        down = min(down, 0.45)

        reason += "; RSI extreme oversold: rebound watch"



    diff = up - down



    conservative_long_filter = str(os.getenv("AUTO_TRADING_CONSERVATIVE_LONG_FILTER", "false")).strip().lower() in {"1", "true", "yes", "y", "on"}

    conservative_blocked_long = False

    if conservative_long_filter and llm_result.used and diff > 0:

        long_up_threshold = float(os.getenv("AUTO_TRADING_LONG_UP_THRESHOLD", "0.75"))

        long_llm_confidence = float(os.getenv("AUTO_TRADING_LONG_LLM_CONFIDENCE", "0.65"))

        blocked_regimes = {"bearish_trend", "bearish_breakdown", "high_volatility"}

        regime = str(llm_result.market_regime).strip().lower()

        risk_level = str(llm_result.risk_level).strip().lower()

        conservative_blocked_long = not (

            up >= long_up_threshold

            and llm_result.decision == "UP"

            and llm_result.confidence >= long_llm_confidence

            and regime not in blocked_regimes

            and risk_level != "high"

        )

        if conservative_blocked_long:

            reason += (

                f"; conservative long filter blocked LONG "

                f"(up={up:.4f}, llm_decision={llm_result.decision}, "

                f"llm_conf={llm_result.confidence:.4f}, regime={regime}, risk={risk_level})"

            )



    if abs(diff) < config.probability_threshold or confidence < config.min_confidence:

        raw_direction = "NEUTRAL"

        decision = "HOLD"

    elif diff > 0 and not conservative_blocked_long:

        raw_direction = "UP"

        decision = "LONG"

    elif diff > 0 and conservative_blocked_long:

        raw_direction = "UP"

        decision = "HOLD"

    else:

        raw_direction = "DOWN"

        decision = "HOLD"

        reason += "; downside/short signal converted to HOLD by long-only policy"



    return TradeSignal(

        decision=decision,

        raw_direction=raw_direction,

        up_probability=round(float(up), 4),

        down_probability=round(float(down), 4),

        confidence=round(float(confidence), 4),

        reason=reason,

        price=float(last["close"]),

        atr=float(last["ATR_14"]),

        rsi=rsi,

        macd=float(last["MACD"]),

        support=float(last["Support_1"]),

        resistance=float(last["Resistance_1"]),

        llm_used=llm_result.used,

        llm_failed=llm_result.failed,

        llm_error=llm_result.error[:500],

        llm_raw_response=llm_result.raw_response[:1000].replace(chr(10), " "),

        signal_source=signal_source,

        llm_parse_error=llm_result.parse_error[:500],

        llm_up_probability=round(float(llm_result.up_probability), 4),

        llm_down_probability=round(float(llm_result.down_probability), 4),

        llm_confidence=round(float(llm_result.confidence), 4),

        llm_decision=llm_result.decision,

        llm_market_regime=llm_result.market_regime,

        llm_risk_level=llm_result.risk_level,

        llm_main_reason=llm_result.main_reason,

        llm_invalid_condition=llm_result.invalid_condition,

    )

