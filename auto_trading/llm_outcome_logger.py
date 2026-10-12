from __future__ import annotations



import argparse

import csv

import os

from datetime import timedelta

from pathlib import Path

from typing import Iterable



import pandas as pd



from .config import TradingConfig, load_config

from .signal_engine import TradeSignal





OUTCOME_FIELDNAMES = [

    "run_at",

    "symbol",

    "timeframe",

    "quote_currency",

    "base_currency",

    "prediction_candle_timestamp",

    "target_candle_timestamp",

    "prediction_close",

    "final_decision",

    "raw_direction",

    "final_up_probability",

    "final_down_probability",

    "final_confidence",

    "signal_source",

    "llm_used",

    "llm_failed",

    "llm_error",

    "llm_parse_error",

    "llm_up_probability",

    "llm_down_probability",

    "llm_confidence",

    "llm_decision",

    "llm_market_regime",

    "llm_risk_level",

    "llm_main_reason",

    "llm_invalid_condition",

    "rsi",

    "macd",

    "atr",

    "actual_close",

    "actual_return",

    "actual_direction",

    "llm_hit",

    "final_long_hit",

    "outcome_status",

]





def default_outcome_log_file(config: TradingConfig) -> str:

    configured = getattr(config, "llm_outcome_log_file", "")

    if configured:

        return configured

    env_path = os.getenv("AUTO_TRADING_LLM_OUTCOME_LOG_FILE", "")

    if env_path:

        return env_path

    return "llm_prediction_outcomes_usdt.csv" if getattr(config, "quote_currency", "KRW").upper() == "USDT" else "llm_prediction_outcomes.csv"





def outcome_logging_enabled(config: TradingConfig) -> bool:

    configured = getattr(config, "enable_llm_outcome_logging", None)

    if configured is not None:

        return bool(configured)

    raw = os.getenv("AUTO_TRADING_ENABLE_LLM_OUTCOME_LOGGING")

    if raw is None:

        return True

    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}





def _timeframe_delta(timeframe: str) -> timedelta:

    tf = timeframe.strip().lower()

    if tf.endswith("m"):

        return timedelta(minutes=int(tf[:-1]))

    if tf.endswith("h"):

        return timedelta(hours=int(tf[:-1]))

    if tf.endswith("d"):

        return timedelta(days=int(tf[:-1]))

    raise ValueError(f"Unsupported timeframe for outcome logging: {timeframe}")





def _timestamp(value) -> pd.Timestamp:

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:

        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")





def _iso(value) -> str:

    return _timestamp(value).isoformat()





def _read_rows(path: str) -> list[dict]:

    if not os.path.exists(path) or os.path.getsize(path) == 0:

        return []

    with open(path, "r", newline="", encoding="utf-8") as f:

        return list(csv.DictReader(f))





def _write_rows(path: str, rows: Iterable[dict]) -> None:

    rows = list(rows)

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=OUTCOME_FIELDNAMES)

        writer.writeheader()

        for row in rows:

            writer.writerow({name: row.get(name, "") for name in OUTCOME_FIELDNAMES})





def _close_lookup(ohlcv: pd.DataFrame) -> dict[str, float]:

    if "close" not in ohlcv.columns:

        raise ValueError("OHLCV dataframe must include a close column")

    if not isinstance(ohlcv.index, pd.DatetimeIndex):

        raise ValueError("OHLCV dataframe index must be a DatetimeIndex")

    lookup: dict[str, float] = {}

    for timestamp, row in ohlcv.iterrows():

        lookup[_iso(timestamp)] = float(row["close"])

    return lookup





def _actual_direction(actual_return: float) -> str:

    if actual_return > 0:

        return "UP"

    if actual_return < 0:

        return "DOWN"

    return "NEUTRAL"





def _llm_hit(llm_decision: str, actual_direction: str) -> str:

    decision = (llm_decision or "").upper()

    if decision not in {"UP", "DOWN", "NEUTRAL"}:

        return ""

    if decision == "NEUTRAL":

        return ""

    return "1" if decision == actual_direction else "0"





def _final_long_hit(final_decision: str, actual_return: float) -> str:

    if final_decision != "LONG":

        return ""

    return "1" if actual_return > 0 else "0"





def update_pending_outcomes(path: str, ohlcv: pd.DataFrame) -> int:

    rows = _read_rows(path)

    if not rows:

        return 0



    close_by_timestamp = _close_lookup(ohlcv)

    updated = 0

    for row in rows:

        if row.get("outcome_status") == "filled" or row.get("actual_return"):

            continue

        target_ts = row.get("target_candle_timestamp", "")

        if not target_ts:

            continue

        try:

            normalized_target = _iso(target_ts)

        except Exception:

            continue

        actual_close = close_by_timestamp.get(normalized_target)

        if actual_close is None:

            continue

        try:

            prediction_close = float(row.get("prediction_close", ""))

        except ValueError:

            continue

        if prediction_close <= 0:

            continue

        actual_return = (actual_close / prediction_close) - 1.0

        actual_direction = _actual_direction(actual_return)

        row["actual_close"] = f"{actual_close:.12g}"

        row["actual_return"] = f"{actual_return:.10f}"

        row["actual_direction"] = actual_direction

        row["llm_hit"] = _llm_hit(row.get("llm_decision", ""), actual_direction)

        row["final_long_hit"] = _final_long_hit(row.get("final_decision", ""), actual_return)

        row["outcome_status"] = "filled"

        updated += 1



    if updated:

        _write_rows(path, rows)

    return updated





def append_prediction_row(

    path: str,

    config: TradingConfig,

    signal: TradeSignal,

    prediction_candle_timestamp: str,

    run_at: str,

) -> None:

    prediction_ts = _timestamp(prediction_candle_timestamp)

    target_ts = prediction_ts + _timeframe_delta(config.timeframe)

    rows = _read_rows(path)

    prediction_iso = prediction_ts.isoformat()



    # Avoid duplicate rows for the same symbol/timeframe/prediction candle.

    for row in rows:

        if (

            row.get("symbol") == config.symbol

            and row.get("timeframe") == config.timeframe

            and row.get("prediction_candle_timestamp") == prediction_iso

        ):

            return



    row = {

        "run_at": run_at,

        "symbol": config.symbol,

        "timeframe": config.timeframe,

        "quote_currency": getattr(config, "quote_currency", ""),

        "base_currency": getattr(config, "base_currency", ""),

        "prediction_candle_timestamp": prediction_iso,

        "target_candle_timestamp": target_ts.isoformat(),

        "prediction_close": f"{signal.price:.12g}",

        "final_decision": signal.decision,

        "raw_direction": signal.raw_direction,

        "final_up_probability": signal.up_probability,

        "final_down_probability": signal.down_probability,

        "final_confidence": signal.confidence,

        "signal_source": signal.signal_source,

        "llm_used": signal.llm_used,

        "llm_failed": signal.llm_failed,

        "llm_error": signal.llm_error,

        "llm_parse_error": getattr(signal, "llm_parse_error", ""),

        "llm_up_probability": getattr(signal, "llm_up_probability", ""),

        "llm_down_probability": getattr(signal, "llm_down_probability", ""),

        "llm_confidence": getattr(signal, "llm_confidence", ""),

        "llm_decision": getattr(signal, "llm_decision", ""),

        "llm_market_regime": getattr(signal, "llm_market_regime", ""),

        "llm_risk_level": getattr(signal, "llm_risk_level", ""),

        "llm_main_reason": getattr(signal, "llm_main_reason", ""),

        "llm_invalid_condition": getattr(signal, "llm_invalid_condition", ""),

        "rsi": signal.rsi,

        "macd": signal.macd,

        "atr": signal.atr,

        "actual_close": "",

        "actual_return": "",

        "actual_direction": "",

        "llm_hit": "",

        "final_long_hit": "",

        "outcome_status": "pending",

    }

    rows.append(row)

    _write_rows(path, rows)





def summarize_outcomes(path: str) -> dict:

    rows = _read_rows(path)

    filled = [r for r in rows if r.get("outcome_status") == "filled"]

    llm_hits = [int(r["llm_hit"]) for r in filled if r.get("llm_hit") in {"0", "1"}]

    final_hits = [int(r["final_long_hit"]) for r in filled if r.get("final_long_hit") in {"0", "1"}]

    returns = [float(r["actual_return"]) for r in filled if r.get("actual_return")]

    return {

        "rows": len(rows),

        "filled": len(filled),

        "pending": len(rows) - len(filled),

        "llm_direction_hit_rate": (sum(llm_hits) / len(llm_hits)) if llm_hits else None,

        "final_long_hit_rate": (sum(final_hits) / len(final_hits)) if final_hits else None,

        "avg_actual_return": (sum(returns) / len(returns)) if returns else None,

    }





def main() -> None:

    parser = argparse.ArgumentParser(description="Summarize LLM prediction outcome log")

    parser.add_argument("--input", default="", help="Outcome CSV path. Defaults to config-derived path.")

    args = parser.parse_args()

    config = load_config()

    path = args.input or default_outcome_log_file(config)

    print(summarize_outcomes(path))





if __name__ == "__main__":

    main()

