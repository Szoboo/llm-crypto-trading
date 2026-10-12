from __future__ import annotations



import math

from typing import Any, Iterable





def _is_nan(value: Any) -> bool:

    try:

        return value is None or (isinstance(value, float) and math.isnan(value))

    except Exception:

        return False





def as_float(value: Any, default: float | None = None) -> float | None:

    if value is None or value == "":

        return default

    try:

        return float(value)

    except Exception:

        return default





def pct(value: Any, signed: bool = False) -> str:

    val = as_float(value)

    if val is None or _is_nan(val):

        return "-"

    sign = "+" if signed and val > 0 else ""

    return f"{sign}{val * 100:.2f}%"





def num(value: Any, digits: int = 4) -> str:

    val = as_float(value)

    if val is None or _is_nan(val):

        return "-"

    return f"{val:.{digits}f}"





def money(value: Any, currency: str = "") -> str:

    val = as_float(value)

    if val is None or _is_nan(val):

        return "-"

    suffix = f" {currency}" if currency else ""

    return f"{val:,.2f}{suffix}"





def integer(value: Any) -> str:

    val = as_float(value)

    if val is None or _is_nan(val):

        return "-"

    return f"{int(round(val)):,}"





def shorten(text: Any, max_len: int = 110) -> str:

    if text is None:

        return "-"

    cleaned = " ".join(str(text).replace("\n", " ").replace("\r", " ").split())

    if not cleaned:

        return "-"

    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 3] + "..."





def _line(width: int = 72, char: str = "=") -> str:

    return char * width





def _section(title: str, rows: Iterable[tuple[str, Any]], width: int = 72) -> None:

    rows = [(str(k), str(v)) for k, v in rows]

    if not rows:

        return

    key_width = max(len(k) for k, _ in rows)

    print(f"\n[{title}]")

    for key, value in rows:

        print(f"{key:<{key_width}} : {value}")





def _title(title: str, width: int = 72) -> None:

    print("\n" + _line(width))

    print(f" {title}")

    print(_line(width))





def _get(mapping: dict, *names: str, default: Any = "") -> Any:

    for name in names:

        value = mapping.get(name)

        if value not in (None, ""):

            return value

    return default





def print_metrics_summary(metrics: dict) -> None:

    """Print a concise, video-friendly metrics report."""

    if not metrics:

        print("계산할 데이터가 없습니다.")

        return



    _title("BACKTEST METRICS SUMMARY")



    _section(

        "PREDICTION",

        [

            ("Accuracy", pct(metrics.get("directional_acc"))),

            ("Majority Baseline", pct(metrics.get("majority_baseline_acc"))),

            ("Actual UP / DOWN", f"{integer(metrics.get('actual_up_count'))} / {integer(metrics.get('actual_down_count'))}"),

            ("Precision", pct(metrics.get("precision"))),

            ("Recall", pct(metrics.get("recall"))),

            ("F1 Score", num(metrics.get("f1_score"), 4)),

            ("AUC-ROC", num(metrics.get("auc_roc"), 4)),

        ],

    )



    _section(

        "CONFUSION MATRIX",

        [

            ("TN", integer(metrics.get("tn"))),

            ("FP", integer(metrics.get("fp"))),

            ("FN", integer(metrics.get("fn"))),

            ("TP", integer(metrics.get("tp"))),

        ],

    )



    _section(

        "TRADING PERFORMANCE",

        [

            ("Cumulative Return", pct(metrics.get("cumulative_return"), signed=True)),

            ("Buy & Hold Return", pct(metrics.get("buy_hold_return"), signed=True)),

            ("Max Drawdown", pct(metrics.get("mdd"), signed=True)),

            ("Sharpe Ratio", num(metrics.get("sharpe"), 4)),

            ("Win Rate", pct(metrics.get("win_rate"))),

            ("Profit Factor", num(metrics.get("profit_factor"), 4)),

            ("Active Return Candles", integer(metrics.get("active_return_candle_count"))),

            ("Position Changes", integer(metrics.get("position_change_count"))),

            ("Short Position Candles", integer(metrics.get("short_position_count"))),

        ],

    )



    _section(

        "SUMMARY",

        [

            ("Total Candles", integer(metrics.get("total_points"))),

            ("Correct Predictions", integer(metrics.get("successful_preds"))),

        ],

    )





def print_threshold_summary(threshold_summary) -> None:

    """Print conservative threshold rows in a compact table."""

    try:

        if threshold_summary is None or len(threshold_summary) == 0:

            return

    except Exception:

        return



    _title("CONSERVATIVE THRESHOLD SUMMARY")



    headers = ["Rule", "Threshold", "Coverage", "Accuracy", "Precision", "Recall", "Long Count", "Long Precision"]

    rows: list[list[str]] = []



    for _, row in threshold_summary.iterrows():

        mode = str(row.get("mode", ""))

        if mode == "current_up_prob_gt_down_prob":

            rule = "Current"

        elif mode == "conservative_up_threshold":

            rule = "UP >= threshold"

        else:

            rule = mode[:18]



        threshold = row.get("up_threshold", "")

        threshold_text = "-" if threshold == "" or _is_nan(as_float(threshold)) else num(threshold, 2)



        long_count = row.get("long_signal_count", "")

        long_count_text = "-" if long_count == "" or _is_nan(as_float(long_count)) else integer(long_count)



        rows.append(

            [

                rule,

                threshold_text,

                pct(row.get("coverage")),

                pct(row.get("directional_acc")),

                pct(row.get("precision")),

                pct(row.get("recall")),

                long_count_text,

                pct(row.get("long_signal_precision")),

            ]

        )



    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)

    print(fmt.format(*headers))

    print(fmt.format(*["-" * w for w in widths]))

    for row in rows:

        print(fmt.format(*row))





def compact_trade_log(result: dict) -> dict:

    """Small result for logger.info. Avoids dumping raw LLM text to console."""

    return {

        "status": result.get("status"),

        "symbol": result.get("symbol"),

        "dry_run": result.get("dry_run"),

        "live_enabled": result.get("live_enabled"),

        "decision": result.get("decision"),

        "raw_direction": result.get("raw_direction"),

        "action": result.get("action"),

        "signal_source": result.get("signal_source"),

        "llm_used": result.get("llm_used"),

        "llm_failed": result.get("llm_failed"),

        "llm_decision": result.get("llm_decision"),

        "llm_confidence": result.get("llm_confidence"),

        "price": result.get("price", result.get("price_krw")),

        "outcome_updated_rows": result.get("llm_outcome_updated_rows"),

    }





def print_trade_summary(result: dict) -> None:

    """Print one dry-run/live trading decision in a compact, demo-friendly format."""

    if not result:

        print("거래 결과가 없습니다.")

        return



    if result.get("status") == "skipped":

        _title("AUTO TRADING RESULT")

        _section("SKIPPED", [("Reason", result.get("reason", "-")), ("Timestamp", result.get("timestamp", "-"))])

        return



    quote = str(_get(result, "quote_currency", default="USDT"))

    base = str(_get(result, "base_currency", default="ETH"))



    _title("DRY-RUN AUTO TRADING RESULT")



    _section(

        "SAFETY",

        [

            ("Status", _get(result, "status", default="-")),

            ("Symbol", _get(result, "symbol", default="-")),

            ("Exchange", _get(result, "exchange", default="-")),

            ("Timeframe", _get(result, "timeframe", default="-")),

            ("Dry Run", _get(result, "dry_run", default="-")),

            ("Live Enabled", _get(result, "live_enabled", default="-")),

        ],

    )



    _section(

        "DECISION",

        [

            ("Final Decision", _get(result, "decision", default="-")),

            ("Raw Direction", _get(result, "raw_direction", default="-")),

            ("Action", _get(result, "action", default="-")),

            ("Signal Source", _get(result, "signal_source", default="-")),

            ("Final UP Prob.", num(_get(result, "up_probability", default=None), 4)),

            ("Final DOWN Prob.", num(_get(result, "down_probability", default=None), 4)),

            ("Final Confidence", num(_get(result, "confidence", default=None), 4)),

        ],

    )



    _section(

        "LLM",

        [

            ("LLM Used", _get(result, "llm_used", default="-")),

            ("LLM Failed", _get(result, "llm_failed", default="-")),

            ("LLM Decision", _get(result, "llm_decision", default="-")),

            ("LLM UP Prob.", num(_get(result, "llm_up_probability", default=None), 4)),

            ("LLM DOWN Prob.", num(_get(result, "llm_down_probability", default=None), 4)),

            ("LLM Confidence", num(_get(result, "llm_confidence", default=None), 4)),

            ("Market Regime", _get(result, "llm_market_regime", default="-")),

            ("Risk Level", _get(result, "llm_risk_level", default="-")),

            ("Parse Error", shorten(_get(result, "llm_parse_error", default=""), 90)),

        ],

    )



    _section(

        "ACCOUNT / POSITION",

        [

            ("Price", money(_get(result, "price", "price_krw", default=None), quote)),

            ("Quote Balance", money(_get(result, "quote_balance", "quote_balance_krw", default=None), quote)),

            ("Base Balance", money(_get(result, "base_balance", "base_balance_eth", default=None), base)),

            ("Position Open", _get(result, "position_open", default="-")),

            ("Current Position", money(_get(result, "current_position_value", "current_position_krw", default=None), quote)),

            ("Unrealized PnL", money(_get(result, "unrealized_pnl", "unrealized_pnl_krw", default=None), quote)),

            ("Unrealized PnL %", pct(_get(result, "unrealized_pnl_pct", default=None))),

        ],

    )



    _section(

        "OUTCOME LOGGER",

        [

            ("Updated Rows", integer(_get(result, "llm_outcome_updated_rows", default=0))),

            ("Log Error", shorten(_get(result, "llm_outcome_log_error", default=""), 100)),

        ],

    )



    _section(

        "SHORT REASON",

        [

            ("Reason", shorten(_get(result, "reason", default=""), 130)),

        ],

    )



    print(_line())

