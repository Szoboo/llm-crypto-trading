from __future__ import annotations



import argparse

import csv

import os

import time

from datetime import datetime, timezone



from .config import load_config

from .console_report import compact_trade_log, print_trade_summary

from .exchange_client import ExchangeClient

from .llm_outcome_logger import (

    append_prediction_row,

    default_outcome_log_file,

    outcome_logging_enabled,

    update_pending_outcomes,

)

from .logger import setup_logger

from .position_manager import (

    apply_position_limits,

    build_exit_plan,

    build_position_snapshot,

    calculate_unrealized_pnl,

    record_entry,

    record_exit,

)

from .risk_manager import build_order_plan

from .scheduler import next_4h_close_time, seconds_until

from .signal_engine import generate_signal

from .state import load_state, save_state





def _append_trade_log(path: str, row: dict) -> None:

    """Append one trade-log row, migrating an existing old CSV header if needed."""

    fieldnames = list(row.keys())



    if os.path.exists(path) and os.path.getsize(path) > 0:

        with open(path, "r", newline="", encoding="utf-8") as f:

            reader = csv.DictReader(f)

            existing_fieldnames = reader.fieldnames or []

            existing_rows = list(reader)



        if existing_fieldnames != fieldnames:

            migrated_fieldnames = fieldnames + [name for name in existing_fieldnames if name not in fieldnames]

            with open(path, "w", newline="", encoding="utf-8") as f:

                writer = csv.DictWriter(f, fieldnames=migrated_fieldnames)

                writer.writeheader()

                for existing_row in existing_rows:

                    writer.writerow({name: existing_row.get(name, "") for name in migrated_fieldnames})

                writer.writerow({name: row.get(name, "") for name in migrated_fieldnames})

            return



    exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists or os.path.getsize(path) == 0:

            writer.writeheader()

        writer.writerow(row)





def _get(obj, *names: str, default=None):

    for name in names:

        if hasattr(obj, name):

            return getattr(obj, name)

    return default





def _buy_order(client: ExchangeClient, plan, quote_balance: float) -> dict:

    if hasattr(client, "create_market_buy_quote") and hasattr(plan, "notional_quote"):

        return client.create_market_buy_quote(plan.notional_quote, quote_balance=quote_balance)

    return client.create_market_buy_krw(plan.notional_krw, quote_balance_krw=quote_balance)





def _exit_amount(exit_plan) -> float:

    return float(_get(exit_plan, "amount_base", "amount_eth", default=0.0) or 0.0)





def run_once(force_same_candle: bool = False) -> dict:

    config = load_config()

    logger = setup_logger(config)

    client = ExchangeClient(config)

    state = load_state(config.state_file)



    raw = client.fetch_closed_ohlcv_df(limit=max(config.lookback_candles, 120))

    if raw.empty:

        raise RuntimeError("No closed OHLCV candles returned")

    last_candle_ts = raw.index[-1].isoformat()



    if not force_same_candle and state.last_candle_timestamp == last_candle_ts:

        result = {"status": "skipped", "reason": "already processed this closed candle", "timestamp": last_candle_ts}

        logger.info(result)

        return result



    signal = generate_signal(config, raw)



    quote_balance = client.fetch_quote_balance()

    base_balance = client.fetch_base_balance()

    last_price = client.fetch_last_price()

    position = build_position_snapshot(quote_balance, base_balance, last_price)

    unrealized_pnl, unrealized_pnl_pct = calculate_unrealized_pnl(state, position)



    exit_plan = build_exit_plan(config, state, signal, position)

    order = None

    plan = build_order_plan(config, signal, quote_balance)

    plan = apply_position_limits(config, state, plan, position)



    action = plan.action

    action_reason = plan.reason



    if exit_plan.action == "SELL":

        order = client.create_market_sell_base(_exit_amount(exit_plan))

        record_exit(state, order, exit_plan.reason)

        action = "SELL"

        action_reason = exit_plan.reason

    elif plan.action == "BUY":

        order = _buy_order(client, plan, quote_balance)

        record_entry(state, plan, order, signal.price)

        action = "BUY"

        action_reason = plan.reason



    state.last_candle_timestamp = last_candle_ts

    state.last_decision = signal.decision

    if isinstance(order, dict) and order.get("id"):

        state.last_order_id = order.get("id")

    save_state(config.state_file, state)



    run_at = datetime.now(timezone.utc).isoformat()

    outcome_updated_rows = 0

    outcome_log_error = ""

    if outcome_logging_enabled(config):

        try:

            outcome_path = default_outcome_log_file(config)

            outcome_updated_rows = update_pending_outcomes(outcome_path, raw)

            append_prediction_row(outcome_path, config, signal, last_candle_ts, run_at)

        except Exception as exc:

            # Outcome logging must never stop trading/dry-run execution.

            outcome_log_error = str(exc)[:500]



    current_position_value = _get(position, "current_position_value", "current_position_krw", default=0.0)

    max_position_value = _get(config, "max_position_quote", "max_position_krw", default=0.0)

    entry_price = _get(state, "entry_price", "entry_price_krw", default=None)

    entry_amount = _get(state, "entry_amount_base", "entry_amount_eth", default=None)

    entry_notional = _get(state, "entry_notional", "entry_notional_krw", default=None)

    stop_loss_state = _get(state, "stop_loss", "stop_loss_krw", default=None)

    take_profit_state = _get(state, "take_profit", "take_profit_krw", default=None)

    notional = _get(plan, "notional_quote", "notional_krw", default=0.0)



    result = {

        "run_at": run_at,

        "status": "ok",

        "dry_run": config.dry_run,

        "live_enabled": config.live_enabled,

        "exchange": config.exchange_id,

        "symbol": config.symbol,

        "timeframe": config.timeframe,

        "quote_currency": getattr(config, "quote_currency", ""),

        "base_currency": getattr(config, "base_currency", ""),

        "last_closed_candle_timestamp": last_candle_ts,

        "decision": signal.decision,

        "raw_direction": signal.raw_direction,

        "up_probability": signal.up_probability,

        "down_probability": signal.down_probability,

        "confidence": signal.confidence,

        "signal_source": signal.signal_source,

        "llm_used": signal.llm_used,

        "llm_failed": signal.llm_failed,

        "llm_error": signal.llm_error,

        "llm_raw_response": signal.llm_raw_response[:500],

        "llm_parse_error": getattr(signal, "llm_parse_error", ""),

        "llm_up_probability": getattr(signal, "llm_up_probability", ""),

        "llm_down_probability": getattr(signal, "llm_down_probability", ""),

        "llm_confidence": getattr(signal, "llm_confidence", ""),

        "llm_decision": getattr(signal, "llm_decision", ""),

        "llm_market_regime": getattr(signal, "llm_market_regime", ""),

        "llm_risk_level": getattr(signal, "llm_risk_level", ""),

        "llm_main_reason": getattr(signal, "llm_main_reason", "")[:500],

        "llm_invalid_condition": getattr(signal, "llm_invalid_condition", "")[:300],

        "llm_outcome_updated_rows": outcome_updated_rows,

        "llm_outcome_log_error": outcome_log_error,

        "price": signal.price,

        "price_krw": signal.price,

        "quote_balance": quote_balance,

        "quote_balance_krw": quote_balance,

        "base_balance": base_balance,

        "base_balance_eth": base_balance,

        "current_position_value": current_position_value,

        "current_position_krw": current_position_value,

        "max_position_value": max_position_value,

        "max_position_krw": max_position_value,

        "position_open": state.position_open,

        "entry_price": entry_price,

        "entry_price_krw": entry_price,

        "entry_amount_base": entry_amount,

        "entry_amount_eth": entry_amount,

        "entry_notional": entry_notional,

        "entry_notional_krw": entry_notional,

        "unrealized_pnl": unrealized_pnl,

        "unrealized_pnl_krw": unrealized_pnl,

        "unrealized_pnl_pct": unrealized_pnl_pct,

        "stop_loss": stop_loss_state if action != "BUY" else plan.stop_loss,

        "stop_loss_krw": stop_loss_state if action != "BUY" else plan.stop_loss,

        "take_profit": take_profit_state if action != "BUY" else plan.take_profit,

        "take_profit_krw": take_profit_state if action != "BUY" else plan.take_profit,

        "rsi": signal.rsi,

        "macd": signal.macd,

        "atr": signal.atr,

        "atr_krw": signal.atr,

        "action": action,

        "estimated_base_amount": plan.amount,

        "estimated_eth_amount": plan.amount,

        "notional": notional,

        "notional_krw": notional,

        "exit_amount_base": _exit_amount(exit_plan) if action == "SELL" else 0.0,

        "exit_amount_eth": _exit_amount(exit_plan) if action == "SELL" else 0.0,

        "order_id": order.get("id") if isinstance(order, dict) else "",

        "order_status": order.get("status") if isinstance(order, dict) else "",

        "exit_plan_action": exit_plan.action,

        "exit_plan_reason": exit_plan.reason,

        "error_message": "",

        "reason": action_reason[:500],

    }

    _append_trade_log(config.trade_log_file, result)

    if os.getenv("AUTO_TRADING_CONSOLE_RUN_LOG", "false").strip().lower() in {"1", "true", "yes", "y", "on"}:

        logger.info(compact_trade_log(result))

    return result





def run_loop(raw_output: bool = False) -> None:

    config = load_config()

    logger = setup_logger(config)



    while True:

        try:

            result = run_once(force_same_candle=False)

            if raw_output:

                print(result)

            else:

                print_trade_summary(result)

        except Exception:

            logger.exception("run_once failed")



        if config.use_candle_scheduler:

            target = next_4h_close_time(delay_seconds=config.scheduler_delay_seconds)

            sleep_seconds = seconds_until(target)

            logger.info({"status": "sleep", "until": target.isoformat(), "seconds": round(sleep_seconds, 1)})

            time.sleep(sleep_seconds)

        else:

            time.sleep(config.poll_seconds)





def main() -> None:

    parser = argparse.ArgumentParser(description="Upbit ETH spot long-only 4h automated trading bot")

    parser.add_argument("--once", action="store_true", help="Run one decision cycle and exit")

    parser.add_argument("--loop", action="store_true", help="Run continuously")

    parser.add_argument("--force", action="store_true", help="Allow processing the same closed candle again")

    parser.add_argument("--raw", action="store_true", help="Print raw result dictionary instead of concise summary")

    args = parser.parse_args()



    if args.loop:

        run_loop(raw_output=args.raw)

    else:

        result = run_once(force_same_candle=args.force)

        if args.raw:

            print(result)

        else:

            print_trade_summary(result)





if __name__ == "__main__":

    main()

