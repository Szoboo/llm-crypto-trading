from __future__ import annotations



import argparse

import csv

import itertools

import math

from dataclasses import replace

from pathlib import Path



import pandas as pd



from .config import TradingConfig, load_config

from .risk_manager import build_order_plan

from .signal_engine import generate_signal





TRADING_PERIODS_PER_YEAR = 365 * 6  # 4h candles





def _parse_float_grid(raw: str) -> list[float]:

    return [float(x.strip()) for x in raw.split(",") if x.strip()]





def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:

    rename = {col: col.strip().lower() for col in df.columns}

    df = df.rename(columns=rename)

    aliases = {

        "datetime": "timestamp",

        "date": "timestamp",

        "time": "timestamp",

        "vol": "volume",

    }

    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    required = ["open", "high", "low", "close", "volume"]

    missing = [col for col in required if col not in df.columns]

    if missing:

        raise ValueError(f"Input CSV is missing required OHLCV columns: {missing}")

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        df = df.dropna(subset=["timestamp"]).set_index("timestamp")

    else:

        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    df = df[required].apply(pd.to_numeric, errors="coerce").dropna()

    return df.sort_index()





def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:

    return _normalise_columns(pd.read_csv(path))





def max_drawdown(equity_curve: list[float]) -> float:

    peak = -math.inf

    worst = 0.0

    for value in equity_curve:

        peak = max(peak, value)

        if peak > 0:

            worst = min(worst, value / peak - 1.0)

    return float(worst)





def sharpe_like(equity_curve: list[float]) -> float:

    if len(equity_curve) < 3:

        return 0.0

    series = pd.Series(equity_curve, dtype="float64")

    returns = series.pct_change().dropna()

    std = float(returns.std())

    if std == 0 or math.isnan(std):

        return 0.0

    return float(returns.mean() / std * math.sqrt(TRADING_PERIODS_PER_YEAR))





def run_backtest(

    df: pd.DataFrame,

    config: TradingConfig,

    initial_equity: float,

    fee_bps: float,

    slippage_bps: float,

) -> dict:

    fee_rate = fee_bps / 10_000.0

    slippage_rate = slippage_bps / 10_000.0

    cash = float(initial_equity)

    base_amount = 0.0

    entry_price: float | None = None

    entry_notional = 0.0

    stop_loss: float | None = None

    take_profit: float | None = None

    equity_curve: list[float] = []

    trade_returns: list[float] = []



    start = max(config.lookback_candles, 120)

    for i in range(start, len(df)):

        row = df.iloc[i]

        close = float(row["close"])

        low = float(row["low"])

        high = float(row["high"])

        history = df.iloc[: i + 1]



        try:

            signal = generate_signal(config, history)

        except Exception:

            equity_curve.append(cash + base_amount * close)

            continue



        if base_amount > 0:

            exit_price: float | None = None

            if stop_loss is not None and low <= stop_loss:

                exit_price = stop_loss * (1.0 - slippage_rate)

            elif take_profit is not None and high >= take_profit:

                exit_price = take_profit * (1.0 - slippage_rate)

            elif config.exit_on_downside_signal and signal.raw_direction == "DOWN":

                exit_price = close * (1.0 - slippage_rate)



            if exit_price is not None:

                proceeds = base_amount * exit_price

                proceeds_after_fee = proceeds * (1.0 - fee_rate)

                cash += proceeds_after_fee

                if entry_price and entry_price > 0:

                    trade_returns.append((exit_price * (1.0 - fee_rate)) / entry_price - 1.0)

                base_amount = 0.0

                entry_price = None

                entry_notional = 0.0

                stop_loss = None

                take_profit = None



        if base_amount <= 0 and signal.decision == "LONG":

            plan = build_order_plan(config, signal, cash)

            if plan.action == "BUY":

                notional = min(plan.notional_quote, cash, config.max_position_quote)

                if notional >= config.min_order_quote:

                    fill_price = close * (1.0 + slippage_rate)

                    amount = (notional * (1.0 - fee_rate)) / fill_price

                    cash -= notional

                    base_amount = amount

                    entry_price = fill_price

                    entry_notional = notional

                    stop_loss = plan.stop_loss

                    take_profit = plan.take_profit



        equity_curve.append(cash + base_amount * close)



    final_equity = equity_curve[-1] if equity_curve else initial_equity

    total_return = final_equity / initial_equity - 1.0

    wins = [ret for ret in trade_returns if ret > 0]

    losses = [ret for ret in trade_returns if ret <= 0]

    gross_win = sum(wins)

    gross_loss = abs(sum(losses))

    trade_count = len(trade_returns)

    win_rate = len(wins) / trade_count if trade_count else 0.0

    profit_factor = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    mdd = max_drawdown(equity_curve)

    sharpe = sharpe_like(equity_curve)



    return {

        "total_return": total_return,

        "max_drawdown": mdd,

        "sharpe_like": sharpe,

        "win_rate": win_rate,

        "trade_count": trade_count,

        "avg_trade_return": sum(trade_returns) / trade_count if trade_count else 0.0,

        "profit_factor": profit_factor,

        "final_equity": final_equity,

    }





def build_tuning_config(base_config: TradingConfig, params: dict[str, float]) -> TradingConfig:

    return replace(

        base_config,

        dry_run=True,

        use_llm=False,  # deterministic and fast tuning; live can still use Vertex after params are chosen.

        min_confidence=params["min_confidence"],

        probability_threshold=params["probability_threshold"],

        atr_stop_multiplier=params["atr_stop_multiplier"],

        take_profit_r_multiple=params["take_profit_r_multiple"],

    )





def run_grid_search(args: argparse.Namespace) -> list[dict]:

    df = load_ohlcv_csv(args.input)

    base_config = load_config()

    if args.symbol:

        base, quote = args.symbol.upper().split("/", 1)

        base_config = replace(base_config, symbol=args.symbol.upper(), base_currency=base, quote_currency=quote)

    if args.initial_equity is None:

        args.initial_equity = base_config.dry_run_balance_quote



    rows: list[dict] = []

    grid = itertools.product(

        _parse_float_grid(args.min_confidence),

        _parse_float_grid(args.probability_threshold),

        _parse_float_grid(args.atr_stop_multiplier),

        _parse_float_grid(args.take_profit_r_multiple),

    )

    for min_conf, prob_threshold, atr_mult, tp_r in grid:

        params = {

            "min_confidence": min_conf,

            "probability_threshold": prob_threshold,

            "atr_stop_multiplier": atr_mult,

            "take_profit_r_multiple": tp_r,

        }

        config = build_tuning_config(base_config, params)

        metrics = run_backtest(df, config, args.initial_equity, args.fee_bps, args.slippage_bps)

        score = metrics["total_return"] - abs(metrics["max_drawdown"]) * args.mdd_penalty

        if metrics["trade_count"] < args.min_trades:

            score -= args.low_trade_penalty

        rows.append({**params, **metrics, "score": score})



    rows.sort(key=lambda row: row["score"], reverse=True)

    for idx, row in enumerate(rows, start=1):

        row["rank"] = idx

    return rows





def write_results(rows: list[dict], output: str | Path) -> None:

    output = Path(output)

    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [

        "rank",

        "min_confidence",

        "probability_threshold",

        "atr_stop_multiplier",

        "take_profit_r_multiple",

        "total_return",

        "max_drawdown",

        "sharpe_like",

        "win_rate",

        "trade_count",

        "avg_trade_return",

        "profit_factor",

        "final_equity",

        "score",

    ]

    with output.open("w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(rows)





def main() -> None:

    parser = argparse.ArgumentParser(description="Grid-search tuner for auto_trading long-only parameters.")

    parser.add_argument("--input", required=True, help="OHLCV CSV with timestamp/open/high/low/close/volume columns")

    parser.add_argument("--output", default="auto_trading_tuning_results.csv")

    parser.add_argument("--symbol", default=None, help="Optional override, e.g. ETH/USDT")

    parser.add_argument("--initial-equity", type=float, default=None)

    parser.add_argument("--fee-bps", type=float, default=5.0)

    parser.add_argument("--slippage-bps", type=float, default=2.0)

    parser.add_argument("--mdd-penalty", type=float, default=0.7)

    parser.add_argument("--min-trades", type=int, default=3)

    parser.add_argument("--low-trade-penalty", type=float, default=0.05)

    parser.add_argument("--min-confidence", default="0.50,0.54,0.58,0.62,0.66")

    parser.add_argument("--probability-threshold", default="0.05,0.10,0.15,0.20")

    parser.add_argument("--atr-stop-multiplier", default="1.0,1.5,2.0,2.5")

    parser.add_argument("--take-profit-r-multiple", default="1.2,1.5,2.0,2.5,3.0")

    args = parser.parse_args()



    rows = run_grid_search(args)

    write_results(rows, args.output)

    print(f"saved {len(rows)} tuning results to {args.output}")

    print("top 5:")

    for row in rows[:5]:

        print(row)





if __name__ == "__main__":

    main()

