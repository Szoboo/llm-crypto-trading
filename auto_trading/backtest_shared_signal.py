"""Minimal backtest helper that reuses the live auto_trading signal engine.

This exists to keep the backtest signal path and live trading signal path aligned.
It intentionally does not place orders.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from .config import load_config
from .signal_engine import generate_signal


def run_shared_signal_backtest(input_csv: str, output_csv: str, window: int = 120) -> None:
    config = load_config()
    df = pd.read_csv(input_csv)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    rows = []
    for end in range(window, len(df) + 1):
        sample = df.iloc[end - window:end].copy()
        signal = generate_signal(config, sample)
        rows.append({
            "timestamp": sample.index[-1].isoformat() if hasattr(sample.index[-1], "isoformat") else str(sample.index[-1]),
            "decision": signal.decision,
            "raw_direction": signal.raw_direction,
            "up_probability": signal.up_probability,
            "down_probability": signal.down_probability,
            "confidence": signal.confidence,
            "price_krw": signal.price,
            "rsi": signal.rsi,
            "macd": signal.macd,
            "atr_krw": signal.atr,
            "reason": signal.reason[:500],
        })

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True) if Path(output_csv).parent != Path(".") else None
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest live shared signal engine on OHLCV CSV")
    parser.add_argument("--input", required=True, help="CSV with timestamp, open, high, low, close, volume")
    parser.add_argument("--output", default="shared_signal_backtest.csv")
    parser.add_argument("--window", type=int, default=120)
    args = parser.parse_args()
    run_shared_signal_backtest(args.input, args.output, args.window)


if __name__ == "__main__":
    main()
