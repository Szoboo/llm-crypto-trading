from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the same core indicators used in the existing capstone pipeline."""
    df = df.copy()
    df["SMA_12"] = df["close"].rolling(12).mean()
    df["SMA_48"] = df["close"].rolling(48).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-6)
    df["RSI_14"] = 100 - (100 / (1 + gain / loss))

    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    )
    df["ATR_14"] = ranges.max(axis=1).rolling(14).mean()
    df["Resistance_1"] = df["high"].rolling(60).max().shift(1)
    df["Support_1"] = df["low"].rolling(60).min().shift(1)
    return df


def trend_slope(series: pd.Series, n: int = 20) -> float:
    tail = series.dropna().tail(n)
    if len(tail) < 2:
        return 0.0
    return float(np.polyfit(range(len(tail)), tail.values, 1)[0])
