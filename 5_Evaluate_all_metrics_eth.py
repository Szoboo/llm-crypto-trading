from __future__ import annotations



from pathlib import Path



import numpy as np

import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# ============================================================

# Professor feedback version: 1D price-tracking evaluation only

# ============================================================

# - 4시간봉 방향성/수익률 중심 평가는 최종 출력에서 사용하지 않음

# - Buy & Hold 비교, 누적수익률, Sharpe, Win Rate 제거

# - 2024-01-01 ~ 2025-04-30 구간만 평가

# - 실제 ETH 시세 vs 예측 ETH 시세 그래프 생성

# - 일별 squared error 그래프 생성

# - MSE, MAE, RMSE, MAPE 중심 평가

#

# 주의:

# 기존 LLM 결과는 직접적인 가격 회귀값이 아니라 up_prob/down_prob 기반 방향 예측값이다.

# 따라서 이 파일은 4시간봉 LLM 확률을 1일 단위로 평균 집계하고,

# 최근 7일 일간 변동성을 이용해 다음 날 예상 종가를 추정한다.

# ============================================================



MASTER_FILE = "eth_master_data_filtered.csv"

RESULT_FILE = "eth_backtest_result_LLM_fulltext_2024_01_2025_04.csv"



START_DATE = "2024-01-01"

END_DATE = "2025-04-30"



DAILY_OUTPUT_FILE = "eth_daily_price_tracking_2024_01_2025_04.csv"

METRICS_OUTPUT_FILE = "eth_daily_price_tracking_metrics_2024_01_2025_04.csv"



CHART_ACTUAL_VS_PREDICTED = "eth_daily_actual_vs_predicted_2024_01_2025_04.png"

CHART_SQUARED_ERROR = "eth_daily_squared_error_2024_01_2025_04.png"

CHART_ERROR_METRICS = "eth_daily_error_metrics_2024_01_2025_04.png"



VOLATILITY_WINDOW = 7

VOLATILITY_SCALE = 1.0

MAX_EXPECTED_RETURN = 0.20





def _require_file(path: str) -> None:

    if not Path(path).exists():

        raise FileNotFoundError(f"필수 파일을 찾을 수 없습니다: {path}")





def _to_datetime_utc(series: pd.Series) -> pd.Series:

    return pd.to_datetime(series, errors="coerce", utc=True)





def _safe_float(series: pd.Series) -> pd.Series:

    return pd.to_numeric(series, errors="coerce")





def _pct(value: float | int | None, digits: int = 2) -> str:

    if value is None or pd.isna(value):

        return "nan"

    return f"{float(value) * 100:.{digits}f}%"





def _num(value: float | int | None, digits: int = 4) -> str:

    if value is None or pd.isna(value):

        return "nan"

    return f"{float(value):.{digits}f}"





def load_daily_dataset() -> pd.DataFrame:

    _require_file(MASTER_FILE)

    _require_file(RESULT_FILE)



    master = pd.read_csv(MASTER_FILE)

    result = pd.read_csv(RESULT_FILE)



    for col in ["timestamp", "close"]:

        if col not in master.columns:

            raise ValueError(f"{MASTER_FILE}에 {col} 컬럼이 없습니다.")



    for col in ["timestamp", "up_prob"]:

        if col not in result.columns:

            raise ValueError(f"{RESULT_FILE}에 {col} 컬럼이 없습니다.")



    master = master.copy()

    result = result.copy()



    master["timestamp"] = _to_datetime_utc(master["timestamp"])

    result["timestamp"] = _to_datetime_utc(result["timestamp"])



    start = pd.Timestamp(START_DATE, tz="UTC")

    end = pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)



    master = master[(master["timestamp"] >= start) & (master["timestamp"] < end)].copy()

    result = result[(result["timestamp"] >= start) & (result["timestamp"] < end)].copy()



    if len(master) == 0:

        raise ValueError(f"{MASTER_FILE}에 평가 구간 데이터가 없습니다: {START_DATE} ~ {END_DATE}")



    if len(result) == 0:

        raise ValueError(

            f"{RESULT_FILE}에 평가 구간 LLM 예측 결과가 없습니다: {START_DATE} ~ {END_DATE}\n"

            "먼저 python .\\4_backtest_loop_last_eth.py 를 실행해 해당 기간의 결과 CSV를 생성하세요."

        )



    master["close"] = _safe_float(master["close"])

    if "open" in master.columns:

        master["open"] = _safe_float(master["open"])

    if "high" in master.columns:

        master["high"] = _safe_float(master["high"])

    if "low" in master.columns:

        master["low"] = _safe_float(master["low"])

    if "volume" in master.columns:

        master["volume"] = _safe_float(master["volume"])



    result["up_prob"] = _safe_float(result["up_prob"])

    if "down_prob" in result.columns:

        result["down_prob"] = _safe_float(result["down_prob"])

    else:

        result["down_prob"] = 1.0 - result["up_prob"]



    master = master.dropna(subset=["timestamp", "close"]).sort_values("timestamp")

    result = result.dropna(subset=["timestamp", "up_prob", "down_prob"]).sort_values("timestamp")



    master["date"] = master["timestamp"].dt.floor("D")

    result["date"] = result["timestamp"].dt.floor("D")



    daily_agg = {"timestamp": "last", "close": "last"}

    if "open" in master.columns:

        daily_agg["open"] = "first"

    if "high" in master.columns:

        daily_agg["high"] = "max"

    if "low" in master.columns:

        daily_agg["low"] = "min"

    if "volume" in master.columns:

        daily_agg["volume"] = "sum"



    daily_price = (

        master.groupby("date", as_index=False)

        .agg(daily_agg)

        .sort_values("date")

        .reset_index(drop=True)

    )



    daily_price = daily_price.rename(columns={"close": "current_close"})

    daily_price["next_close"] = daily_price["current_close"].shift(-1)

    daily_price["target_date"] = daily_price["date"].shift(-1)

    daily_price["actual_daily_return"] = (

        daily_price["next_close"] / daily_price["current_close"] - 1.0

    )



    daily_prob = (

        result.groupby("date", as_index=False)

        .agg(

            up_prob_daily=("up_prob", "mean"),

            down_prob_daily=("down_prob", "mean"),

            signal_count=("up_prob", "count"),

            last_prediction_timestamp=("timestamp", "last"),

        )

        .sort_values("date")

        .reset_index(drop=True)

    )



    daily_prob["direction_signal"] = (

        daily_prob["up_prob_daily"] - daily_prob["down_prob_daily"]

    )



    daily = pd.merge(daily_price, daily_prob, on="date", how="inner")

    daily = daily.dropna(subset=["next_close", "actual_daily_return"]).copy()

    daily = daily.sort_values("date").reset_index(drop=True)



    if len(daily) == 0:

        raise ValueError("가격 데이터와 LLM 예측 결과를 일 단위로 병합한 결과가 비어 있습니다.")



    daily["known_daily_return"] = daily["current_close"].pct_change()

    daily["daily_volatility_proxy"] = (

        daily["known_daily_return"]

        .abs()

        .rolling(VOLATILITY_WINDOW, min_periods=3)

        .mean()

    )



    fallback_vol = daily["known_daily_return"].abs().median()

    if pd.isna(fallback_vol) or fallback_vol == 0:

        fallback_vol = 0.03



    daily["daily_volatility_proxy"] = daily["daily_volatility_proxy"].fillna(fallback_vol)



    daily["predicted_daily_return"] = (

        daily["direction_signal"] * daily["daily_volatility_proxy"] * VOLATILITY_SCALE

    ).clip(-MAX_EXPECTED_RETURN, MAX_EXPECTED_RETURN)



    daily["predicted_next_close"] = daily["current_close"] * (

        1.0 + daily["predicted_daily_return"]

    )



    daily["prediction_error"] = daily["predicted_next_close"] - daily["next_close"]

    daily["abs_error"] = daily["prediction_error"].abs()

    daily["squared_error"] = daily["prediction_error"] ** 2

    daily["abs_pct_error"] = daily["abs_error"] / daily["next_close"].replace(0, np.nan)



    daily["actual_direction"] = (daily["actual_daily_return"] > 0).astype(int)

    daily["predicted_direction"] = (daily["predicted_daily_return"] > 0).astype(int)



    daily["date_str"] = daily["date"].dt.strftime("%Y-%m-%d")

    daily["target_date_str"] = daily["target_date"].dt.strftime("%Y-%m-%d")



    return daily





def calculate_daily_metrics(daily: pd.DataFrame) -> dict:

    y_true_price = daily["next_close"].astype(float)

    y_pred_price = daily["predicted_next_close"].astype(float)



    mse = mean_squared_error(y_true_price, y_pred_price)

    rmse = float(np.sqrt(mse))

    mae = mean_absolute_error(y_true_price, y_pred_price)

    mape = float(daily["abs_pct_error"].dropna().mean() * 100)



    try:

        corr = float(np.corrcoef(y_true_price, y_pred_price)[0, 1])

    except Exception:

        corr = np.nan



    try:

        r2 = float(r2_score(y_true_price, y_pred_price))

    except Exception:

        r2 = np.nan



    daily_direction_accuracy = float(

        (daily["actual_direction"] == daily["predicted_direction"]).mean()

    )



    mean_actual_close = float(y_true_price.mean())

    mae_pct_of_price = float(mae / mean_actual_close * 100) if mean_actual_close else np.nan

    rmse_pct_of_price = float(rmse / mean_actual_close * 100) if mean_actual_close else np.nan



    return {

        "evaluation_unit": "1D",

        "period": f"{START_DATE} ~ {END_DATE}",

        "sample_days": int(len(daily)),

        "mse": float(mse),

        "rmse": float(rmse),

        "mae": float(mae),

        "mape_pct": float(mape),

        "mae_pct_of_avg_price": mae_pct_of_price,

        "rmse_pct_of_avg_price": rmse_pct_of_price,

        "actual_predicted_corr": corr,

        "r2_score": r2,

        "daily_direction_accuracy": daily_direction_accuracy,

        "mean_actual_close": mean_actual_close,

        "mean_predicted_close": float(y_pred_price.mean()),

        "mean_abs_daily_return": float(daily["actual_daily_return"].abs().mean()),

    }





def print_metrics(metrics: dict) -> None:

    print()

    print("=" * 78)

    print(" DAILY ETH PRICE-TRACKING EVALUATION")

    print("=" * 78)

    print()

    print("[DATASET]")

    print(f"Evaluation Unit : {metrics['evaluation_unit']}")

    print(f"Period          : {metrics['period']}")

    print(f"Sample Days     : {metrics['sample_days']}")

    print()

    print("[PRICE ERROR METRICS]")

    print(f"MSE             : {_num(metrics['mse'], 4)}")

    print(f"RMSE            : {_num(metrics['rmse'], 4)}")

    print(f"MAE             : {_num(metrics['mae'], 4)}")

    print(f"MAPE            : {metrics['mape_pct']:.2f}%")

    print(f"MAE / Avg Price : {metrics['mae_pct_of_avg_price']:.2f}%")

    print(f"RMSE / Avg Price: {metrics['rmse_pct_of_avg_price']:.2f}%")

    print()

    print("[PRICE TREND FOLLOWING CHECK]")

    print(f"Actual-Predicted Correlation : {_num(metrics['actual_predicted_corr'], 4)}")

    print(f"R2 Score                     : {_num(metrics['r2_score'], 4)}")

    print(f"Daily Direction Accuracy     : {_pct(metrics['daily_direction_accuracy'])}")

    print()

    print("[NOTE]")

    print("Buy & Hold 비교, 누적수익률, Sharpe, Win Rate는 교수님 피드백에 따라 최종 출력에서 제외했습니다.")

    print("최종 평가는 2024-01-01 ~ 2025-04-30 구간의 1일 단위 ETH 가격 추종 성능입니다.")

    print()





def make_charts(daily: pd.DataFrame, metrics: dict) -> None:

    try:

        import matplotlib.pyplot as plt

    except ImportError as exc:

        raise ImportError(

            "그래프 생성을 위해 matplotlib이 필요합니다. pip install matplotlib 실행 후 다시 시도하세요."

        ) from exc



    x = pd.to_datetime(daily["target_date"])



    plt.figure(figsize=(12, 6))

    plt.plot(x, daily["next_close"], marker="o", linewidth=2, label="Actual ETH close")

    plt.plot(x, daily["predicted_next_close"], marker="o", linewidth=2, label="Predicted ETH close")

    plt.title("Daily ETH Actual vs Predicted Close")

    plt.xlabel("Date")

    plt.ylabel("ETH price")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(CHART_ACTUAL_VS_PREDICTED, dpi=220, bbox_inches="tight")

    plt.close()



    plt.figure(figsize=(12, 5))

    plt.bar(x, daily["squared_error"])

    plt.title(f"Daily Squared Error  |  MSE = {metrics['mse']:.2f}")

    plt.xlabel("Date")

    plt.ylabel("(Predicted close - Actual close)^2")

    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(CHART_SQUARED_ERROR, dpi=220, bbox_inches="tight")

    plt.close()



    metric_names = ["MSE", "MAE", "RMSE", "MAPE (%)"]

    metric_values = [

        metrics["mse"],

        metrics["mae"],

        metrics["rmse"],

        metrics["mape_pct"],

    ]



    plt.figure(figsize=(9, 5))

    plt.bar(metric_names, metric_values)

    plt.title("Daily Price Tracking Error Metrics")

    plt.ylabel("Value")

    for i, v in enumerate(metric_values):

        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()

    plt.savefig(CHART_ERROR_METRICS, dpi=220, bbox_inches="tight")

    plt.close()





def main() -> None:

    daily = load_daily_dataset()

    metrics = calculate_daily_metrics(daily)



    daily.to_csv(DAILY_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    pd.DataFrame([metrics]).to_csv(METRICS_OUTPUT_FILE, index=False, encoding="utf-8-sig")



    make_charts(daily, metrics)

    print_metrics(metrics)



    print("[OUTPUT FILES]")

    print(f"Daily prediction data : {DAILY_OUTPUT_FILE}")

    print(f"Daily metrics summary : {METRICS_OUTPUT_FILE}")

    print(f"Actual vs predicted   : {CHART_ACTUAL_VS_PREDICTED}")

    print(f"Squared error chart   : {CHART_SQUARED_ERROR}")

    print(f"Error metrics chart   : {CHART_ERROR_METRICS}")

    print()





if __name__ == "__main__":

    main()

