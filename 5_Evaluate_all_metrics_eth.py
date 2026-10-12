import pandas as pd

import numpy as np

import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, roc_auc_score



def calculate_all_metrics(df):



    # -----------------------------

    # 컬럼 체크

    # -----------------------------

    if "ret_strategy" not in df.columns:

        raise ValueError("ret_strategy 컬럼 없음")



    df = df.dropna(subset=["ret_strategy"]).copy()

    if len(df) == 0:

        return None



    # -----------------------------

    # y_pred 생성 

    # -----------------------------

    if "up_prob" in df.columns:

        df["y_pred"] = (df["up_prob"] > 0.5).astype(int)

    else:

        df["y_pred"] = np.nan



    returns = df["ret_strategy"].values



    # ======================

    # 1. Regression Metrics

    # ======================

    if "y_true" in df.columns and "up_prob" in df.columns:

        y_true = df["y_true"].values

        y_prob = df["up_prob"].values



        mse = mean_squared_error(y_true, y_prob)

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(y_true, y_prob)



        mask = y_true != 0

        mape = np.mean(np.abs((y_true[mask] - y_prob[mask]) / y_true[mask])) * 100 if np.sum(mask) > 0 else np.nan

        

        # (보조) AUC-ROC 추가

        try:

            auc_roc = roc_auc_score(y_true, y_prob)

        except ValueError:

            auc_roc = np.nan

    else:

        mse = rmse = mae = mape = auc_roc = np.nan



    # ======================

    # 2. Directional Accuracy & F1 Score

    # ======================

    if "y_true" in df.columns and "y_pred" in df.columns:

        directional_acc = np.mean(df["y_true"] == df["y_pred"])

        

        # F1 Score 계산

        f1 = f1_score(df["y_true"], df["y_pred"], zero_division=0)



        if "pos_strategy" in df.columns:

            non_flat = df[df["pos_strategy"] != 0]

            directional_acc_nf = np.mean(non_flat["y_true"] == non_flat["y_pred"]) if len(non_flat) > 0 else np.nan

        else:

            directional_acc_nf = np.nan

    else:

        directional_acc = directional_acc_nf = f1 = np.nan



    # ======================

    # 3. Trading Metrics

    # ======================



    cumulative_return = (1 + returns).prod() - 1



    n_candles_per_year = 6 * 365

    mean_ret = np.mean(returns)

    std_ret = np.std(returns)



    sharpe = (mean_ret / std_ret) * np.sqrt(n_candles_per_year) if std_ret != 0 else 0



    cumulative_curve = (1 + returns).cumprod()

    rolling_max = pd.Series(cumulative_curve).cummax()

    drawdowns = (cumulative_curve - rolling_max) / rolling_max

    mdd = drawdowns.min()



    trades = returns[returns != 0]

    win_rate = np.mean(trades > 0) if len(trades) > 0 else 0

    trade_ratio = len(trades) / len(df) if len(df) > 0 else 0



    return {

        "mse": mse, "rmse": rmse, "mae": mae, "mape": mape,

        "auc_roc": auc_roc,

        "directional_acc": directional_acc,

        "directional_acc_nf": directional_acc_nf,

        "f1_score": f1,

        "win_rate": win_rate,

        "cumulative_return": cumulative_return,

        "sharpe": sharpe,

        "mdd": mdd,

        "trade_ratio": trade_ratio,

        "total_points": len(df),

        "successful_preds": int(np.sum(df["y_true"] == df["y_pred"])) if "y_true" in df.columns else 0

    }



def print_metrics(metrics):



    if metrics is None:

        print("계산할 데이터가 없습니다.")

        return



    print("--- [Regression Metrics] ---")

    print(f"MSE: {metrics['mse']:.6f}")

    print(f"RMSE: {metrics['rmse']:.6f}")

    print(f"MAE: {metrics['mae']:.6f}")

    if not np.isnan(metrics['mape']):

        print(f"MAPE: {metrics['mape']:.2f}%")

    if not np.isnan(metrics['auc_roc']):

        print(f"(보조) AUC-ROC: {metrics['auc_roc']:.4f}")

    else:

        print("(보조) AUC-ROC: nan")



    print("\n--- [Directional Accuracy] ---")

    print(f"Total Accuracy: {metrics['directional_acc']*100:.2f}%")

    print(f"Non-Flat Accuracy: {metrics['directional_acc_nf']*100:.2f}%")

    print(f"F1 Score: {metrics['f1_score']:.4f}")



    print("\n--- [Trading Performance (Annualized)] ---")

    print(f"Cumulative Return: {metrics['cumulative_return']*100:.2f}%")

    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")

    print(f"Annualized Sharpe Ratio: {metrics['sharpe']:.4f}")

    print(f"Max Drawdown (MDD): {metrics['mdd']*100:.2f}%")

    print(f"Trade Ratio: {metrics['trade_ratio']*100:.2f}%")



    print("\n--- [Summary] ---")

    print(f"Total Candles: {metrics['total_points']}")

    print(f"Correct Predictions: {metrics['successful_preds']}")





if __name__ == "__main__":



    RESULT_FILE = "eth_backtest_result_LLM_fixed.csv"



    if os.path.exists(RESULT_FILE):

        df = pd.read_csv(RESULT_FILE)

        metrics = calculate_all_metrics(df)

        print_metrics(metrics)

    else:

        print(f"파일을 찾을 수 없습니다: {RESULT_FILE}")