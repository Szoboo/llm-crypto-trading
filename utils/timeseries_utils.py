# utils/timeseries_utils.py

import pandas as pd

import numpy as np

import torch 

import re



def timeseries_to_text(df_recent: pd.DataFrame, features: list, max_rows: int = 30) -> str:

    """

    Converts the tail of a DataFrame into a formatted text string for LLM prompts.



    Args:

        df_recent: DataFrame containing the time series data (index should be datetime).

        features: List of column names (features) to include in the text.

        max_rows: Maximum number of recent rows to include.



    Returns:

        A formatted string representing the recent time series data.

    """

    df_tail = df_recent.tail(max_rows)

    text = "Recent Ethereum Data:\n"

    # Ensure specified features exist in the DataFrame

    valid_features = [f for f in features if f in df_tail.columns]

    if not valid_features:

        return "Recent Ethereum Data: No valid features specified or found.\n"



    for index, row in df_tail.iterrows():

        text += f"Date: {index.strftime('%Y-%m-%d')}"

        for feature in valid_features:

            # Format based on feature type (e.g., price vs. volume vs. score)

            if 'price' in feature.lower() or 'close' in feature.lower() or 'high' in feature.lower() or 'low' in feature.lower() or 'open' in feature.lower():

                text += f", {feature}: {row[feature]:.2f}"

            elif 'volume' in feature.lower():

                text += f", {feature}: {row[feature]:.0f}"

            elif 'sentiment' in feature.lower():

                 text += f", {feature}: {row[feature]:.3f}"

            else: # General numeric formatting

                 text += f", {feature}: {row[feature]:.4f}"

        text += "\n"

    return text



def calculate_timeseries_stats(series: pd.Series, top_k_lags: int = 5) -> dict:

    """

    Calculates basic statistics (min, max, median, trend, top lags) for a time series.



    Args:

        series: Pandas Series representing the time series (e.g., closing price).

        top_k_lags: Number of top lags to calculate based on autocorrelation.



    Returns:

        A dictionary containing the calculated statistics.

    """

    stats = {}

    stats['min_val'] = series.min()

    stats['max_val'] = series.max()

    stats['median_val'] = series.median()



    # Calculate trend (simple difference sum)

    trend_val = series.diff().sum()

    stats['trend'] = 'upward' if trend_val > 0 else ('downward' if trend_val < 0 else 'flat')



    # Calculate top lags using FFT (inspired by TimeLLM code)

    try:

        # Ensure series is long enough for FFT/lags calculation

        if len(series) > top_k_lags * 2 : # Heuristic minimum length

            series_torch = torch.tensor(series.values, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Shape [1, 1, T]

            q_fft = torch.fft.rfft(series_torch, dim=-1)

            k_fft = torch.fft.rfft(series_torch, dim=-1)

            res = q_fft * torch.conj(k_fft)

            corr = torch.fft.irfft(res, dim=-1)

            mean_value = torch.mean(corr, dim=1).squeeze(0) # Shape [T]

            # Ensure top_k is not larger than available correlation values

            k = min(top_k_lags, len(mean_value) -1)

            if k > 0:

                 # Exclude lag 0 (self-correlation) when finding top lags

                 _, lags_indices = torch.topk(mean_value[1:], k, dim=-1)

                 stats['lags'] = (lags_indices + 1).tolist() # Add 1 back to get actual lag

            else:

                 stats['lags'] = []

        else:

             print("Warning: Series too short for reliable lag calculation.")

             stats['lags']=[]



    except Exception as e:

        print(f"Error calculating lags: {e}")

        stats['lags'] = [] # Return empty list on error



    return stats



def parse_llm_response(response_text: str):

    """

    Gemini 응답에서 BTC 가격을 안정적으로 추출



    방법

    1. 모든 숫자 추출

    2. 1000 이상 숫자만 필터

    3. 마지막 숫자를 가격으로 사용

    """



    try:

        # 모든 숫자 찾기

        numbers = re.findall(r'\d+(?:\.\d+)?', response_text)



        if not numbers:

            return None



        # float 변환

        numbers = [float(n) for n in numbers]



        # BTC 가격 후보만 필터 (1000 이상)

        price_candidates = [n for n in numbers if n > 1000]



        if not price_candidates:

            return None



        # 마지막 값 사용

        predicted_price = price_candidates[-1]



        return predicted_price



    except Exception:

        return None