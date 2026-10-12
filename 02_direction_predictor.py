import pandas as pd

import os

import numpy as np

import re

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI

from langchain_core.prompts import PromptTemplate



from utils.timeseries_utils import timeseries_to_text



# -----------------------------

# 설정

# -----------------------------

DATA_FILE_ETH = "eth_master_data_filtered.csv"

OUTPUT_RESULT_FILE = "gemini_prediction.txt"



MODEL_NAME_LLM = "gemini-2.5-flash"

LOOKBACK_CANDLES = 96

TEMPERATURE_SETTING = 0.1



# -----------------------------

# 프롬프트

# -----------------------------

DIRECTION_TEMPLATE = """

You are a quantitative crypto trading analyst.



Predict ETH direction for next 4h candle.



[DATA]

{recent_data_text}



Support: {support}

Resistance: {resistance}

Trend slope: {trend_slope}

Trend: {trend}



[Rules]

- RSI > 70 → bearish

- RSI < 30 → bullish

- Distance > 5% → not near

- Follow strong trend



Output format:



UP_probability:

DOWN_probability:

Confidence:

Decision:

Reason:

"""



DirectionPredictPrompt = PromptTemplate(

    input_variables=[

        "recent_data_text", "support", "resistance",

        "trend_slope", "trend"

    ],

    template=DIRECTION_TEMPLATE

)



# -----------------------------

# LLM 결과 파싱

# -----------------------------

def parse_llm_output(text):



    def extract(key):

        match = re.search(f"{key}[: ]+([0-9\\.]+)", text)

        if match:

            val = float(match.group(1))

            if val > 1:  # 40% → 0.4

                val /= 100

            return val

        return 0.5



    return {

        "UP_probability": extract("UP_probability"),

        "DOWN_probability": extract("DOWN_probability"),

        "Confidence": extract("Confidence"),

        "Decision": "UP" if "Decision: UP" in text else "DOWN"

    }



# -----------------------------

# 규칙 강제

# -----------------------------

def enforce_rules(result, df, sentiment):



    rsi = df["RSI_14"].iloc[-1]



    close = df["close"].iloc[-1]

    support = df["Support_1"].iloc[-1]

    resistance = df["Resistance_1"].iloc[-1]



    dist_to_sup = abs(close - support) / close

    dist_to_res = abs(resistance - close) / close



    up = result["UP_probability"]

    down = result["DOWN_probability"]



    # RSI

    if rsi > 70:

        down = max(down, 0.55)

        up = min(up, 0.45)



    elif rsi < 30:

        up = max(up, 0.55)

        down = min(down, 0.45)



    # Support / Resistance

    if dist_to_res < 0.02:

        down += 0.05



    if dist_to_sup < 0.02:

        up += 0.05



    # Sentiment

    if sentiment > 0.05:

        up += sentiment * 0.3

    elif sentiment < -0.05:

        up += sentiment * 0.3



    # 정규화

    total = up + down

    up /= total

    down /= total



    result["UP_probability"] = round(up, 2)

    result["DOWN_probability"] = round(down, 2)

    result["Decision"] = "UP" if up > down else "DOWN"



    return result



# -----------------------------

# 메인 실행

# -----------------------------

def run_direction_prediction():



    load_dotenv()



    df = pd.read_csv(DATA_FILE_ETH, index_col="timestamp", parse_dates=True)



    df_recent = df.tail(LOOKBACK_CANDLES)



    text_data = timeseries_to_text(

        df_recent,

        ["close", "volume", "RSI_14", "MACD"],

        LOOKBACK_CANDLES

    )



    support = df_recent["low"].tail(20).min()

    resistance = df_recent["high"].tail(20).max()



    trend_slope = np.polyfit(range(20), df_recent["close"].tail(20), 1)[0]

    trend = "UP" if trend_slope > 0 else "DOWN"



    # 핵심: sentiment 가져오기

    sentiment = df_recent["sentiment_mean"].iloc[-1]



    if pd.isna(sentiment):

        sentiment = 0



    prompt = DirectionPredictPrompt.format(

        recent_data_text=text_data,

        support=support,

        resistance=resistance,

        trend_slope=trend_slope,

        trend=trend

    )



    llm = GoogleGenerativeAI(

        model=MODEL_NAME_LLM,

        temperature=TEMPERATURE_SETTING

    )



    print("[INFO] Gemini 호출 중...")



    response = llm.invoke(prompt)

    output_text = str(response)



    print("\n=== LLM RAW ===")

    print(output_text)



    parsed = parse_llm_output(output_text)



    # 핵심 수정

    final_result = enforce_rules(parsed, df_recent, sentiment)



    print("\n=== FINAL RESULT ===")

    print(final_result)



    with open(OUTPUT_RESULT_FILE, "w") as f:

        f.write(str(final_result))



    return final_result





if __name__ == "__main__":

    run_direction_prediction()