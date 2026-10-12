import pandas as pd

import os

import time

import re

import numpy as np

from tqdm import tqdm

from dotenv import load_dotenv



from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate

from google.cloud import aiplatform



# 설정값

MASTER_DATA_FILE = "eth_master_data_filtered.csv"

BACKTEST_RESULT_FILE = "eth_backtest_result_LLM_fixed.csv"



LOOKBACK = 20

FEE_PER_TRADE = 0.0005

SLIPPAGE = 0.0002  # 0.02% 슬리피지 추가



# 전략 파라미터

THRESHOLD = 0.1

CONFIRM_N = 2

COOLDOWN_PERIOD = 3

SAVE_INTERVAL = 5



PROMPT_TEMPLATE = """

Predict ETH direction for NEXT 4h candle.



[DATA]

{data}



Trend: {trend}

ATR: {atr}

Sentiment: {sentiment}



Output EXACTLY:



UP_probability: 0.xx

DOWN_probability: 0.xx

Decision: UP or DOWN

"""



Prompt = PromptTemplate(

    input_variables=["data", "trend", "atr", "sentiment"],

    template=PROMPT_TEMPLATE

)



def parse_probs(text):

    try:

        up = re.search(r"UP_probability[: ]*([0-9]*\.?[0-9]+)", text)

        down = re.search(r"DOWN_probability[: ]*([0-9]*\.?[0-9]+)", text)

        if up and down:

            up_val = float(up.group(1))

            down_val = float(down.group(1))

            if up_val > 1: up_val /= 100

            if down_val > 1: down_val /= 100

            return up_val, down_val, True

    except:

        pass

    return 0.5, 0.5, False



def strategy_llm(up, down, df, sentiment,

                 prev_position, prev_direction,

                 confirm_count, cooldown):

    

    # 1. 쿨다운 감소

    if cooldown > 0:

        cooldown -= 1



    if df is None or len(df) == 0 or "RSI_14" not in df.columns:

        return prev_position, prev_direction, confirm_count, cooldown



    rsi = df["RSI_14"].iloc[-1]

    diff = up - down



    # 2. Threshold 미달 시 기존 포지션 유지 (전략의 보수성)

    if abs(diff) < THRESHOLD:

        return prev_position, prev_direction, 0, cooldown



    direction = 1 if diff > 0 else -1



    # 3. RSI 필터

    if (direction == 1 and rsi > 75) or (direction == -1 and rsi < 25):

        return prev_position, prev_direction, confirm_count, cooldown



    # 4. Confirm 로직

    if direction == prev_direction:

        confirm_count += 1

    else:

        confirm_count = 1



    if confirm_count < CONFIRM_N:

        return prev_position, direction, confirm_count, cooldown



    # 5. 진입 제어 (쿨다운)

    if cooldown > 0:

        return prev_position, direction, confirm_count, cooldown



    # 6. 포지션 변경 시 쿨다운 설정

    if prev_position != direction:

        cooldown = COOLDOWN_PERIOD



    return direction, direction, confirm_count, cooldown



def run_backtest():

    load_dotenv()

    aiplatform.init(project=os.getenv("GCP_PROJECT_ID"), location=os.getenv("GCP_LOCATION", "asia-northeast3"))



    df = pd.read_csv(MASTER_DATA_FILE, index_col="timestamp", parse_dates=True)

    df.sort_index(inplace=True)

    

    # 최근 30일 데이터 필터링

    cutoff_date = df.index.max() - pd.Timedelta(days=30)

    df = df[df.index >= cutoff_date]



    print(f"[INFO] Rows to test: {len(df)}")



    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    chain = Prompt | llm



    # 변수 초기화

    results = []

    balance_strategy = 1000

    prev_pos_strategy = 0

    prev_direction = 0

    confirm_count = 0

    cooldown = 0



    # Resume 기능

    if os.path.exists(BACKTEST_RESULT_FILE):

        df_existing = pd.read_csv(BACKTEST_RESULT_FILE)

        results = df_existing.to_dict("records")

        last_ts = pd.to_datetime(results[-1]["timestamp"])

        start_idx = df.index.get_loc(last_ts) + 1

        

        last = results[-1]

        balance_strategy = last["balance_strategy"]

        prev_pos_strategy = last["pos_strategy"]

        prev_direction = last["prev_direction"]

        confirm_count = last["confirm_count"]

        cooldown = last["cooldown"]

        print(f"[RESUME] Starting from {last_ts}")

    else:

        start_idx = LOOKBACK + 1



    for i in tqdm(range(start_idx, len(df))):

        current_df = df.iloc[i - LOOKBACK:i]

        prev_close = current_df["close"].iloc[-1]

        next_close = df.iloc[i]["close"]

        actual_return = (next_close - prev_close) / prev_close

        sentiment = current_df["sentiment_mean"].iloc[-1]



        # LLM 호출

        text_data = current_df[["close", "volume", "RSI_14", "MACD"]].to_string()

        prompt_input = {

            "data": text_data,

            "trend": "UP" if prev_close > current_df["close"].mean() else "DOWN",

            "atr": "HIGH" if current_df["ATR_14"].iloc[-1] > current_df["ATR_14"].mean() else "LOW",

            "sentiment": sentiment

        }



        try:

            res = chain.invoke(prompt_input)

            up, down, parse_success = parse_probs(res.content)

        except:

            up, down, parse_success = 0.5, 0.5, False



        # --- 포지션 결정 분리 ---

        

        # 1. 전략 포지션 (기존 로직)

        pos_strategy, prev_direction, confirm_count, cooldown = strategy_llm(

            up, down, current_df, sentiment, prev_pos_strategy, prev_direction, confirm_count, cooldown

        )

        

        # 2. 순수 Binary 포지션 (항상 참여)

        pos_binary = 1 if up > down else -1

        

        # 3. Threshold 포지션 (관망 포함)

        if abs(up - down) < THRESHOLD:

            pos_threshold = 0

        else:

            pos_threshold = 1 if up > down else -1



        # --- 수익률 및 비용 계산 (슬리피지 포함) ---

        def calc_trade_ret(pos, prev_pos, ret):

            fee_costs = abs(pos - prev_pos) * (FEE_PER_TRADE + SLIPPAGE)

            return (ret * pos) - fee_costs



        ret_strategy = calc_trade_ret(pos_strategy, prev_pos_strategy, actual_return)

        balance_strategy *= (1 + ret_strategy)



        # 결과 기록

        results.append({

            "timestamp": df.index[i],

            "up_prob": up,

            "y_true": 1 if actual_return > 0 else 0,

            # 포지션 정보

            "pos_strategy": pos_strategy,

            "pos_binary": pos_binary,

            "pos_threshold": pos_threshold,

            # 수익률 정보

            "ret_strategy": ret_strategy,

            "balance_strategy": balance_strategy,

            # 상태 정보

            "prev_direction": prev_direction,

            "confirm_count": confirm_count,

            "cooldown": cooldown,

            "parse_success": 1 if parse_success else 0

        })



        if len(results) % SAVE_INTERVAL == 0:

            pd.DataFrame(results).to_csv(BACKTEST_RESULT_FILE, index=False)



    pd.DataFrame(results).to_csv(BACKTEST_RESULT_FILE, index=False)

    print(f"[COMPLETE] Results saved to {BACKTEST_RESULT_FILE}")



if __name__ == "__main__":

    run_backtest()