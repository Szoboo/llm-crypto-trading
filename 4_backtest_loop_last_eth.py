import os

import re

import time

from pathlib import Path



import numpy as np

import pandas as pd

from tqdm import tqdm

from dotenv import load_dotenv



from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate



# 설정값

# 설정값

BASE_DIR = Path(__file__).resolve().parent



load_dotenv(BASE_DIR / ".env")

load_dotenv()



MASTER_DATA_FILE = BASE_DIR / "eth_master_data_filtered.csv"



# 기존 실패 결과 파일과 섞이지 않도록 새 파일 사용

START_DATE = "2024-01-01"

END_DATE = "2025-04-30"



BACKTEST_RESULT_FILE = BASE_DIR / "eth_backtest_result_LLM_fulltext_2024_01_2025_04.csv"



MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")



LOOKBACK = 20

FEE_PER_TRADE = 0.0005

SLIPPAGE = 0.0002  # 0.02% 슬리피지 추가



# 전략 파라미터

THRESHOLD = 0.1

CONFIRM_N = 2

COOLDOWN_PERIOD = 3

SAVE_INTERVAL = 5



PROMPT_TEMPLATE = """

You are a strict ETH direction classifier.



Predict ETH direction for the NEXT 4h candle using only the data below.



[DATA]

{data}



Trend: {trend}

ATR: {atr}

Sentiment: {sentiment}



STRICT OUTPUT RULES:

- Do not write analysis.

- Do not use markdown.

- Do not add explanations.

- Output exactly three lines.

- Probabilities must be decimals between 0 and 1.



UP_probability: 0.xx

DOWN_probability: 0.xx

Decision: UP or DOWN

"""



Prompt = PromptTemplate(

    input_variables=["data", "trend", "atr", "sentiment"],

    template=PROMPT_TEMPLATE

)



def load_vertex_config():

    """

    Vertex AI 방식:

    - Gemini Developer API key 사용 안 함

    - GCP_PROJECT_ID / GCP_LOCATION 사용

    - 인증은 gcloud ADC 사용

    """

    project_id = os.getenv("GCP_PROJECT_ID")

    location = os.getenv("GCP_LOCATION", "us-central1")



    if not project_id:

        raise RuntimeError(

            "GCP_PROJECT_ID가 없습니다. .env에 GCP_PROJECT_ID를 설정하세요."

        )



    # langchain-google-genai가 Vertex backend를 쓰도록 명시

    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

    os.environ["GOOGLE_CLOUD_LOCATION"] = location



    return project_id, location



def parse_probs(text):

    """

    Gemini 응답에서 UP/DOWN 확률을 최대한 안정적으로 추출한다.



    1순위: UP_probability / DOWN_probability 정규식 파싱

    2순위: JSON-like 응답에서 up_probability / down_probability 파싱

    3순위: Decision만 있을 경우 보수적 확률로 fallback

    4순위: 분석형 장문 응답일 경우 bullish/bearish 키워드 기반 fallback



    반환:

        up, down, parse_success



    parse_success=False인 경우는 원본 포맷은 깨졌지만 fallback으로 진행했다는 뜻이다.

    백테스트를 중단하지 않고 결과 CSV에 parse_success=0으로 기록한다.

    """

    text = str(text or "")

    normalized = text.replace("\n", " ")



    patterns = [

        (r"UP_probability\s*[:=]\s*([0-9]*\.?[0-9]+)", r"DOWN_probability\s*[:=]\s*([0-9]*\.?[0-9]+)"),

        (r"up_probability\s*['\"]?\s*[:=]\s*([0-9]*\.?[0-9]+)", r"down_probability\s*['\"]?\s*[:=]\s*([0-9]*\.?[0-9]+)"),

        (r"UP\s*probability\s*[:=]\s*([0-9]*\.?[0-9]+)", r"DOWN\s*probability\s*[:=]\s*([0-9]*\.?[0-9]+)"),

        (r"UP\s*[:=]\s*([0-9]*\.?[0-9]+)", r"DOWN\s*[:=]\s*([0-9]*\.?[0-9]+)"),

    ]



    for up_pat, down_pat in patterns:

        try:

            up_match = re.search(up_pat, normalized, flags=re.IGNORECASE)

            down_match = re.search(down_pat, normalized, flags=re.IGNORECASE)

            if up_match and down_match:

                up_val = float(up_match.group(1))

                down_val = float(down_match.group(1))



                if up_val > 1:

                    up_val /= 100

                if down_val > 1:

                    down_val /= 100



                up_val = max(0.0, min(1.0, up_val))

                down_val = max(0.0, min(1.0, down_val))



                total = up_val + down_val

                if total > 0:

                    up_val /= total

                    down_val /= total



                return up_val, down_val, True

        except Exception:

            pass



    # Decision만 있는 경우 fallback

    decision_match = re.search(r"Decision\s*[:=]\s*(UP|DOWN|NEUTRAL)", normalized, flags=re.IGNORECASE)

    if decision_match:

        decision = decision_match.group(1).upper()

        if decision == "UP":

            return 0.60, 0.40, False

        if decision == "DOWN":

            return 0.40, 0.60, False

        return 0.50, 0.50, False



    # 장문 분석 응답 fallback: bearish/bullish 키워드 기반

    lower = normalized.lower()



    bullish_words = [

        "bullish", "uptrend", "upward", "rebound", "breakout",

        "positive momentum", "strong momentum", "rising", "increase"

    ]

    bearish_words = [

        "bearish", "downtrend", "downward", "depreciation", "correction",

        "pullback", "overbought", "declining", "decrease", "weakness"

    ]



    bullish_score = sum(1 for word in bullish_words if word in lower)

    bearish_score = sum(1 for word in bearish_words if word in lower)



    if bullish_score > bearish_score:

        return 0.55, 0.45, False

    if bearish_score > bullish_score:

        return 0.45, 0.55, False



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

    project_id, location = load_vertex_config()



    if not MASTER_DATA_FILE.exists():

        raise FileNotFoundError(f"master dataset을 찾을 수 없습니다: {MASTER_DATA_FILE}")



    df = pd.read_csv(MASTER_DATA_FILE, index_col="timestamp", parse_dates=True)

    df.sort_index(inplace=True)



    # 지정 기간 데이터 필터링: 2024-01-01 ~ 2025-04-30

    if df.index.tz is None:

        df.index = df.index.tz_localize("UTC")

    else:

        df.index = df.index.tz_convert("UTC")



    start = pd.Timestamp(START_DATE, tz="UTC")

    end = pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)



    df = df[(df.index >= start) & (df.index < end)].copy()



    if len(df) <= LOOKBACK + 1:

        raise ValueError(

            f"백테스트 구간 데이터가 부족합니다: {START_DATE} ~ {END_DATE}, rows={len(df)}"

        )



    print(f"[INFO] Backtest period: {START_DATE} ~ {END_DATE}")

    print(f"[INFO] Rows to test: {len(df)}")

    print(f"[INFO] Result file: {BACKTEST_RESULT_FILE.name}")



    llm = ChatGoogleGenerativeAI(

    model=MODEL_NAME,

    project=project_id,

    location=location,

    vertexai=True,

    temperature=0.1

)



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



        response_text = ""

        up, down, parse_success = 0.5, 0.5, False



        try:

            res = chain.invoke(prompt_input)

            response_text = getattr(res, "content", str(res))

            up, down, parse_success = parse_probs(response_text)



            if not parse_success:

                print("\n[PARSE WARNING] Gemini 응답이 지정 포맷과 달라 fallback 확률을 사용합니다.")

                print(response_text[:700])



        except Exception as e:

            print(f"\n[LLM ERROR] {type(e).__name__}: {e}")

            raise



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

            "down_prob": down,

            "y_true": 1 if actual_return > 0 else 0,

            "actual_return": actual_return,

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

        prev_pos_strategy = pos_strategy



        if len(results) % SAVE_INTERVAL == 0:

            pd.DataFrame(results).to_csv(BACKTEST_RESULT_FILE, index=False)



    pd.DataFrame(results).to_csv(BACKTEST_RESULT_FILE, index=False)

    print(f"[COMPLETE] Results saved to {BACKTEST_RESULT_FILE}")



if __name__ == "__main__":

    run_backtest()