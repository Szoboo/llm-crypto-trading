import pandas as pd

import os

import re

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI

from langchain_core.prompts import PromptTemplate



# ==========================================

# 프롬프트

# ==========================================

STRATEGY_TEMPLATE = """



[입력 데이터]



방향 예측 결과 (확률 + 이유 포함):

{prediction_result}



기술 지표:

{technical_indicators}



뉴스 감정 점수 (FinBERT 기반):

{news_sentiment}



뉴스 존재 여부:

{has_news}



[규칙]



1. UP 확률이 높으면 LONG 우선 고려

2. DOWN 확률이 높더라도 "조정(pullback)"이면 SHORT 제한

3. RSI > 70 → LONG 제한 (과매수)

4. RSI < 30 → SHORT 제한 (과매도)

5. sentiment > 0.05 → LONG 가중

6. sentiment < -0.05 → SHORT 가중

7. 뉴스 없으면 sentiment 영향 무시

8. 확률 차이가 작으면 HOLD 고려



[출력]



Decision: LONG or SHORT or HOLD



Reason:

"""



GeminiStrategyPrompt = PromptTemplate(

    template=STRATEGY_TEMPLATE,

    input_variables=[

        "prediction_result",

        "technical_indicators",

        "news_sentiment",

        "has_news"

    ]

)



# ==========================================

# Decision 파싱 (핵심 수정)

# ==========================================

def parse_decision(text):



    # 1차: 정확한 Decision 라인 추출

    match = re.search(r"Decision:\s*(LONG|SHORT|HOLD)", text, re.IGNORECASE)

    if match:

        return match.group(1).upper()



    # 2차: fallback (최후)

    text_upper = text.upper()



    if "HOLD" in text_upper:

        return "HOLD"

    elif "LONG" in text_upper:

        return "LONG"

    elif "SHORT" in text_upper:

        return "SHORT"



    return "ERROR"



# ==========================================

# 메인 실행

# ==========================================

if __name__ == "__main__":



    print("=== Strategy Recommendation ===")



    load_dotenv()



    # -----------------------------

    # 2번 결과 그대로 사용

    # -----------------------------

    if not os.path.exists("gemini_prediction.txt"):

        print("예측 파일 없음 (2번 코드 먼저 실행)")

        exit()



    with open("gemini_prediction.txt", "r", encoding="utf-8") as f:

        prediction_text = f.read()



    # -----------------------------

    # 데이터 로드

    # -----------------------------

    master_df = pd.read_csv(

        "eth_master_data.csv",

        index_col="timestamp",

        parse_dates=True

    )



    last_row = master_df.iloc[-1]



    # -----------------------------

    # 기술 지표

    # -----------------------------

    rsi_val = last_row["RSI_14"]

    macd_val = last_row["MACD"]



    tech_indicators = f"RSI={rsi_val:.2f}, MACD={macd_val:.4f}"



    # -----------------------------

    # sentiment

    # -----------------------------

    sentiment = last_row.get("sentiment_mean", 0)

    has_news = last_row.get("has_news", 0)



    if pd.isna(sentiment):

        sentiment = 0



    sentiment = float(sentiment)

    has_news = int(has_news)



    # -----------------------------

    # LLM 호출

    # -----------------------------

    llm = GoogleGenerativeAI(

        model="models/gemini-2.5-flash",

        temperature=0.2

    )



    chain = GeminiStrategyPrompt | llm



    response = chain.invoke({

        "prediction_result": prediction_text,

        "technical_indicators": tech_indicators,

        "news_sentiment": round(sentiment, 3),

        "has_news": has_news

    })



    text = getattr(response, "content", str(response))



    print("\n=== STRATEGY RAW ===")

    print(text)



    # -----------------------------

    # -----------------------------

    decision = parse_decision(text)



    print("\n=== FINAL DECISION ===")

    print(decision)



    # -----------------------------

    # 저장

    # -----------------------------

    result_df = pd.DataFrame([{

        "timestamp": master_df.index[-1],

        "decision": decision,

        "sentiment": sentiment

    }])



    result_df.to_csv("strategy_result.csv", index=False)



    print("\n[OK] 전략 결과 저장 완료")