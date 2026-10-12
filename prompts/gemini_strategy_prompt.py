from langchain_core.prompts import PromptTemplate



STRATEGY_TEMPLATE = """

당신은 Ethereum 단기 트레이딩 전문 헤지펀드 전략가입니다.



가능한 포지션은 두 가지입니다.



LONG

SHORT



관망은 허용되지 않습니다.



================================================



[입력 데이터]



가격 예측 결과

{prediction_result}



현재 기술적 지표

{technical_indicators}



최근 뉴스

{news_context}



================================================



[분석 절차]



1. 예측 수익률 계산

현재 가격 대비 상승 또는 하락 예상폭 계산



2. 기술적 지표 분석

- RSI 상태

- MACD 방향

- 단기 추세 강도



3. 뉴스 영향 분석

시장 심리 평가



4. 리스크 분석

- 높은 변동성 여부

- 손실 가능성 평가



================================================



[포지션 결정 규칙]



LONG 조건



- 예측 수익률 > +0.8%

- RSI < 70

- MACD 상승

- 뉴스 중립 이상



SHORT 조건



- 예측 수익률 < -0.8%

- RSI > 35

- MACD 하락

- 뉴스 부정



================================================



[Confidence 계산 규칙]



높은 Confidence (0.7 ~ 1.0)



- 예측 방향

- 기술 지표

- 뉴스



이 세 요소가 같은 방향일 때



중간 Confidence (0.4 ~ 0.7)



- 일부 지표 충돌



낮은 Confidence (0.0 ~ 0.4)



- 근거 부족

- 지표 충돌 심함

- 시장 방향 불명확



근거가 부족할 때는 반드시 낮은 Confidence를 부여하십시오.



================================================



[리스크 관리]



1. 변동성이 높을 경우 보수적으로 판단

2. 손실 가능성이 크면 반대 포지션 선택

3. 연속 손실 가능성이 있는 경우 방향 신중 판단



================================================



[출력]



Decision: LONG 또는 SHORT



Confidence: 0.0 ~ 1.0



Reason:

- Prediction 근거

- Technical 근거

- News 근거 (반드시 사용한 뉴스 문장을 인용할 것)

"""



GeminiStrategyPrompt = PromptTemplate(

    template=STRATEGY_TEMPLATE,

    input_variables=["prediction_result", "technical_indicators", "news_context"]

)



def get_gemini_strategy(prediction_context, tech_indicators, news_context, llm):



    chain = GeminiStrategyPrompt | llm



    response = chain.invoke({

        "prediction_result": prediction_context,

        "technical_indicators": tech_indicators,

        "news_context": news_context

    })



    text = getattr(response, "content", str(response)).strip()



    decision = "ERROR"



    if "LONG" in text.upper():

        decision = "LONG"



    elif "SHORT" in text.upper():

        decision = "SHORT"



    return decision, text