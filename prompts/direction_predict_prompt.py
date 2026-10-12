from langchain_core.prompts import PromptTemplate



DIRECTION_TEMPLATE = """

You are a top-tier quantitative crypto trading analyst.



Goal:

Predict whether Ethereum (ETH) will move UP or DOWN in the next 4-hour candle.



===========================

[Recent {num_periods} Candles]

{recent_data_text}



===========================

[Market Statistics]

Min: {min_val}

Max: {max_val}

Median: {median_val}

Trend: {trend}



===========================

[Market Structure]

Support: {support}

Resistance: {resistance}

Trend slope: {trend_slope}

Structure: {price_structure}





===========================

[Final Validation Step - MUST FOLLOW]



Before finalizing the answer, verify:



1. If RSI > 70 → DOWN_probability must be higher than UP_probability

2. If RSI < 30 → UP_probability must be higher than DOWN_probability

3. If distance to support/resistance > 5%, you MUST NOT describe it as "near"

4. If trend is downward, UP decision requires strong reversal evidence

5. If any rule is violated, you MUST correct the decision



===========================



===========================

RSI rules:

- RSI > 70 → bearish bias

- RSI < 30 → bullish bias



Support/Resistance:

- Distance < 2% → near

- Distance > 5% → not near



Trend rule:

- Do not go against STRONG trend



Decision constraints:

1. RSI > 70 → DOWN probability ≥ 0.55 unless strong breakout

2. RSI < 30 → UP probability ≥ 0.55 unless strong breakdown

3. Not near S/R → do not use as main reason

4. Strong trend → must follow trend unless clear reversal

5. Decision MUST match probabilities



If reasoning contradicts indicators, adjust decision accordingly.



===========================



Output strictly in this format:



UP_probability: 0.00 - 1.00

DOWN_probability: 0.00 - 1.00

Confidence: 0.00 - 1.00



Decision: UP or DOWN



Reason:

"""