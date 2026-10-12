# 매매 정책



## 기본 정책



- 거래소: Upbit

- 마켓: `ETH/KRW`

- 시장 유형: 현물

- 기준 봉: 4시간봉

- 허용 진입: Long 매수

- 금지 진입: Short, 선물, 레버리지

- 허용 청산: 보유 중인 현물 ETH 매도



## 신호 정책



`signal_engine.py`는 내부적으로 상승/하락 확률을 계산하지만 최종 매매 신호는 아래 두 가지로 제한한다.



| 내부 방향 | 최종 결정 | 의미 |

|---|---|---|

| UP | LONG | 신규 현물 매수 후보 |

| DOWN | HOLD | 숏 금지 정책에 따라 관망 |

| NEUTRAL | HOLD | 방향성 부족으로 관망 |



하락 신호는 절대 숏 주문으로 변환하지 않는다. 하락 신호 기반 청산은 `.env`의 `AUTO_TRADING_EXIT_ON_DOWNSIDE_SIGNAL=true`일 때만 사용하며, 이 경우에도 청산은 기존 Long 포지션 매도이지 숏 진입이 아니다.



## 포지션 제한



- `AUTO_TRADING_MAX_POSITION_KRW`: 최대 ETH 평가금액 제한

- `AUTO_TRADING_ALLOW_PYRAMIDING=false`: 이미 봇이 관리 중인 Long 포지션이 있으면 추가 매수 차단

- `AUTO_TRADING_MIN_ORDER_KRW=5000`: Upbit KRW 최소 주문금액 기준



## 리스크 관리



신규 매수 시 다음 값을 계산해 상태 파일에 저장한다.



- 진입가

- 예상 ETH 수량

- 진입금액

- 손절가

- 익절가



이후 실행 주기마다 현재가와 비교해 손절/익절 조건을 확인한다.



## LLM 사용 정책



Gemini는 최종 결정자가 아니라 보조 판단자다. 이 프로젝트는 Gemini Developer API key가 아니라 기존 Vertex AI / ADC 구조를 사용한다.



1. 기술지표 기반 확률을 먼저 계산한다.

2. `AUTO_TRADING_USE_LLM=true`이고 Vertex AI ADC 인증 및 `GCP_PROJECT_ID`/`GCP_LOCATION` 설정이 있으면 Gemini 확률을 보조로 반영한다.

3. LLM 실패 시 자동으로 기술지표 단독 판단으로 전환한다.

4. CSV 로그에는 `llm_used`, `llm_failed`, `llm_error`, `signal_source`가 기록된다.

