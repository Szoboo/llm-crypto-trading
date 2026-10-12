# Upbit ETH/KRW Long-Only 4H Auto Trading Module



이 모듈은 캡스톤디자인 프로젝트용 자동매매 실행 레이어입니다. 기존 ETH/USDT 예측·백테스트 파이프라인을 직접 훼손하지 않고, `auto_trading/` 폴더 안에서 Upbit `ETH/KRW` 현물 Long-only 자동매매만 담당합니다.



## 1. 정책



- 거래소: Upbit

- 마켓: `ETH/KRW`

- 주기: 4시간봉

- 포지션 방향: Long only

- SHORT 신호: 주문하지 않고 `HOLD`로 변환

- 기본 실행: 모의거래

- 실거래: `.env`에서 명시적으로 켜야 함

- Gemini 호출: Vertex AI / ADC 사용. Gemini Developer API key 사용 안 함

- Gemini 실패 또는 비활성화 시: 기술지표 단독 판단으로 fallback



## 2. 설치



```bash

pip install -r requirements_auto_trading.txt

```



`.env.example`을 `.env`로 복사합니다.



```bash

copy .env.example .env

```



Windows PowerShell에서는 다음도 가능합니다.



```powershell

Copy-Item .env.example .env

```



Linux/macOS에서는 다음을 사용할 수 있습니다.



```bash

cp .env.example .env

```



## 3. Vertex AI 인증



```bash

gcloud auth application-default login

gcloud config set project project1-491615

gcloud auth application-default set-quota-project project1-491615

gcloud services enable aiplatform.googleapis.com

```



`.env`에는 다음 값이 필요합니다.



```env

GCP_PROJECT_ID=project1-491615

GCP_LOCATION=us-central1

AUTO_TRADING_GOOGLE_MODEL=gemini-2.5-flash

AUTO_TRADING_USE_LLM=true

```



Gemini 없이 기술지표만으로 테스트하려면 다음처럼 설정합니다.



```env

AUTO_TRADING_USE_LLM=false

```



## 4. 안전한 기본값



```env

AUTO_TRADING_DRY_RUN=true

AUTO_TRADING_CONFIRM_LIVE=NO

AUTO_TRADING_SYMBOL=ETH/KRW

AUTO_TRADING_TIMEFRAME=4h

AUTO_TRADING_BASE_ORDER_KRW=10000

AUTO_TRADING_MAX_ORDER_KRW=30000

AUTO_TRADING_MIN_ORDER_KRW=5000

```



`DRY_RUN=true`이면 실제 주문은 나가지 않고 `auto_trading_trades.csv`에 의사결정과 가상 주문만 기록됩니다.



## 5. 실행



1회 실행:



```bash

python -m auto_trading.trader --once

```



반복 실행:



```bash

python -m auto_trading.trader --loop

```



중복 캔들 방지:



- 마지막으로 처리한 4시간봉 timestamp는 `auto_trading_state.json`에 저장됩니다.

- 같은 4시간봉은 다시 주문하지 않습니다.

- 테스트 목적으로 같은 캔들을 강제로 다시 처리하려면:



```bash

python -m auto_trading.trader --once --force

```



## 6. 실거래 전환



실거래는 다음을 모두 만족해야 합니다.



```env

AUTO_TRADING_DRY_RUN=false

AUTO_TRADING_CONFIRM_LIVE=YES

UPBIT_ACCESS_KEY=발급받은_키

UPBIT_SECRET_KEY=발급받은_시크릿

```



권장 순서:



1. API 키는 처음에 조회 권한만 부여

2. `--once`로 데이터 수집/신호 생성 확인

3. `--loop`로 최소 며칠간 모의거래 로그 확인

4. 주문 권한을 켠 뒤에도 최소 주문금액으로 시작



## 7. 신호 처리 방식



`signal_engine.py`는 UP/DOWN 확률을 계산하지만 최종 주문 방향은 다음처럼 제한합니다.



- UP 우세 + 기준 충족 → `LONG` → 시장가 매수

- DOWN 우세 → `HOLD`

- 확률 차이 부족 → `HOLD`

- 신뢰도 부족 → `HOLD`

- RSI 과열 → 신규 Long 차단



Gemini는 최종 결정자가 아니라 보조 판단자입니다. 기술지표 확률과 Gemini 확률을 섞어 사용하며, Vertex AI 호출이 실패하면 자동매매 사이클은 중단되지 않고 기술지표 단독 판단으로 전환됩니다.



따라서 이 프로젝트에서는 숏 포지션, 공매도, 선물 레버리지를 사용하지 않습니다.



## 8. 기존 ETH/USDT 백테스트와의 관계



이 모듈은 실행 레이어입니다. 기존 `4_backtest_loop_last_eth.py`와 `5_Evaluate_all_metrics_eth.py`의 성능 비교 결과를 대체하지 않습니다.



보고서에서는 다음처럼 구분하세요.



- 연구/평가 레이어: ETH/USDT 4H, full_text sentiment, Gemini 방향 예측, 백테스트 지표

- 실행/자동매매 레이어: Upbit ETH/KRW 4H, Long-only spot, DRY_RUN 주문 로그



## 9. 로그



실행 결과는 `auto_trading_trades.csv`에 누적됩니다.



주요 컬럼:



- `last_closed_candle_timestamp`

- `decision`

- `raw_direction`

- `up_probability`

- `down_probability`

- `confidence`

- `signal_source`

- `llm_used`

- `llm_failed`

- `llm_error`

- `llm_raw_response`

- `action`

- `notional_krw`

- `entry_amount_eth`

- `entry_notional_krw`

- `unrealized_pnl_krw`

- `unrealized_pnl_pct`

- `stop_loss_krw`

- `take_profit_krw`

- `exit_plan_action`

- `exit_plan_reason`

- `reason`



## 10. 단순 실행 환경 권장안



### 개발/시연용



- OS: Windows 10/11, macOS, Ubuntu 중 하나

- Python: 3.10 또는 3.11 권장

- 실행 방식: VS Code 터미널에서 `python -m auto_trading.trader --once`

- 목적: 코드 시연, 로그 확인, 발표용 결과 생성



### 장시간 테스트용



- OS: Ubuntu 22.04 LTS 서버 또는 개인 PC

- Python: 3.10 또는 3.11

- 실행 방식: `python -m auto_trading.trader --loop`

- 네트워크: 안정적인 인터넷 필요

- 로그 관리: `auto_trading_trades.csv`, `auto_trading_state.json` 백업



### 실거래 최소 권장안



- 서버: 항상 켜져 있는 클라우드 VM 또는 개인 미니 PC

- 권한: API 키는 필요한 권한만 부여, 출금 권한 금지

- 자금: 캡스톤 테스트용 소액

- 운영: 에러 로그 확인, 주문 내역 수동 검증



## 11. 추가 문서



```text

docs/architecture.md

docs/trading_policy.md

docs/risk_management.md

docs/operation_guide.md

```



## 12. 주의



이 코드는 교육/연구용 예시입니다. 수익을 보장하지 않으며, 실거래 손실 책임은 사용자에게 있습니다. 반드시 모의거래 로그와 실제 업비트 주문 규칙을 검증한 뒤 소액으로만 테스트하세요.



`.env`, API key, 서비스 계정 JSON은 절대 ZIP, GitHub, 노션, 카카오톡 등에 공유하지 마세요.

