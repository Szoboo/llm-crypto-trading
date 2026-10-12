# 운영 가이드



## 1. 설치



```bash

python -m venv .venv

source .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1

pip install -r requirements_auto_trading.txt

```



## 2. 환경변수



`.env.example`을 `.env`로 복사한 뒤 API 키와 설정값을 입력한다.



```bash

cp .env.example .env

```



처음에는 반드시 모의거래로 실행한다.



```env

AUTO_TRADING_DRY_RUN=true

AUTO_TRADING_CONFIRM_LIVE=NO

```



LLM 보조 판단을 사용할 경우 기존 Vertex AI / ADC 구조를 유지한다.



```env

AUTO_TRADING_USE_LLM=true

GCP_PROJECT_ID=project1-491615

GCP_LOCATION=us-central1

AUTO_TRADING_GOOGLE_MODEL=gemini-2.5-flash

```



Gemini 없이 기술지표 단독 판단으로 실행하려면 다음처럼 설정한다.



```env

AUTO_TRADING_USE_LLM=false

```



## 3. 1회 실행



```bash

python -m auto_trading.trader --once

```



## 4. 반복 실행



```bash

python -m auto_trading.trader --loop

```



기본값은 4시간봉 마감 시각 이후 30초에 맞춰 실행된다.



## 5. 로그 파일



| 파일 | 의미 |

|---|---|

| `auto_trading_trades.csv` | 매매 판단 및 주문 결과 |

| `auto_trading_runtime.log` | 런타임 로그 및 예외 |

| `auto_trading_state.json` | 마지막 캔들, 포지션 상태 |



## 6. 서버/tmux 실행 예시



```bash

tmux new -s trading

source .venv/bin/activate

python -m auto_trading.trader --loop

```



세션에서 빠져나오기:



```text

Ctrl + B, 그 다음 D

```



다시 접속:



```bash

tmux attach -t trading

```



## 7. 테스트



```bash

python -m pytest -q

```



## 8. 공유 금지 파일



아래 파일은 API 키나 개인 거래 상태를 포함할 수 있으므로 공유하지 않는다.



```text

.env

auto_trading_state.json

auto_trading_trades.csv

auto_trading_runtime.log

```

