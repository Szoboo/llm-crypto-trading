# 리스크 관리



## 주문 금액 산정



`risk_manager.py`는 다음 값 중 가장 작은 값을 신규 매수 주문금액으로 사용한다.



```text

base_order_krw

max_order_krw

quote_balance_krw × max_account_allocation

risk_based_notional

```



`risk_based_notional`은 ATR 기반 손절폭과 계정 위험 한도를 이용해 계산한다.



## 손절/익절



신규 매수 시:



```text

stop_distance = max(ATR × atr_stop_multiplier, price × 0.005)

stop_loss = price - stop_distance

take_profit = price + stop_distance × take_profit_r_multiple

```



기본값:



```text

AUTO_TRADING_ATR_STOP_MULTIPLIER=1.5

AUTO_TRADING_TAKE_PROFIT_R_MULTIPLE=2.0

```



## 중복 매수 방지



`position_manager.py`는 아래 조건에서 신규 매수를 차단한다.



- 이미 봇이 관리하는 포지션이 있고 `AUTO_TRADING_ALLOW_PYRAMIDING=false`

- 현재 ETH 평가금액이 `AUTO_TRADING_MAX_POSITION_KRW` 이상

- 남은 매수 가능 금액이 최소 주문금액보다 작음



## 실거래 안전장치



실거래는 다음 조건이 모두 맞아야 가능하다.



```text

AUTO_TRADING_DRY_RUN=false

AUTO_TRADING_CONFIRM_LIVE=YES

UPBIT_ACCESS_KEY 존재

UPBIT_SECRET_KEY 존재

```



또한 이 버전은 다음 설정만 허용한다.



```text

AUTO_TRADING_EXCHANGE=upbit

AUTO_TRADING_MARKET_TYPE=spot

AUTO_TRADING_SYMBOL=ETH/KRW

AUTO_TRADING_TIMEFRAME=4h

AUTO_TRADING_ALLOW_SHORT=false

```

