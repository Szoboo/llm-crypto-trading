# Upbit Long-only 자동매매 시스템 아키텍처



## 목적



이 시스템은 Upbit `ETH/KRW` 현물 마켓에서 4시간봉 마감 데이터를 기준으로 `LONG` 또는 `HOLD`만 판단하는 자동매매 모듈이다. 숏 포지션, 선물, 레버리지는 프로젝트 범위에서 제외한다.



## 전체 흐름



```text

trader.py

  ↓

exchange_client.py

  - Upbit OHLCV 조회

  - KRW/ETH 잔고 조회

  - 시장가 매수/매도 주문

  ↓

signal_engine.py

  - RSI, MACD, SMA, ATR 기반 정량 신호

  - 선택적 Gemini 보조 판단(Vertex AI / ADC 사용)

  - SHORT 없이 LONG/HOLD만 반환

  ↓

risk_manager.py

  - 주문금액 계산

  - 손절가/익절가 계산

  ↓

position_manager.py

  - 현재 포지션 평가

  - 중복 매수 제한

  - 손절/익절 청산 판단

  ↓

state.py

  - 마지막 처리 캔들

  - 진입가, 수량, 손절가, 익절가 저장

  ↓

logger.py / CSV

  - 런타임 로그

  - 거래 판단 로그

```



## 주요 설계 원칙



1. **Long-only**: 최종 주문은 현물 매수 또는 보유 Long 포지션 청산만 허용한다.

2. **Closed candle only**: 진행 중인 4시간봉은 판단에 사용하지 않는다.

3. **Dry-run first**: 기본값은 실제 주문을 보내지 않는 모의거래다.

4. **Fail-fast validation**: 실거래에 필요한 키와 안전 설정이 없으면 실행 초기에 실패한다.

5. **Separation of concerns**: 신호 생성, 리스크 계산, 포지션 관리, 거래소 연결을 분리한다.



## 주요 파일



| 파일 | 책임 |

|---|---|

| `auto_trading/trader.py` | 실행 진입점, 전체 의사결정 흐름 |

| `auto_trading/config.py` | `.env` 설정 로딩 및 검증 |

| `auto_trading/exchange_client.py` | Upbit API 연결 및 주문 |

| `auto_trading/signal_engine.py` | Long/Hold 신호 생성 |

| `auto_trading/risk_manager.py` | 주문금액, 손절/익절 계산 |

| `auto_trading/position_manager.py` | 보유 상태, 중복 매수, 청산 판단 |

| `auto_trading/scheduler.py` | 4시간봉 마감 시각 기준 실행 |

| `auto_trading/logger.py` | 런타임 로그 파일 기록 |

| `auto_trading/backtest_shared_signal.py` | 백테스트에서 동일 신호 엔진 사용 |

