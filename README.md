# ETH News RAG & LLM Decision Support

금융 뉴스와 ETH 가격 데이터를 활용한 캡스톤 팀 프로젝트입니다. 뉴스 감성 분석, 시계열 특성 생성, Gemini 기반 예측 및 전략 판단, 백테스트·평가와 자동매매 모듈을 포함합니다.

## 코드 구성

| 파일·폴더 | 역할 |
| --- | --- |
| `0_gnews_collector_eth.py` | Ethereum 뉴스 수집 및 정제 |
| `1_preprocess_data_eth_.py` | Binance ETH/USDT 4시간봉 수집, 기술 지표 및 뉴스 감성 특성 생성 |
| `master_filtered.py` | 실험 기간에 맞춘 데이터 필터링 |
| `02_direction_predictor.py` | 상승·하락 확률 예측 및 규칙 보정 |
| `3_recommend_strategy_eth.py` | 예측·기술 지표·감성 정보를 활용한 전략 판단 |
| `4_backtest_loop_last_eth.py` | Vertex AI 기반 Gemini 호출 및 백테스트, 응답 파싱·예외 처리 |
| `4_backtest_loop_last__indicator_eth.py` | 기술 지표를 결합한 백테스트 |
| `5_Evaluate_all_metrics_eth.py` | 가격 추적·오차 평가 및 시각화 |
| `6_threshold_search_eth.py` | 현재 평가 스크립트로 안내하는 이전 분석 진입점 |
| `langchain_utils/rag_builder.py` | 시점 기반 뉴스 필터링, FAISS 검색, 시간 감쇠 재정렬 |
| `prompts/`, `utils/` | 예측·전략 프롬프트, 시계열 텍스트화 및 응답 파싱 |
| `auto_trading/` | 주문·포지션·위험 제어, Vertex AI 기반 LLM 신호, 결과 기록 |
| `tests/` | 자동매매 구성·위험 제어·응답 파싱 등의 테스트 |
| `docs/` | 아키텍처, 운영 및 위험 관리 문서 |

## RAG 검색 모듈

`build_rag_context()`는 전달된 의사결정 시점 이전의 최근 7일 뉴스로 검색 대상을 제한합니다. 임베딩으로 FAISS 인덱스를 구성하고, 검색 거리 기반 점수에 시간 감쇠를 적용해 최신성을 반영합니다. 기본적으로 최대 8개 문서를 LLM 입력용 문맥으로 구성합니다.

## 환경 준비

Python 3.10 이상을 사용합니다.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_project.txt -r requirements_auto_trading.txt
```

`.env.example`을 `.env`로 복사하고 본인의 설정을 입력합니다. 실제 인증 파일과 키는 저장소에 포함하지 않습니다. 외부 API 호출에는 비용이 발생할 수 있습니다.

## Vertex AI / Gemini

`4_backtest_loop_last_eth.py`는 Google Cloud 프로젝트·지역과 `vertexai=True` 설정을 통해 Vertex AI의 Gemini를 호출하도록 구성되어 있습니다. `auto_trading/signal_engine.py`도 Vertex AI 사용 환경변수를 설정합니다.

Vertex AI 사용 시 `GCP_PROJECT_ID`, `GCP_LOCATION`과 Google Cloud 인증(ADC)을 준비해야 합니다. 기존 예측·전략 스크립트는 별도의 Gemini API 키 설정을 사용할 수 있습니다. Gemini는 모델이며 Vertex AI는 해당 모델을 호출하는 플랫폼입니다.

## 데이터 및 실행

뉴스 및 가격 CSV는 별도로 준비합니다. 각 파일의 입력 경로와 수집·평가 기간을 확인한 뒤 필요한 단계를 실행합니다.

```bash
python 0_gnews_collector_eth.py
python 1_preprocess_data_eth_.py
python master_filtered.py
python 02_direction_predictor.py
python 3_recommend_strategy_eth.py
```

백테스트와 평가는 `4_backtest_loop_last_eth.py`, `5_Evaluate_all_metrics_eth.py`의 설정을 확인합니다. 주요 데이터는 `eth_news_data.csv`, `eth_master_data.csv`, `eth_master_data_filtered.csv`입니다.

자동매매 실행과 세부 설정은 [자동매매 안내](auto_trading/README_AUTO_TRADING.md)와 [운영 가이드](docs/operation_guide.md)를 참고합니다. 기본 실행은 모의거래이며, 실제 주문에는 별도의 명시적 설정과 거래소 인증이 필요합니다.

## 공개 범위

원본 뉴스·가격 데이터, 실행 로그·거래 기록, 상태 파일, 모델 캐시, 실제 `.env`와 인증 파일, 수정 전 백업 및 일회성 패치 스크립트는 제외했습니다.

프로젝트는 연구 및 실험 목적이며 투자 권유나 수익 보장이 아닙니다. LLM의 확률·Confidence 출력은 실제 결과에 대해 보정된 확률임을 의미하지 않습니다.

## 참고

- 뉴스 감성 모델: `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis`
- 의미 검색: Sentence Transformers / FAISS
- LLM: Google Gemini / Vertex AI / LangChain

패키지와 모델은 각 프로젝트의 이용 조건을 따릅니다.
