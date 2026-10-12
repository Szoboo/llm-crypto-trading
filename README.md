# ETH News RAG & LLM Decision Support

금융 뉴스와 ETH 4시간봉 가격 데이터를 활용한 캡스톤 팀 프로젝트의 코드 모음입니다. 뉴스 감성 분석, 시계열 특성 생성, Gemini 기반 예측 및 전략 판단, 백테스트와 평가 모듈을 포함합니다.

## 프로젝트와 개인 기여

팀 프로젝트이며 전체 시스템의 단독 개발을 의미하지 않습니다. 정현수의 담당 내용은 RAG 뉴스 검색, LLM 예측·전략 프롬프트, 뉴스·가격 반응 기반 감성 특성 및 위험 제어·백테스트 개선입니다. 이 설명은 프로젝트 참여자의 역할 설명을 기준으로 작성했습니다.

## 코드 구성

| 파일 | 역할 |
| --- | --- |
| `0_gnews_collector_eth.py` | Ethereum 관련 뉴스 수집, 중복 제거 및 정제 |
| `1_preprocess_data_eth_.py` | Binance ETH/USDT 4시간봉 수집, 기술 지표 및 DeBERTa 기반 뉴스 감성 특성 생성 |
| `master_filtered.py` | 실험 기간에 맞춘 통합 데이터 필터링 |
| `02_direction_predictor.py` | 최근 96개 캔들 입력 기반 상승·하락 확률 예측과 규칙 보정 |
| `3_recommend_strategy_eth.py` | 방향 예측 및 기술 지표·감성 정보를 활용한 LONG/SHORT/HOLD 전략 판단 |
| `4_backtest_loop_last_eth.py` | LLM 확률 기반 전략 백테스트 |
| `4_backtest_loop_last__indicator_eth.py` | 기술 지표를 결합한 전략 백테스트 |
| `5_Evaluate_all_metrics_eth.py` | 예측 및 거래 성과 평가 |
| `langchain_utils/rag_builder.py` | 시점 기반 뉴스 필터링, FAISS 검색, 시간 감쇠 재정렬 |
| `prompts/` | 방향·가격 예측 및 전략 판단 프롬프트 |
| `utils/timeseries_utils.py` | 시계열 텍스트화, 통계 요약, FFT 기반 lag 계산 및 응답 파싱 |

## RAG 검색 모듈

`build_rag_context()`는 호출자가 제공한 의사결정 시점 이전의 최근 7일 뉴스로 검색 대상을 제한합니다. Hugging Face 임베딩 객체를 외부에서 전달받아 FAISS 인덱스를 구성하며, 검색 거리로 계산한 `1 / (1 + distance)` 점수에 `exp(-0.15 * 경과일수)`를 곱해 재정렬합니다. 기본적으로 최대 8개 문서를 LLM 입력용 문맥으로 구성합니다.

## 환경 준비

Python 3.10 이상이 필요합니다. 필요한 패키지는 `requirements.txt`를 통해 설치합니다.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`.env.example`을 `.env`로 복사하고 본인의 `GOOGLE_API_KEY`를 입력합니다. 백테스트 스크립트는 `aiplatform.init()`도 호출하므로 Google Cloud 프로젝트 및 인증 설정이 필요할 수 있습니다. 모델명과 수집 기간은 각 파일의 상수를 확인해야 합니다. Google API 호출에는 비용이 발생할 수 있습니다.

## 실행 순서

프로젝트 루트에서 실행합니다. 뉴스 및 가격 데이터는 별도로 준비해야 합니다. 첫 두 단계는 외부 서비스에서 데이터를 수집하고 모델을 다운로드하며, 실행 시간이 길어질 수 있습니다.

```bash
python 0_gnews_collector_eth.py
python 1_preprocess_data_eth_.py
python master_filtered.py
python 02_direction_predictor.py
python 3_recommend_strategy_eth.py
```

백테스트가 필요하면 두 백테스트 파일 중 실험 목적에 맞는 하나를 실행하고 `5_Evaluate_all_metrics_eth.py`의 입력 파일 설정을 확인합니다.

## 데이터 파일

원본 뉴스 텍스트, 실제 `.env`, 모델 캐시, 실행 로그 및 백테스트 CSV는 공개용 사본에서 제외했습니다. 뉴스 데이터를 직접 수집하거나 사용 권한이 있는 데이터로 구성해야 합니다.

- `eth_news_data.csv`: title, description, content, published date, url, publisher, real_url, news_count, candle_4h, full_text, sentiment_score
- `eth_master_data.csv` / `eth_master_data_filtered.csv`: timestamp, OHLCV, SMA_12, SMA_48, RSI_14, MACD, MACD_SIGNAL, ATR_14, Resistance_1, Support_1, Target_Up, sentiment_mean/count/weighted, has_news 및 sentiment lag1 컬럼

## 실험 과정과 현재 한계

프로젝트 진행 중에는 보수적인 매매 조건에 의해 HOLD가 반복되는 문제를 경험했습니다.

LLM이 출력한 확률과 Confidence는 경험적인 규칙으로 조정된 값이며, calibration을 검증한 확률이 아닙니다. RAG의 시점 필터만으로 전체 파이프라인의 미래 정보 누수가 없다고 보장할 수는 없으며, 캔들 시점과 감성 특성의 이용 가능 시점도 별도 점검이 필요합니다. 가격 파서는 1000보다 큰 숫자 중 마지막 값을 선택하므로 입력 가격대와 응답 형식에 영향을 받습니다.

본 프로젝트는 연구 및 실험 목적으로 작성되었으며, 수익성이나 성능을 보장하는 자료가 아닙니다.

## 참고

- 금융 뉴스 감성 모델: `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis`
- 의미 검색: Sentence Transformers / FAISS
- LLM 연동: LangChain / Google Gemini

패키지와 모델은 각 프로젝트의 이용 조건을 따릅니다. 팀 코드의 재사용 라이선스는 아직 지정하지 않았습니다.
