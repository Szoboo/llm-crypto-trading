import pandas as pd

import numpy as np

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document

from tqdm.auto import tqdm

import time

import torch 

import traceback

from typing import List, Tuple

from datetime import datetime, timedelta, timezone



# v1.1 신선도 설정 (백테스트/실전 공통 상수)

RAG_RECENT_DAYS = 7     # v1.1 기본값: 최근 7일만 검색

RAG_MAX_DOCS = 8        # 컨텍스트로 넣을 문서 최대 개수

RAG_TIME_DECAY = 0.15   # 시간 감쇠 계수 (하루당 가중치 하락)



# --- (기존 v1.0 함수들: 호환성을 위해 유지) ---



def prepare_news_documents(news_df_recent: pd.DataFrame) -> List[Document]:

    """

    [v1.2] 입력받은 DataFrame을 RAG Document 객체 리스트로 변환합니다.

    (이 함수는 더 이상 날짜 필터링을 직접 수행하지 않습니다.)

    (v1.1 패치에서는 build_rag_context 함수가 이 역할을 대체합니다)

    """

    print(f"[INFO] RAG: 뉴스 {len(news_df_recent)}개를 Document로 변환 중...")

    try:

        if news_df_recent.empty:

            print("[WARN] RAG: Document로 변환할 뉴스가 없습니다.")

            return []



        temp_date = pd.to_datetime(news_df_recent['published date'], utc=True, errors='coerce')

        recent_news_df = news_df_recent.copy() # 복사본 사용

        recent_news_df['date_utc_str'] = temp_date.dt.tz_convert(None).dt.floor('D').astype(str)

        recent_news_df['unique_id'] = recent_news_df['date_utc_str'] + "||" + recent_news_df['title'].fillna('').str.strip()

        recent_news_df.drop_duplicates(subset=['unique_id'], inplace=True, keep='first') 



        documents = []

        for _, row in recent_news_df.iterrows():

            content = f"Title: {row['title']}\nDescription: {row['description']}"

            metadata = {"date": str(row['published date'])} 

            documents.append(Document(page_content=content, metadata=metadata))



        print(f"[OK] RAG: Document 변환 완료 ({len(documents)}개).")

        return documents

    except Exception as e:

        print(f"[ERROR] RAG 오류: 뉴스 데이터 처리(prepare) 중 오류 발생 - {e}")

        return []



def build_rag_vectorstore(documents: List[Document], embedding_model: HuggingFaceEmbeddings) -> FAISS | None:

    """

    Document 리스트와 초기화된 임베딩 모델을 받아 FAISS 벡터 저장소를 구축합니다.

    (v1.1 패치에서는 build_rag_context 함수가 이 역할을 대체합니다)

    """

    if not documents:

        print("[WARN] RAG: 벡터 저장소를 만들 문서가 없습니다.")

        return None

        

    print(f"[INFO] RAG: 벡터 저장소 구축 중... (문서 {len(documents)}개)")

    start_time = time.time()

    try:

        vectorstore = FAISS.from_documents(documents, embedding=embedding_model)

        end_time = time.time()

        print(f"[OK] RAG: 벡터 저장소 구축 완료 (소요 시간: {end_time - start_time:.2f}초).")

        return vectorstore

    except Exception as e:

        print(f"[ERROR] RAG 오류: 벡터 저장소 구축 중 오류 발생 - {e}")

        return None



def query_rag_context(

    vectorstore: FAISS, 

    query: str,

    raw_k: int, 

    final_k: int,

    prefix_len: int

) -> str:

    """

    구축된 vectorstore를 쿼리하고, 중복 제거 및 점수 변환을 거쳐 최종 컨텍스트 문자열을 반환합니다.

    (v1.1 패치에서는 build_rag_context 함수가 이 역할을 대체합니다)

    """

    print(f"[INFO] RAG: '{query}' 쿼리로 유사 뉴스 검색 중 (상위 {raw_k}개)...")

    try:

        retrieved_docs_with_dist = vectorstore.similarity_search_with_score(query, k=raw_k)



        if not retrieved_docs_with_dist:

            return "쿼리와 관련된 뉴스를 찾지 못했습니다."



        print(f"[INFO] RAG: {len(retrieved_docs_with_dist)}개 검색 결과 중복 제거 필터링...")

        seen_titles = set()

        filtered_docs = []

        for doc, dist in retrieved_docs_with_dist:

            title_prefix = doc.page_content[:prefix_len].strip()

            if title_prefix not in seen_titles:

                seen_titles.add(title_prefix)

                filtered_docs.append((doc, dist))

        

        final_docs = filtered_docs[:final_k]

        print(f"[OK] RAG: {len(final_docs)}개 뉴스를 최종 컨텍스트로 선택.")



        news_context = "최근 시장 뉴스 및 분석 요약:\n"

        for doc, dist in final_docs:

            similarity = 1.0 / (1.0 + float(dist)) 

            news_context += (

                f"- (유사도≈{similarity:.2f}) 날짜: {doc.metadata.get('date', 'N/A')}\n"

                f"  내용: {doc.page_content.replace(chr(10), ' ').strip()}\n"

            )

        print("[OK] RAG: 컨텍스트 생성 완료.")

        return news_context.strip()

        

    except Exception as e:

        print(f"[ERROR] RAG 오류: 쿼리 또는 컨텍스트 생성 중 오류 발생 - {e}")

        return f"뉴스 분석 중 오류 발생: {e}"



# =========================

# v1.1 패치: 신규 유틸 함수들

# =========================

def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:

    """Timestamp를 UTC-aware로 강제 변환"""

    if ts.tzinfo is None:

        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")



def _filter_recent_df(df: pd.DataFrame, decision_dt_utc: datetime, recent_days: int) -> pd.DataFrame:

    """

    결정 시각 기준 최근 N일(UTC)만 필터링

    (참고: 원본 btc_news_data.csv의 컬럼명에 맞게 'published date_utc', 'published date'를 찾도록 수정)

    """

    

    # 원본 데이터의 날짜 컬럼명 확인

    if "published date_utc" in df.columns:

        col = "published date_utc"

    elif "published date" in df.columns:

        col = "published date"

    else:

        print("[ERROR] RAG: 뉴스 데이터에 'published date_utc' 또는 'published date' 컬럼이 없습니다.")

        raise ValueError("RAG: News data missing required date column.")



    _df = df.copy()

    _df[col] = pd.to_datetime(_df[col], errors="coerce", utc=False) # 먼저 Naive로 파싱

    _df[col] = _df[col].apply(lambda x: _ensure_utc(x) if pd.notna(x) else pd.NaT)

    _df.dropna(subset=[col], inplace=True) # 날짜 파싱 실패한 행 제거



    start_utc = decision_dt_utc - timedelta(days=recent_days)

    

    # 둘 다 UTC-aware이므로 비교 가능

    mask = (_df[col] >= start_utc) & (_df[col] <= decision_dt_utc)

    return _df.loc[mask].sort_values(col, ascending=False)



def _time_decay_bonus(pub_dt_utc: datetime, decision_dt_utc: datetime, decay: float) -> float:

    """최근일수록 더 높은 보너스 (유사도 점수에 가중)"""

    # pub_dt_utc와 decision_dt_utc가 모두 aware-datetime인지 확인

    pub_dt_utc = _ensure_utc(pd.to_datetime(pub_dt_utc))

    decision_dt_utc = _ensure_utc(pd.to_datetime(decision_dt_utc))



    days = max((decision_dt_utc - pub_dt_utc).days, 0)

    # days=0 (오늘) -> 보너스 1.0

    # 멀어질수록 1.0 미만으로 감소

    return np.exp(-decay * days)



def build_rag_context(

    news_df_all: pd.DataFrame,

    user_query: str,

    decision_dt_utc: datetime,

    embedding_model: HuggingFaceEmbeddings, # [수정] 임베딩 모델을 인자로 받도록 변경

    recent_days: int = RAG_RECENT_DAYS,

    max_docs: int = RAG_MAX_DOCS,

) -> str:

    """

    v1.1 메인 진입점: 신선도 강제 + 시간 감쇠 재랭킹

    """

    print(f"[INFO] RAG (v1.1): {decision_dt_utc.date()} 기준 {recent_days}일 필터링 및 시간 감쇠 적용...")

    try:

        if not isinstance(decision_dt_utc, datetime):

            decision_dt_utc = pd.to_datetime(decision_dt_utc).to_pydatetime()

            

        if decision_dt_utc.tzinfo is None:

             decision_dt_utc = decision_dt_utc.replace(tzinfo=timezone.utc)

        else:

             decision_dt_utc = decision_dt_utc.astimezone(timezone.utc)

        

        # 1) 최신 N일 필터링

        df_recent = _filter_recent_df(news_df_all, decision_dt_utc, recent_days)

        if len(df_recent) == 0:

            print("[WARN] RAG (v1.1): 최근 N일 내 유의미한 뉴스가 없습니다.")

            return "최근 N일 내 유의미한 뉴스가 없습니다."



        print(f"[INFO] RAG (v1.1): {len(df_recent)}개 뉴스로 RAG 구축 시작...")



        # 2) 도큐먼트 생성

        docs: List[Document] = []

        # (참고: 원본 btc_news_data.csv의 컬럼명에 맞게 'published date_utc', 'published date'를 찾도록 수정)

        time_col = "published date_utc" if "published date_utc" in df_recent.columns else "published date"

        

        for _, row in df_recent.iterrows():

            # (참고: 원본 btc_news_data.csv에는 'content' 컬럼이 없으므로 'title', 'description'만 사용)

            content = str(row.get("title", "")) + " " + str(row.get("description", ""))

            pub_dt = pd.to_datetime(row[time_col], errors="coerce")

            if pd.isna(pub_dt): continue

            

            pub_dt = _ensure_utc(pub_dt)

            

            docs.append(

                Document(

                    page_content=content.strip(),

                    metadata={

                        "date": pub_dt.isoformat(),

                        "source": str(row.get("source name", "")), # 'source' -> 'source name'

                        "url": str(row.get("url", "")),

                        "title": str(row.get("title", "")),

                    },

                )

            )

        

        if not docs:

            print("[WARN] RAG (v1.1): Document 변환 후 뉴스가 없습니다.")

            return "최근 N일 내 유의미한 뉴스가 없습니다."



        # 3) 임베딩 인덱스 구성 및 1차 검색

        # (수정: 임베딩 모델 로딩을 매번 하지 않고, 인자로 받은 모델 사용)

        start_time = time.time()

        vs = FAISS.from_documents(docs, embedding_model)

        print(f"[INFO] RAG (v1.1): 벡터 저장소 구축 완료 (소요 시간: {time.time() - start_time:.2f}초).")



        # k값 설정: 검색할 문서가 16개 미만이면 k값을 문서 개수만큼으로 제한

        k_search = min(len(docs), max(16, max_docs * 3))

        

        # [수정] FAISS는 relevance_scores가 아닌 distance_score (낮을수록 좋음) 사용

        retrieved = vs.similarity_search_with_score(user_query, k=k_search)

        # retrieved: List[Tuple[Document, distance]]



        # 4) 시간 감쇠 보너스 적용 후 재정렬

        rescored: List[Tuple[Document, float]] = []

        for doc, dist in retrieved:

            pub_iso = doc.metadata.get("date")

            if not pub_iso: continue

            

            pub_dt = pd.to_datetime(pub_iso, errors="coerce")

            if pd.isna(pub_dt): continue



            pub_dt = _ensure_utc(pub_dt)

            

            # [수정] 시간 감쇠 보너스 (np.exp(-decay * days))는 0~1 사이의 값 (최신=1)

            # 유사도(1 / (1 + dist)) * 보너스

            similarity = 1.0 / (1.0 + float(dist))

            bonus = _time_decay_bonus(pub_dt, decision_dt_utc, RAG_TIME_DECAY)

            

            final_score = similarity * bonus

            rescored.append((doc, final_score))



        rescored.sort(key=lambda x: x[1], reverse=True) # 최종 점수(높을수록 좋음)로 내림차순 정렬

        final = rescored[:max_docs]



        # 5) 컨텍스트 생성

        lines = ["최근 시장 뉴스 및 분석 요약:"]

        for doc, s in final:

            lines.append(f"- (가중유사도≈{s:.2f}) 날짜: {doc.metadata.get('date','N/A')}")

            lines.append(f"  제목: {doc.metadata.get('title','').strip()}")

            # (참고: content가 아닌 page_content 사용)

            lines.append(f"  내용: {doc.page_content.replace(chr(10),' ').strip()}")

        

        print("[OK] RAG (v1.1): 신선도/시간 감쇠 RAG 컨텍스트 생성 완료.")

        return "\n".join(lines).strip()

    

    except Exception as e:

        print(f"[ERROR] RAG (v1.1) 오류: build_rag_context 중 오류 발생 - {e}")

        traceback.print_exc()

        return f"뉴스 분석 중 오류 발생: {e}"