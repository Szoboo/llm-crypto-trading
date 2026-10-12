from gnews import GNews

import pandas as pd

import time

import random

import os

import ast

import re



# 본문 추출용

try:

    from newspaper import Article

    NEWSPAPER_AVAILABLE = True

except Exception:

    NEWSPAPER_AVAILABLE = False



# -----------------------------

# 1. 설정

# -----------------------------

KEYWORDS = '"Ethereum" OR "ETH price" OR "ETH crypto" OR "Vitalik Buterin"'

FILENAME = "eth_news_data.csv"



BASE_START_DATE = pd.Timestamp("2022-09-15", tz="UTC")

END_DATE = pd.Timestamp("2026-02-15", tz="UTC")



MAX_CONTENT_CHARS = 1500



COLUMNS = [

    'title',

    'description',

    'content',

    'published date',

    'url',

    'publisher',

    'real_url',

    'news_count',

    'candle_4h'

]



print(f"목표 수집 기간: {BASE_START_DATE.date()} ~ {END_DATE.date()}")



# -----------------------------

# 2. 안전한 파싱

# -----------------------------

def safe_eval(x):

    try:

        if isinstance(x, dict):

            return x

        if isinstance(x, str):

            x = x.strip().strip('"').strip("'")

            if x.startswith("{"):

                return ast.literal_eval(x)

        return {}

    except Exception:

        return {}



def extract_publisher(x):

    return safe_eval(x).get('title', '')



def extract_real_url(x, fallback):

    return safe_eval(x).get('href', fallback)



# -----------------------------

# 3. title 정제 함수 (중복 제거용)

# -----------------------------

def clean_title(text):

    text = str(text).lower()

    text = re.sub(r'[^a-z0-9 ]', '', text)

    return text.strip()



# -----------------------------

# 4. 본문 추출 함수

# -----------------------------

def fetch_article_content(url: str) -> str:

    if not isinstance(url, str) or not url.strip():

        return ""



    if not NEWSPAPER_AVAILABLE:

        return ""



    try:

        article = Article(url)

        article.download()

        article.parse()

        text = article.text or ""

        text = re.sub(r'\s+', ' ', text).strip()

        return text[:MAX_CONTENT_CHARS]

    except Exception:

        return ""



# -----------------------------

# 5. 후처리

# -----------------------------

def postprocess_news(df):

    df['published date'] = pd.to_datetime(

        df['published date'], utc=True, errors='coerce'

    )

    df = df.dropna(subset=['published date', 'url'])



    if 'publisher' in df.columns:

        df['real_url'] = df.apply(

            lambda row: extract_real_url(row['publisher'], row['url']),

            axis=1

        )

        df['publisher'] = df['publisher'].apply(extract_publisher)

    else:

        df['real_url'] = df['url']

        df['publisher'] = ''



    df['title'] = df['title'].fillna('').astype(str)

    df['description'] = df['description'].fillna('').astype(str)



    # 짧은 뉴스 제거 완화

    df = df[df['title'].str.len() > 10]



    # 노이즈 제거

    noise_keywords = ['zurich', 'university', 'ethics']

    pattern = '|'.join(noise_keywords)

    df = df[~df['title'].str.lower().str.contains(pattern, na=False)]



    # 본문 추출

    print("  -> 기사 본문 추출 중...")

    df['content'] = df['real_url'].apply(fetch_article_content)

    df['content'] = (

        df['content']

        .fillna('')

        .astype(str)

        .str.replace(r'\s+', ' ', regex=True)

        .str.strip()

    )



    # feature 생성

    df['news_count'] = 1

    df['candle_4h'] = (

        df['published date'].dt.floor('4h')

        + pd.Timedelta(hours=4)

    )



    return df



# -----------------------------

# 6. 기존 데이터 로드

# -----------------------------

if os.path.exists(FILENAME):

    try:

        existing_df = pd.read_csv(FILENAME)

        print(f"기존 데이터: {len(existing_df)}건")



        if not existing_df.empty:

            existing_df['published date'] = pd.to_datetime(

                existing_df['published date'], utc=True, errors='coerce'

            )

            START_DATE = existing_df['published date'].max().floor('D') + pd.Timedelta(days=1)

        else:

            START_DATE = BASE_START_DATE



    except Exception as e:

        print(f"CSV 읽기 에러 → 새로 시작: {e}")

        existing_df = pd.DataFrame(columns=COLUMNS)

        START_DATE = BASE_START_DATE

else:

    existing_df = pd.DataFrame(columns=COLUMNS)

    START_DATE = BASE_START_DATE

    print("기존 파일 없음. 새로 시작")



print(f"수집 시작일: {START_DATE.date()}")



# 파일 초기화

if not os.path.exists(FILENAME):

    pd.DataFrame(columns=COLUMNS).to_csv(FILENAME, index=False)



# -----------------------------

# 7. 뉴스 수집

# -----------------------------

date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')



for single_date in date_range:

    start_tuple = (single_date.year, single_date.month, single_date.day)

    next_day = single_date + pd.Timedelta(days=1)

    end_tuple = (next_day.year, next_day.month, next_day.day)



    print(f"[{single_date.date()}]", end=" ")



    try:

        google_news = GNews(

            language='en',

            start_date=start_tuple,

            end_date=end_tuple,

            max_results=50

        )



        articles = google_news.get_news(KEYWORDS)



        if articles:

            df = pd.DataFrame(articles)

            df = postprocess_news(df)



            if not df.empty:

                df = df[COLUMNS]



                df.to_csv(

                    FILENAME,

                    mode='a',

                    header=False,

                    index=False,

                    encoding='utf-8-sig'

                )

                print(len(df))

            else:

                print("filtered 0")

        else:

            print("0")



        time.sleep(random.uniform(1, 2))



    except Exception as e:

        print(f"ERROR: {e}")

        time.sleep(5)



print("\n크롤링 수집 완료")



# -----------------------------

# 8. 중복 제거

# -----------------------------

print("중복 제거 시작...")

df = pd.read_csv(FILENAME)



# 잘못된 header 제거

df = df[df['url'] != 'url']



# clean_title 생성

df['title_clean'] = df['title'].apply(clean_title)



# 중복 제거

df.drop_duplicates(subset=['url'], inplace=True)

df.drop_duplicates(subset=['title_clean'], inplace=True)



# 정렬

df['published date'] = pd.to_datetime(df['published date'], utc=True, errors='coerce')

df.sort_values(by=['published date'], inplace=True)



# 저장 시 title_clean 제거

df.drop(columns=['title_clean'], inplace=True)



df.to_csv(FILENAME, index=False, encoding='utf-8-sig')



print(f"최종 데이터 수: {len(df)}")

print("뉴스 수집 완료:", FILENAME)