import os



os.environ['HF_HOME'] = './hf_cache_dir'



import ccxt

import pandas as pd

import numpy as np

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from torch.nn.functional import softmax

from tqdm import tqdm

import random

import time

import warnings

import logging



# -----------------------------

# 설정

# -----------------------------

PRICE_START_DATE = '2022-01-01T00:00:00Z'

SYMBOL = 'ETH/USDT'

TIMEFRAME = '4h'



NEWS_CSV_FILE = "eth_news_data.csv"

OUTPUT_CSV_FILE = "eth_master_data.csv"



# 아직은 기존 금융 모델로 먼저 테스트

SENTIMENT_MODEL_NAME = "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis"

MAX_CONTENT_CHARS = 1500



SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)



warnings.filterwarnings("ignore")

logging.getLogger("ccxt").setLevel(logging.ERROR)



# -----------------------------

# 1. 가격 데이터 수집

# -----------------------------

def fetch_price_data():

    exchange = ccxt.binance()

    since = exchange.parse8601(PRICE_START_DATE)

    all_ohlcv = []



    while True:

        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since)

        if not ohlcv:

            break



        all_ohlcv.extend(ohlcv)

        last_timestamp = ohlcv[-1][0]

        since = last_timestamp + (4 * 60 * 60 * 1000)

        time.sleep(exchange.rateLimit / 1000)



    df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    df.set_index('timestamp', inplace=True)

    df.index = df.index.tz_localize('UTC')

    df.sort_index(inplace=True)



    return df



# -----------------------------

# 2. 기술적 지표

# -----------------------------

def calculate_technical_indicators(df):

    df['SMA_12'] = df['close'].rolling(12).mean()

    df['SMA_48'] = df['close'].rolling(48).mean()



    delta = df['close'].diff()

    gain = delta.clip(lower=0).rolling(14).mean()

    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-6)

    df['RSI_14'] = 100 - (100 / (1 + gain / loss))



    exp1 = df['close'].ewm(span=12, adjust=False).mean()

    exp2 = df['close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = exp1 - exp2

    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()



    ranges = pd.concat([

        df['high'] - df['low'],

        abs(df['high'] - df['close'].shift()),

        abs(df['low'] - df['close'].shift())

    ], axis=1)



    df['ATR_14'] = ranges.max(axis=1).rolling(14).mean()



    df['Resistance_1'] = df['high'].rolling(60).max().shift(1)

    df['Support_1'] = df['low'].rolling(60).min().shift(1)



    df['Target_Up'] = np.where(df['close'].shift(-1) > df['close'], 1.0, 0.0)

    df.loc[df.index[-1], 'Target_Up'] = np.nan



    return df



# -----------------------------

# 3. full_text 생성

# -----------------------------

def build_full_text(df_news: pd.DataFrame) -> pd.DataFrame:

    df_news = df_news.copy()



    df_news['title'] = df_news.get('title', '').fillna('').astype(str)

    df_news['description'] = df_news.get('description', '').fillna('').astype(str)



    if 'content' not in df_news.columns:

        df_news['content'] = ''



    df_news['content'] = (

        df_news['content']

        .fillna('')

        .astype(str)

        .str.replace(r'\s+', ' ', regex=True)

        .str.strip()

        .str[:MAX_CONTENT_CHARS]

    )



    df_news['full_text'] = (

        df_news['title'] + '. ' +

        df_news['description'] + '. ' +

        df_news['content']

    ).str.replace(r'\s+', ' ', regex=True).str.strip()



    return df_news



# -----------------------------

# 4. 감성 분석 (title-only -> full_text)

# -----------------------------

def process_news_sentiment(news_csv):

    if not os.path.exists(news_csv):

        print("뉴스 없음")

        return None



    df_news = pd.read_csv(news_csv)

    df_news = build_full_text(df_news)



    if 'sentiment_score' not in df_news.columns:

        df_news['sentiment_score'] = pd.NA



    missing = df_news['sentiment_score'].isna()



    if missing.sum() > 0:

        print(f"   → {missing.sum()}개 분석 (DeBERTa / full_text)")



        try:

            tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)

            model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)



            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model.to(device)

            model.eval()



            def calc_sentiment_score(text):

                if not isinstance(text, str) or not text.strip():

                    return 0.0



                inputs = tokenizer(

                    text,

                    return_tensors="pt",

                    truncation=True,

                    max_length=512

                ).to(device)



                with torch.no_grad():

                    outputs = model(**inputs)



                probs = softmax(outputs.logits, dim=1).cpu().numpy()[0]



                # [negative, neutral, positive] 가정

                return float(probs[2] - probs[0])



            tqdm.pandas(desc="   DeBERTa(full_text)")

            df_news.loc[missing, 'sentiment_score'] = (

                df_news.loc[missing, 'full_text'].progress_apply(calc_sentiment_score)

            )



            df_news.to_csv(news_csv, index=False)



        except Exception as e:

            print("DeBERTa 실패")

            print(e)

            return None



    return df_news



# -----------------------------

# 5. 뉴스 feature 생성

# -----------------------------

def process_news_features(df_news):

    if df_news is None or df_news.empty:

        return None



    df_news['published date'] = pd.to_datetime(df_news['published date'], utc=True)



    df_news['candle_4h'] = (

        df_news['published date'].dt.floor('4h')

        + pd.Timedelta(hours=4)

    )



    sentiment_agg = df_news.groupby('candle_4h').agg(

        sentiment_mean=('sentiment_score', 'mean'),

        sentiment_count=('url', 'count')

    )



    sentiment_agg['sentiment_weighted'] = (

        sentiment_agg['sentiment_mean'] *

        np.log1p(sentiment_agg['sentiment_count'])

    )



    sentiment_agg['sentiment_weighted'] = sentiment_agg['sentiment_weighted'].clip(-1, 1)



    return sentiment_agg



# -----------------------------

# 6. 병합

# -----------------------------

def merge_all(df_price, sentiment_agg):

    df_master = df_price.join(sentiment_agg, how='left')



    for col in ['sentiment_mean','sentiment_count','sentiment_weighted']:

        if col in df_master.columns:

            df_master[col] = df_master[col].fillna(0)



    df_master['has_news'] = (df_master['sentiment_count'] > 0).astype(int)



    df_master['sentiment_mean_lag1'] = df_master['sentiment_mean'].shift(1)

    df_master['sentiment_count_lag1'] = df_master['sentiment_count'].shift(1)

    df_master['sentiment_weighted_lag1'] = df_master['sentiment_weighted'].shift(1)



    return df_master



# -----------------------------

# 실행

# -----------------------------

if __name__ == "__main__":

    df_price = fetch_price_data()

    df_price = calculate_technical_indicators(df_price)



    df_news = process_news_sentiment(NEWS_CSV_FILE)

    sentiment_agg = process_news_features(df_news)



    df_master = merge_all(df_price, sentiment_agg)



    df_master.dropna(inplace=True)

    df_master.to_csv(OUTPUT_CSV_FILE)



    print("[OK] eth_master_data.csv 저장 완료")