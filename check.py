import pandas as pd



df = pd.read_csv("eth_news_data.csv")



print(df.columns.tolist())

print("rows:", len(df))



if "sentiment_score" in df.columns:

    print("sentiment_score non-null:", df["sentiment_score"].notna().sum())

    print(df["sentiment_score"].head(10))

else:

    print("sentiment_score column 없음")



if "full_text" in df.columns:

    print("full_text non-null:", df["full_text"].notna().sum())

    print(df["full_text"].head(3))

else:

    print("full_text column 없음")