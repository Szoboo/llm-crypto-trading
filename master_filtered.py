import pandas as pd



df = pd.read_csv("eth_master_data.csv")



# ✅ timestamp UTC 기준으로 변환

df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')



# ✅ 기준 날짜도 UTC로 맞춤

start = pd.Timestamp("2022-09-15", tz="UTC")

end = pd.Timestamp("2026-02-15", tz="UTC")



# ✅ 필터링

df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]



# NaT 제거 (안전)

df = df.dropna(subset=['timestamp'])



# 저장

df.to_csv("eth_master_data_filtered.csv", index=False)



print(len(df))