# 1_merge_news.py
# pip install pandas glob2
import pandas as pd, glob

files = glob.glob("data/news_raw/*.csv")  # adjust to your folder
assert files, "No per-ticker CSVs found."

news = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
# basic cleaning
news.dropna(subset=["ticker","published_utc","title"], inplace=True)
news.drop_duplicates(subset=["ticker","published_utc","title"], inplace=True)
news.sort_values(["ticker","published_utc"], ascending=[True, True], inplace=True)
news.to_csv("news_raw.csv", index=False)
print("Merged → news_raw.csv", len(news))
