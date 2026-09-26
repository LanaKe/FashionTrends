from pytrends.request import TrendReq
import pandas as pd
import time

pytrends = TrendReq(hl='en-US', tz=360)

keywords = [
    'creamy yellow', 'hot pants', 'moccasins', 'bubble hem', 
    'wide leg trousers', 'Y2K fashion', 'thong sandals', 'strapless dress', 'barrel leg jeans',
    'ballet flat shoes'
]

# Pull 5 at a time (Google Trends limit per call)
all_data = {}
for i in range(0, len(keywords), 5):
    batch = keywords[i:i+5]
    pytrends.build_payload(batch, timeframe='2024-01-01 2026-06-30', geo='')
    data = pytrends.interest_over_time()
    if 'isPartial' in data.columns:
        data = data.drop(columns=['isPartial'])
    if not data.empty:
        for kw in batch:
            if kw in data.columns:
                all_data[kw] = data[kw]
    time.sleep(30)  # be polite to the API

df_trends = pd.DataFrame(all_data)
df_trends.to_csv('trends_data2.csv')
print(df_trends.describe())

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, kw in zip(axes.flatten(), df_trends.columns):
    ax.plot(df_trends.index, df_trends[kw], color='#6C3FC5', linewidth=1.5)
    ax.set_title(kw, fontsize=8)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig('trends_overview.png', dpi=150)
plt.show()

