# Create virtual environment
#python -m venv fashion_env
#source fashion_env/bin/activate   # Windows: fashion_env\Scripts\activate

# Install all dependencies
'''
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch
pip install pytrends
pip install pandas numpy scikit-learn matplotlib seaborn umap-learn
pip install Pillow requests tqdm jupyter

# Verify CLIP installation
python -c "import open_clip; print('CLIP ready')"
'''

import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import time

Trending_topics = TrendReq(hl='en-US', tz=360)

print("problemi")
pytrends = TrendReq(hl='en-US', tz=360)

from pytrends.request import TrendReq
import pandas as pd
import time, json

pytrends = TrendReq(hl='en-US', tz=360)
'''
keywords = [
    'quiet luxury fashion', 'mob wife aesthetic', 'ballet flat shoes',
    'wide leg trousers', 'micro bag trend', 'gorpcore style',
    'Y2K fashion', 'coastal grandmother', 'barrel leg jeans', 'sheer fabric trend',
]

# Google Trends API limit: 5 keywords per call. Pull in batches.
all_data = {}
for i in range(0, len(keywords), 5):
    batch = keywords[i:i+5]
    pytrends.build_payload(batch, timeframe='2022-01-01 2024-12-31', geo='US')
    df = pytrends.interest_over_time()
    if not df.empty:
        for kw in batch:
            if kw in df.columns:
                all_data[kw] = df[kw].values
    time.sleep(4)  # stay well under rate limit

df_trends = pd.DataFrame(all_data, index=df.index)
df_trends.to_csv('trends_data.csv')

from pytrends.request import TrendReq
import pandas as pd

# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=360) '''

# Build payload with NO keywords, but specifying category 185 (Apparel)
# 'today 3-m' looks at the last 90 days for fresh/seasonal trends
pytrends.build_payload(kw_list=[''], cat=185, timeframe='today 3-m', geo='US')

# Fetch related topics and queries
related_queries = pytrends.related_queries()

# The empty string payload stores results under the '' key
rising_fashion_trends = related_queries['']['rising']
top_fashion_trends = related_queries['']['top']

print("--- FASTEST RISING FASHION TRENDS ---")
print(rising_fashion_trends.head(15))

# A seed list of incredibly broad fashion pillars
seed_keywords = ['clothing', 'outfit', 'aesthetic style', 'shoes']

all_discovered_trends = []

for seed in seed_keywords:
    # Anchor to category 185 to keep it strictly fashion-related
    pytrends.build_payload(kw_list=[seed], cat=185, timeframe='today 1-m', geo='US')
    related = pytrends.related_queries()
    
    if seed in related and related[seed]['rising'] is not None:
        rising_df = related[seed]['rising']
        all_discovered_trends.append(rising_df)

# Combine and look at your dynamically discovered trend landscape
trend_universe = pd.concat(all_discovered_trends).drop_duplicates(subset=['query'])
print(trend_universe.sort_values(by='value', ascending=False).head(20))