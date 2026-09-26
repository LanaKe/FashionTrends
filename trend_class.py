import pandas as pd
import numpy as np

# 1. Load data and set date index
df = pd.read_csv('trends_data2.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Identify the keyword columns
keywords = df.columns.tolist()

# Prepare a dictionary to hold our classifications
annotations = {}

# 2. Resample to monthly data for the PERSISTENT condition
df_monthly = df.resample('ME').mean()

for kw in keywords:
    series = df[kw]
    series_monthly = df_monthly[kw]
    
    # Calculate base statistical metrics
    mean_vol = series.mean()
    peak_vol = series.max()
    
    # --- CONDITION 1: FADED ---
    # mean < 15 across window
    if mean_vol < 15:
        annotations[kw] = 'FADED'
        continue
        
    # --- CONDITION 2: PERSISTENT ---
    # mean volume >= 30, present in >= 8 of 12 months above 20
    # (Checking the trailing 12 months of data)
    last_12_months = series_monthly.tail(30)
    months_above_15 = (last_12_months > 15).sum()
    
    if mean_vol >= 20 and months_above_15 >= 24:
        annotations[kw] = 'PERSISTENT'
        continue
        
    # --- CONDITION 3: SPIKE_FADE ---
    # peak >= 60, drops below 20 within 12 weeks of that peak
    has_spike_fade = False
    if peak_vol >= 20:
        # Find where the maximum peak occurred
        peak_idx = series.idxmax()
        # Look at the 12 weeks immediately following the peak
        post_peak_12w = series.loc[peak_idx:].head(13) # includes peak week + 12 weeks
        if (post_peak_12w < 20).any():
            has_spike_fade = True
            
    if has_spike_fade:
        annotations[kw] = 'SPIKE_FADE'
        continue
        
    # --- CONDITION 4: EMERGING ---
    # monotonically increasing over final 6 months (~26 weeks)
    # Note: Strictly monotonic means diff() > 0; non-decreasing means diff() >= 0
    final_6_months = series.tail(26)
    if final_6_months.diff().dropna().ge(0).all():
        annotations[kw] = 'EMERGING'
        continue
        
    # --- DEFAULT CASE ---
    annotations[kw] = 'UNCLASSIFIED / STABLE'

# 3. Output results
results_df = pd.DataFrame.from_dict(annotations, orient='index', columns=['Classification'])
results_df.index.name = 'Keyword'

print("=== Keyword Trend Classification ===")
print(results_df)

# Optional: Save annotations to a new CSV
results_df.to_csv('classified_trends.csv')