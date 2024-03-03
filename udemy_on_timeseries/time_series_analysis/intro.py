import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# library for time-series analysis (TSA)
from statsmodels.tsa.filters.hp_filter import hpfilter
# hp_filter -> Hodrick Prescott filter (refer to notes for further detail)
# learning stats model library

df = pd.read_csv("../../data/macrodata.csv", index_col=0, parse_dates=True)
print(df.head())
# plt.plot(df["realgdp"])
# plt.show()

gdp_cycle, gdp_trend = hpfilter(df["realgdp"], lamb=1600) # as the data is quarterly
# in case of annual -> x = 6.25, in case of monthly -> x = 129,600

# we append gdp_trend to our df
df['trend'] = gdp_trend
plt.plot(df[['trend', 'realgdp']]['2005-01-01':])
plt.show()
