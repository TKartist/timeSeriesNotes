import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

df = pd.read_csv("../../data/co2_mm_mlo.csv")
df['date'] = pd.to_datetime({'year':df['year'], 'month':df['month'], 'day':1})

df = df.set_index('date')
df.index.freq='MS'

# print(df.head())

# we use interpolated column instead of average as it is cleaned ver.
# result = seasonal_decompose(df['interpolated'], model='add')
# result.plot()
# plt.show()

# print(auto_arima(df['interpolated'], seasonal=True, m=12).summary())
train = df.iloc[:717]
test = df.iloc[717:]
# model = SARIMAX(train['interpolated'],order=(0, 1, 1), seasonal_order=(1, 0, 1, 12))
# results = model.fit()
# # print(results.summary())
# start = len(train)
# end = len(train) + len(test) - 1
# predictions = results.predict(start, end, typ='levels').rename('SARIMA pred')
# test['interpolated'].plot(legend=True, figsize=(12, 8))
# predictions.plot(legend=True)
# plt.show()

start = len(df)
end = len(df) + 12
model = SARIMAX(df['interpolated'],order=(0, 1, 1), seasonal_order=(1, 0, 1, 12))
results = model.fit()
predictions = results.predict(start, end, typ='levels').rename('SARIMA pred')
df["interpolated"].plot(legend=True, figsize=(12, 8))
predictions.plot(legend=True)
plt.show()


