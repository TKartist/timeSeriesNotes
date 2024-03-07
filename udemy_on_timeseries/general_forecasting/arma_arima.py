import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# forecasting library
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.tools import diff
from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

df1 = pd.read_csv("../../data/DailyTotalFemaleBirths.csv", index_col="Date", parse_dates=True)
df1.index.freq = "D"
df1 = df1[:120]
df2 = pd.read_csv("../../data/TradeInventories.csv", index_col="Date", parse_dates=True)
df2.index.freq = "MS"

# ARMA
# df1['Births'].plot(figsize=(12, 5))
# plt.show()

# somewhat stationary data
# ADF-test


def adf_test(series, title=''):
    print(f'Augumented Dickey Fuller test: {title}\n')

    result = adfuller(series.dropna(), autolag='AIC')
    labels = ["ADF test statistics", "p-val", "# lag used", "# observations"]

    out = pd.Series(result[0:4], index=labels)
    for key, val in result[4].items():
        out[f'critical value ({key})'] = val
    
    print(out.to_string())

    if result[1] < 0.05:
        print("\nStrong evidence against Null Hypothesis")
        print("Rejecting the Null Hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("\nWeak evidence against Null Hypothesis")
        print("Fail to reject the Null Hypothesis")
        print("Data has a unit root and is non-stationary")

result = seasonal_decompose(df2['Inventories'], model='add')
# result.plot()
# plt.show()

# plot_acf(df2['Inventories'], lags=40)
# plot_pacf(df2['Inventories'], lags=40)
# plt.show()

# choosing d-val -> keep running diff() func until data is stationary

stepwise_fit = auto_arima(df2['Inventories'], start_p=0, start_q=0, max_p=2, max_q=2, seasonal=False)

train = df2.iloc[:252]
test = df2.iloc[252:]
model = ARIMA(train['Inventories'], order=(1,1,1))
results = model.fit()

start = len(train)
end = len(train) + len(test)
predictions = results.predict(start=start, end=end, typ='levels').rename('ARIMA(1,1,1) Preds')

test['Inventories'].plot(legend=True, figsize=(12, 8))
predictions.plot(legend=True)
plt.show()