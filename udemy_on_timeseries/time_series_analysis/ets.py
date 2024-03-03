import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# ETS -> Error, Trend, Seasonality

airline = pd.read_csv("../../data/airline_passengers.csv", index_col="Month", parse_dates=True)
# clearn data


# from plot, we can see the trend is non-linear and upward, so we use
# multiplicative model
result = seasonal_decompose(airline['Thousands of Passengers'], model='multiplicative')
# result.plot()
# print(result.trend)
# plt.show()

# simple moving average
roll = airline.rolling(window=6).mean()
roll2 = airline.rolling(window=12).mean()
airline['ma6'] = roll
airline['ma12'] = roll2
plt.plot(airline[['ma6', 'ma12','Thousands of Passengers']])
plt.show()

