import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing Holt-Winters Model (Simple Exponential Smoothing)
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv("../../data/airline_passengers.csv", index_col="Month", parse_dates=True)
df.dropna(inplace=True)

# we need to set frequency of the df.index

df.index.freq = 'MS' # sets frequency as Month Start frequency

# Please check offset aliases for pandas to see other possible frequency string values

print(df.index)

span = 12
alpha = 2 / (span + 1) # eq for alpha using span

df['EWMA12'] = df['Thousands of Passengers'].ewm(alpha=alpha, adjust=False).mean()

# same thing but with simple exponential smoothing
model = SimpleExpSmoothing(df['Thousands of Passengers'])
fitted_model = model.fit(smoothing_level=alpha, optimized=False)
df['SES12'] = fitted_model.fittedvalues.shift(-1) # because all the data gets shifted when we are doing fitting

print(df.head())

# Double Exponential Smoothing Model

df['DES_add_12'] = ExponentialSmoothing(df['Thousands of Passengers'],trend='add').fit().fittedvalues.shift(-1) # change add to mul if want to use multiplicative model
print(df.head())
# df.plot(figsize=(12, 5))
# df.iloc[:24].plot(figsize=(12, 5)) # the curves are too close to each other when ploting all data, so we zoomed into see the difference
# plt.show()
# as the data gets long, it is possible additive and multiplicative model over-exaggerates, so we need some dampening


# Triple Exponential Smoothing Model
df['TES_mul_12'] = ExponentialSmoothing(df['Thousands of Passengers'], trend='mul', seasonal='mul', seasonal_periods=12).fit().fittedvalues
# deciding if seasonality is additive or multiplicative is much harder, so it is better to try different
# variations and see which one performs better

df[['Thousands of Passengers', 'SES12', 'DES_add_12',
       'TES_mul_12']].iloc[-24:].plot(figsize=(12, 5))
plt.show()

# if we plot first 2 years, clearly "simpler" double exponential smoothing model works better
# however, as we go into the future, the triple exponential model starts to improve
# as we are trying to do forecasting, this property of triple (more complicated) is more beneficial

# try the Statsmodels Time Series Exercises when have time (lecture 47)