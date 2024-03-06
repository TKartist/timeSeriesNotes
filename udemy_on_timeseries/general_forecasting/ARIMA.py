import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Case 1: ARIMA(p != 0, d = 0, q = 0) -> Basically just AutoRegression

# forecasting using a linear combination of past values of the variable.
# autoregression = regression of the variable against itself.
# An autoregression is run against a set of lagged values of order 'p'.

# autoregressive model specifies that the output variable depends linearly on
# its own previous values and on a stochastic term (imperfectly predictable term)

# Formula for AR (AutoRegression)
# y(t) = c + phi_1 * y_(t - 1) + phi_2 * y_(t - 2) + ... + phi_p * y_(t - p) + eps_t
# where phi = lag coefficient
# eps_t = white noise

# AR(1) = y(t) = c + phi_1 * y_(t - 1) + eps_t
# predicting y(t) just from y_(t - 1)

# Don't go too far back as it will pick up unnecessary noise
# Also very mathematically complex

# Implementation

from statsmodels.tsa.ar_model import AR, ARResults

df = pd.read_csv("../../data/uspopulation.csv", index_col="DATE", parse_dates=True)
df.index.freq = "MS"
# df.plot()
# plt.show()

# assuming we want to predict the population growth in the next year
train_size = len(df) - 12
test_size = 12

train_set = df[:train_size]
test_set = df[train_size:]
print(len(train_set))

import warnings
warnings.filterwarnings('ignore')

model = AR(train_set['PopEst'])

# AR(1) model
AR1fit = model.fit(maxlag = 1) # many options so use grid search for best options

maxlag_val = AR1fit.k_ar
params = AR1fit.params

start = len(train_set)
end = len(train_set) + len(test_set) - 1

# how to predict using fitted AR model
prediction1 = AR1fit.predict(start=start, end=end)
prediction1 = prediction1.rename('AR(1) pred')
print(prediction1)

model2 = AR(train_set['PopEst'])


AR2fit = model2.fit(maxlag=2)
prediction2 = AR2fit.predict(start=start, end=end)
prediction2 = prediction2.rename("AR(2) pred")




# model.fit(ic=) ic -> criterion used for selecting the optimal lag length
# there are many criterions but we are going to use t-stat here
# because we use RMSE to decide which performs better

model3 = AR(train_set['PopEst'])

ARfit = model3.fit(ic='t-stat') # not specifying the maxlag as 
# ic will find optimal lag length

print(ARfit.params)

prediction8 = ARfit.predict(start, end)
prediction8 = prediction8.rename('AR(8) pred')

from sklearn.metrics import mean_squared_error

labels = ['AR1', "AR2", "AR8"]
preds = [prediction1, prediction2, prediction8]

for i in range(3):
    error = mean_squared_error(test_set['PopEst'], preds[i])
    print(f'{labels[i]} MSE was : {error}')

# df[84:].plot(figsize=(12, 8), legend=True)
# prediction1.plot(legend=True)
# prediction2.plot(legend=True)
# prediction8.plot(legend=True)

# plt.show()

# Forecasting the FUTURE!!!
model4 = AR(df['PopEst'])
ARfit = model4.fit() # let statsmodel figure out the lag size
forecasted_values = ARfit.predict(start=len(df), end=len(df)+12).rename('Forecast')
# 12 months as our test set was 12 months

df['PopEst'].plot(figsize=(12,8), legend=True)
forecasted_values.plot(legend=True)
plt.show()
