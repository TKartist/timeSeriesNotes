import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# label :- column we are trying to predict

# example of SARIMAX: X for eXogenous variable:
# for example, we try to predict number of customers at a resturant
# we have historical data, we can use SARIMA based model to use
# historical lagged values to predict future visit numbers

# BUT what if we had some other features we wanted to include, like holidays?

# when setting eXogenous variable, we need to have intuition on if the data
# is correlated. to check the correlation, we can run a correlation test.

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../../data/RestaurantVisitors.csv", index_col="date", parse_dates=True)
df.index.freq = "D"

df1 = df.dropna()

# converting data type
cols = ['rest1','rest2','rest3','rest4','total']
for col in cols:
    df1[col] = df1[col].astype(int)

# ax = df1['total'].plot(figsize=(16,9))
# # plt.show()
    


# now we get seasonal decomposition of our data
from statsmodels.tsa.seasonal import seasonal_decompose

# results = seasonal_decompose(df1['total'])
# results.seasonal.plot()
# plt.show()

# from it, it seems like the data has a weekly seasonal component

train = df1.iloc[:436]
test = df1.iloc[436:]

# from pmdarima import auto_arima

# z = auto_arima(df1['total'], seasonal=True, m=7).summary()

# print(z)

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train['total'], order=(1,0,0), seasonal_order=(2,0,0,7),
                enforce_invertibility=False)

# invertibility : basically the calculation of AR is linear func of past and
# current observation, and there is a value theta (if < 1) which tries to 
# assign smaller weight to older data. However, if theta >= 1, it gives
# constant weight or more weight to past data. Not so preferable
# hence, the stats library is encoded in a way which forces theta to be
# less than 1.
# which doesn't make sense and only skews data sometimes if the data's
# theta is already less than 1. Hence, we disable the invertibility here.

results = model.fit()

start = len(train)
end = len(train) + len(test) - 1
predictions = results.predict(start, end).rename('SARIMA Model')

ax = test['total'].plot(legend=True, figsize=(16,9))

holidays = test.query('holiday==1').index

for day in holidays:
    ax.axvline(x=day, color='black', alpha=0.8)

predictions.plot(legend=True)
# plt.show()

from statsmodels.tools.eval_measures import rmse
k = rmse(test['total'], predictions)
print(k)
print(test['total'].mean())

# How to add exogenous variable into our SARIMA model