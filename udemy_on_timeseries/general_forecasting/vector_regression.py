import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# VAR models
# In SARIMAX, the model is effected by eXogenous variables
# but NOT Vice Versa
# There are cases when the eXogenous variable and model effects
# each other, what do we do in this case? we USE VectorAutoRegression

# Case to : VAR model
# change in personal consumption expenditures C_t were forecast
# based on changes in personal disposable income i_t

# unlike AR(p), VAR model of order p,
# denoted VAR(p), considers each variable y_k in the system.

# forecast into the future using VAR for 2 time series that we believe have effects
# on each other

# we will use M2 Money Stock and personal Consumption from FRED
# M2 Money Stock -> savings deposits, small-denomination time deposits
# - balances in retail money market mutual funds

# according to VAR equation, y1 = Personal Consumption Expenditures
# y2 = M2 money stock

# optimal order (p) for our VAR model
# auto_arima wont do the grid search but we can run various p-val on a loop
# and check which model has the best AIC score

# we will also need to manually check for stationarity and difference the time-series
# if they are not stationary (Augumented Dickey Fuller test)
# we will notice the time series require different differencing "order"
# we will difference them the same amount however, in order to make sure they have
# the same number of rows

# libraries


from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse

import warnings
warnings.filterwarnings('ignore')

# importing data with pandas

df = pd.read_csv("../../data/M2SLMoneyStock.csv", index_col=0, parse_dates=True)
df.index.freq = "MS"

sp = pd.read_csv("../../data/PCEPersonalSpending.csv", index_col=0, parse_dates=True)
sp.index.freq="MS"

# join df and sp together permanently
df = df.join(sp)
df.dropna(inplace=True)
print(df.head())

# df.plot(figsize=(16,10))
# plt.show()

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

adf_test(df['Money'])
adf_test(df['Spending'])

# both non-stationary, so we difference
df_transformed = df.diff() # keep original
adf_test(df_transformed['Money'])
adf_test(df_transformed['Spending'])

# one of them is still non-stationary so
df_transformed = df_transformed.diff().dropna()
print(df_transformed)
adf_test(df_transformed['Money'])
adf_test(df_transformed['Spending'])

# let's build train and test sets
# number of observations
nobs = 12
train = df_transformed[:-nobs] # -nobs -> nobs from the end
test = df_transformed[-nobs:]

# Grid Search for ORDER value 'p' AR of VAR model

model = VAR(train)
for p in [1, 2, 3, 4, 5, 6, 7]:
    results = model.fit(p)
    print(f'ORDER {p}')
    print(f'AIC: {results.aic}')
    print('\n')

# best order is p = 5
results = model.fit(5)
print(results.summary())

# Grab 5 lagged values, right before the test starts
# has to be Numpy Array
lags = train.values[-5:]

ax = results.forecast(y=lags, steps=12)
print(ax)

idx = pd.date_range('2015-01-01', periods=12, freq="MS") # building index for forecast result
df_forecast = pd.DataFrame(data=ax, index=idx, columns=["Money_2d", "Spending_2d"])

# now we have money and spending 2 difference
# we have to revert it to the forecast

df_forecast['Money1d'] = (df['Money'].iloc[-nobs-1] - df['Money'].iloc[-nobs-2]) + df_forecast['Money_2d'].cumsum()
df_forecast['MoneyForecast'] = df['Money'].iloc[-nobs-1] + df_forecast['Money1d'].cumsum()

df_forecast['Spending1d'] = (df['Spending'].iloc[-nobs-1] - df['Spending'].iloc[-nobs-2]) + df_forecast['Spending_2d'].cumsum()
df_forecast['SpendingForecast'] = df['Spending'].iloc[-nobs-1] + df_forecast['Spending1d'].cumsum()

test_range = df[-nobs:]

test_range['Money'].plot(legend=True, figsize=(12, 8))
df_forecast['MoneyForecast'].plot(legend=True)
plt.show()
