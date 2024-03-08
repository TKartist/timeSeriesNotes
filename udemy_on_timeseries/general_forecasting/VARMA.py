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


# VARMA is just expanding ARMA with VAR
from statsmodels.tsa.api import VAR, VARMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse
from pmdarima import auto_arima

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
# money_order = auto_arima(df['Money'], maxiter=1000) # overriding maximum iteration limit for maximum likelihood estimator
# spending_order = auto_arima(df['Spending'], maxiter=1000)
# print(money_order.summary()) (1, 2, 2)
# print(spending_order.summary()) (1, 1, 2)

# from summary it looks like order for VARMA is (1, 2) taking the
# d-value from the SARIMAX models

model = VARMAX(train, order=(1, 2), trend='c') # c for constant linear fit
results = model.fit(maxiter=1000, disp=False)
print(results.summary())

# now we just forecast

df_forecast = results.forecast(12)

# Now we invert the transformation back

df_forecast['Money1d'] = (df['Money'].iloc[-nobs-1] - df['Money'].iloc[-nobs-2]) + df_forecast['Money'].cumsum()
df_forecast['MoneyForecast'] = df['Money'].iloc[-nobs-1] + df_forecast['Money1d'].cumsum()

df_forecast['Spending1d'] = (df['Spending'].iloc[-nobs-1] - df['Spending'].iloc[-nobs-2]) + df_forecast['Spending'].cumsum()
df_forecast['SpendingForecast'] = df['Spending'].iloc[-nobs-1] + df_forecast['Spending'].cumsum()

test_clean = df[-nobs:]
df_forecast['MoneyForecast'].plot()
test_clean.plot()
plt.show()

# Surprisingly, VARMA performs shit compared to VAR Model