import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Stationarity test (augumented Dickey-Fuller Test)
# classic null hypothesis test
# assume phi = 1 -> if p < 0.05, we reject
# null hypothesis and the data is stationary
# otherwise it is non-stationary

# Granger Causality Test
# determine if one time series is useful in forecasting causality
# correlation is easy to view but causality is hard (correlation through out the time)

# AIC and BIC

# AIC (Akaike Information Criterion)
# Evaluates a collection of models and estimates the quality of
# each model relative to the others
# penalties are provided for the number of parameters used in an effort
# to thwart overfitting (simple models are rewarded more than complex models)
# tries to find most cost efficient model

# Overfitting results in performing very well on training data,
# but poorly on new unseen data

# BIC (Bayesian Information Criterion)
# similar to AIC but approaches the model comparison with a Bayesian approach

import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
# adfuller -> augumented dickey fuller test
# grangercausalitytest -> to check causality between data

df1 = pd.read_csv("../../data/airline_passengers.csv", index_col="Month", parse_dates=True)
df1.index.freq = "MS"

df2 = pd.read_csv("../../data/DailyTotalFemaleBirths.csv", index_col="Date", parse_dates=True)
df2.index.freq = "D"

test_result = adfuller(df1['Thousands of Passengers']) # runs AIC
# print(help(adfuller)) -> to get full info of function
# print(test_result)
dfout = pd.Series(test_result[0:4], index=["ADF test statistics", "p-val", "# lag used", "# observations"])

for key, val in test_result[4].items():
    dfout[f'critical value ({key})'] = val

# print(dfout)

# function for adf_test

####################################################################################
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
#################################################################################

adf_test(df1["Thousands of Passengers"], "Airline Passengers")
print("\n")
adf_test(df2["Births"], "Female Births")

df3 = pd.read_csv("../../data/samples.csv", index_col=0, parse_dates=True)
df3.index.freq = "MS"

# df3['a'].iloc[2:].plot(figsize=(12,8), legend=True)
# df3['d'].shift(2).plot(legend=True)
# plt.show()
# you can clearly see the causality now, so how do we check it?

cause = grangercausalitytests(df3[['a', 'd']], maxlag=3) # so max comparison is x against x - 3
# higher maxlag -> more likely to find the causality BUT higher computation time
# what we are looking for is basically where the p-val is very low
# p-val was 0 for lag 2 and 3

# now lets see an example where there isn't causality and see what the p-val is
cause2 = grangercausalitytests(df3[['b', 'd']], maxlag=3)
# p-val was all above few thousands, meaning no causality