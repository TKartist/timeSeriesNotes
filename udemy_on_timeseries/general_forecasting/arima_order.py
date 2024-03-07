import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# How to decide order = (p, d, q) values of ARIMA model

# if AC plot shows +ve autocorrelation at the first lag (lag-1)
# then it suggest to use the AR terms in relation to the lag 
# if -ve at first lag, then it suggests using MA terms
# This will allow you to decide what actual values of p,d, and q to provide
# ur ARIMA model

# p - max lag
# d - number of differencing
# q - size of the moving average window

# id of an AR model is often best done with PACF
# id of an MA model is often best done with ACF

# pmdarima (Pyramid ARIMA) is a separate library designed to perform grid
# searches across multiple combinations of p, d, q, and P, D, Q
# MOST EFFECTIVE

# PMD uses AIC -> Akaike Information Criterion (2k - 2 ln (L))
# where 'k' is no. of estimated parameters in the model
# and 'L' is the maximum value of the likelihood function for the model

df1 = pd.read_csv("../../data/airline_passengers.csv", index_col="Month", parse_dates=True)
df1.index.freq="MS"
print(df1.head())

df2 = pd.read_csv("../../data/DailyTotalFemaleBirths.csv", index_col="Date", parse_dates=True)
df2.index.freq="D"

from pmdarima import auto_arima

import warnings
warnings.filterwarnings('ignore')

# stepwise_fit = auto_arima(df2['Births'], start_p=0, start_q=0,max_p=6, max_q=3, trace=True)

# print(stepwise_fit.summary())

stepwise_fit = auto_arima(df1['Thousands of Passengers'], start_p=0, start_q=0,max_p=4, max_q=4, 
                          seasonal=True, trace=True, errpr_action='ignore', suppress_warnings=True,
                          stepwise=True,m=12)
print(stepwise_fit.summary())




