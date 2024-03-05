import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf, acf, pacf, pacf_yw, pacf_ols

import warnings
warnings.filterwarnings('ignore')

df1 = pd.read_csv('../../data/airline_passengers.csv', index_col='Month', parse_dates=True)
df1.index.freq = "MS"

df2 = pd.read_csv('../../data/DailyTotalFemaleBirths.csv', index_col="Date", parse_dates=True)
df2.index.freq = "D"

print(df1.head())
print(df2.head())

df = pd.DataFrame({'a' : [13, 5, 11, 12, 9]})
print(df)

acf_arr = acf(df['a'])
print(acf_arr)

pacf_arr = pacf_yw(df['a'], nlags=4, method='mle') # 5 rows so max 4 nlags
# mle = maximum likelihood estimation
print(pacf_arr)

pacf_arr2 = pacf_ols(df['a'], nlags=4) # pacf of ordinary least square estimates instead of yule walker
print(pacf_arr2)

# various pacf calculation methods because there are multiple methods of
# measuring correlation value. i.e. Pierson Correlation Coefficient etc.


# Now lets see plotting capability of statsmodels

from pandas.plotting import lag_plot

# lag_plot(df1['Thousands of Passengers'])
#lag_plot(df2['Births']) # Birth is stationary data so there is lack of correlation

# now let's build out acf and pacf plots

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# plot_acf(df1, lags=40)
# plot_acf(df2, lags=40)
# df2.plot(figsize=(12,5))
plot_pacf(df2, lags=40, title='Daily Female Births')
plt.show()


