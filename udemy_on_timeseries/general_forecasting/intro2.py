import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.tools import diff

df2 = pd.read_csv('../../data/samples.csv', index_col=0, parse_dates=True)
# df2.plot(subplots=True)
# 'a' is stationary
# 'b' is non-stationary
# plt.show()
col_b_diff = df2['b'] - df2['b'].shift(1) # this manual process can be calculated using "diff" function -> diff(df2['b'],k_diff=1)
# print(col_b_diff)
diff(df2['b'],k_diff=1).plot() # creates a series of differences all(df[n] - df[n - 1])
plt.show()