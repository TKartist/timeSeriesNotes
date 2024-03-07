import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from statsmodels.tools.eval_measures import mse, rmse, meanabs
from statsmodels.graphics.tsaplots import month_plot, quarter_plot
# above plot, plots time series data into month and quarters

np.random.seed(42)
df = pd.DataFrame(np.random.randint(20, 30, (50, 2)), columns=['test', 'predictions'])
# print(df)

eval = meanabs(df['test'], df['predictions'])
# print(eval)

df = pd.read_csv("../../data/airline_passengers.csv", index_col="Month", parse_dates=True)
df.index.freq = "MS"

month_plot(df["Thousands of Passengers"])
plt.show()

# resample data to quarterly
dfq = df["Thousands of Passengers"].resample(rule="Q").mean()
# quarter_plot(dfq)
# plt.show()
