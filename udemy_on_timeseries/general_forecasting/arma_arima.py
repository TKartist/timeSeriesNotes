import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# forecasting library
from statsmodels.tsa.arima_model import ARMA, ARIMA, ARMAResults, ARIMAResults
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df1 = pd.read_csv("../../DailyTotalFemaleBirths.csv", index_col="Date", parse_dates=True)
df1.index.freq = "D"
df1 = df1[:120]
df2 = pd.read_csv()