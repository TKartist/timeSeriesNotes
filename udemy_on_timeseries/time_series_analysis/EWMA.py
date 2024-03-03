import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

airline = pd.read_csv("../../data/airline_passengers.csv", index_col="Month", parse_dates=True)
airline.dropna(inplace=True)

#SMA
airline['sma6'] = airline["Thousands of Passengers"].rolling(window=6).mean()
airline['sma12'] = airline["Thousands of Passengers"].rolling(window=12).mean()

# EWMA example:
airline['EWMA-12'] = airline['Thousands of Passengers'].ewm(span=2).mean()

airline[["Thousands of Passengers", "sma6", "sma12", "EWMA-12"]].plot(figsize=(10, 8))
plt.show()