import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../data/starbucks.csv", index_col="Date", parse_dates=True)
print(df.head())
print("------------------------")
# print(df.tail())
# df = df.shift(1)
# print(df.head())
# df = df.shift(-1)
# print(df.tail())
# df = df.shift(1, freq="M")  # pushes all data in the month to last date of the month
# print(df.head())

# plt.plot(df["Close"])
# plt.show()
# rolled = df.rolling(window=30).mean()
# plt.plot(rolled["Close"])

# gets mean of all the data which happened before
expand = df["Close"].expanding().mean()
plt.plot(expand)
plt.show()
