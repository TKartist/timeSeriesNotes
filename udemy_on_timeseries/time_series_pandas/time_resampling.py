import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../../data/starbucks.csv")
# print(df.head())

df2 = pd.read_csv("../../data/starbucks.csv", index_col="Date")
# print(df2.head())

print("_______________________________\n")
# print(df2.index)

# parse_dates, converts Dates into proper DateTimeIndex
df3 = pd.read_csv("../../data/starbucks.csv", index_col="Date", parse_dates=True)
# print(df3.index)
print("-------------------------\n")
x = df3.resample(rule="M").mean()  # -> can use 'Apply' function as well
# print(x)

z = df3["Close"].resample("A").mean().plot.bar(title="Good", color=["#1f77b4"])
