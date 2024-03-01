import pandas as pd
import numpy as np

# pands -> panel data (built on numpy)
# series/dataframe
# missing data
# groupby
# operations
# data I/O

# Series

# labels = ["a", "b", "c"]
# mylist = [10, 20, 30]
# arr = np.array(mylist)

# d = {"a": 10, "b": 20, "c": 30}

# print(pd.Series(data=mylist, index=labels))

# Dataframe : Multiple series
from numpy.random import randn

np.random.seed(101)
rand_mat = randn(5, 4)

# build dataframe from rand_mat
df = pd.DataFrame(data=rand_mat)
print(df)
df = pd.DataFrame(
    data=rand_mat, index=["a", "b", "c", "d", "e"], columns=["w", "x", "y", "z"]
)
print(df)

mylist = ["x", "y"]
print(df[mylist])

# creating new column
df["New"] = df["x"] + df["y"]
print(df)

# delete/drop a column
# df.drop("New") -> doesn't run as they don't know row or column
# df.drop("New", axis=1) -> doesn't reflect the drop
df.drop("New", axis=1, inplace=True)  # inplace is to confirm the drop
print(df)

# how to select row
# 1. df.loc['a'] -> via row name
# 2. df.iloc[0] -> via row number
print(df.loc["a"])
print(df.iloc[0])

# how to check all elements that satisfy certain conditions
print(df > 0)  # prints df of bool
print(df[df > 0])  # if df > 0, it replaces with NaN

# the following prints the row a value of column y after filtering the rows which has w value less or equal to 0
print(df[df["w"] > 0]["y"].loc["a"])

# in series "and" and "or" keywords don't work, we need to use "&" and "|"

print(df.reset_index())  # again not inplace
print(df.info())  # gives memory usage, datatype, no. cols etc.
# df.describe() -> mean, std, median, 25th, 75th percentile, max, min etc.
# df['someCol'].value_counts() -> returns the accumulation of different values

# how to drop rows with invalid data
df = pd.DataFrame({"A": [1, 2, np.nan], "B": [5, 1, np.nan], "C": [1, np.nan, 3]})

# NaN is unaccepted, so we can drop all the rows which has NaN by:
# df.dropna(inplace=True)
print("drop tables and stuff \n")
print(df)
# if we wanna drop columns, we can set axis=1 as usual

# we can also set a threshold (if more than threshold values missing, we drop)
print(df.dropna(thresh=2))

# we can fill nan with a certain value as following
print(df.fillna(121))
