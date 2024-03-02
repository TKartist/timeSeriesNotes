import pandas as pd
import numpy as np

# groupby introduction

# split, apply, combine
data = {
    "Company": ["GOOG", "GOOG", "MSFT", "MSFT", "FB", "FB"],
    "Person": ["Sam", "Charlie", "Amy", "Vanessa", "Carl", "Sarah"],
    "Sales": [200, 120, 340, 124, 243, 350],
}

df = pd.DataFrame(data)
print(df)

df2 = df.groupby("Company")
print(df2["Sales"].describe())

# how to get unique values
print(df2["Company"].unique())

# number of unique values
print(df2["Company"].unique())

# unique values and their count
print(df2["Company"].value_counts())

# print google's data on Sam
print("\n")
print(df[(df["Company"] == "GOOG") & (df["Person"] == "Sam")])


# apply on dataframe
def times_two(number):
    return number * 2


df["new"] = df["Sales"].apply(times_two)  # apply takes function as param
print(df)
