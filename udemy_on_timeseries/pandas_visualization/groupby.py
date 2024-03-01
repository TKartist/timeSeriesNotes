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
