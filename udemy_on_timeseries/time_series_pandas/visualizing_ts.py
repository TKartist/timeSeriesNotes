import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates

df = pd.read_csv("../../data/starbucks.csv", index_col="Date", parse_dates=True)
# index has to be DatetimeIndex
# df.index = pd.to_datetime(df.index) -> in case the date in dataset is still string
print(df.head())
# plt.plot(df["Volume"])
# plt.show()

title = "Title"
ylabel = "ylabel"
xlabel = "xlabel"

plt.plot(df)
plt.title(title)
plt.tight_layout()
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.ylim(50, 60)
plt.show()
