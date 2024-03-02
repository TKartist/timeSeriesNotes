import pandas as pd
import numpy as np
from datetime import datetime

# a lot of methods for timeseries data in pandas

# Datetime index

my_year = 2020
my_month = 1
my_day = 2
my_hour = 13
my_min = 30
my_sec = 15

my_date = datetime(my_year, my_month, my_day)

print(my_date)

my_datetime = datetime(my_year, my_month, my_day, my_hour, my_min, my_sec)
print(my_datetime)

timeArr = np.array(["2020-03-15", "2020-03-16", "2020-03-17"], dtype="datetime64[Y]")
print(timeArr)

timeArrange = np.arange("2018-06-01", "2018-06-23", 7, dtype="datetime64[D]")
print(timeArrange)

timePd = pd.date_range("2020-01-01", periods=7, freq="M")
print(timePd)

timeUS = pd.to_datetime(["2/1/2018", "3/1/2018"], format="%d/%m/%Y")
# default without format is US
print(timeUS)
print(timeUS.max())
