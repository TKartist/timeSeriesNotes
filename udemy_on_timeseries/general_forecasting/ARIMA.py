import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Case 1: ARIMA(p != 0, d = 0, q = 0) -> Basically just AutoRegression

# forecasting using a linear combination of past values of the variable.
# autoregression = regression of the variable against itself.
# An autoregression is run against a set of lagged values of order 'p'.