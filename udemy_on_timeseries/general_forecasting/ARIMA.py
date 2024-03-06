import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Case 1: ARIMA(p != 0, d = 0, q = 0) -> Basically just AutoRegression

# forecasting using a linear combination of past values of the variable.
# autoregression = regression of the variable against itself.
# An autoregression is run against a set of lagged values of order 'p'.

# autoregressive model specifies that the output variable depends linearly on
# its own previous values and on a stochastic term (imperfectly predictable term)

# Formula for AR (AutoRegression)
# y(t) = c + phi_1 * y_(t - 1) + phi_2 * y_(t - 2) + ... + phi_p * y_(t - p) + eps_t
# where phi = lag coefficient
# eps_t = white noise

# AR(1) = y(t) = c + phi_1 * y_(t - 1) + eps_t
# predicting y(t) just from y_(t - 1)

# Don't go too far back as it will pick up unnecessary noise
# Also very mathematically complex

