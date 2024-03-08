import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# VAR models
# In SARIMAX, the model is effected by eXogenous variables
# but NOT Vice Versa
# There are cases when the eXogenous variable and model effects
# each other, what do we do in this case? we USE VectorAutoRegression

# Case to : VAR model
# change in personal consumption expenditures C_t were forecast
# based on changes in personal disposable income i_t

# unlike AR(p), VAR model of order p,
# denoted VAR(p), considers each variable y_k in the system.

