import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing Holt-Winter Model
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# for evaluation of forecast
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("../../data/airline_passengers.csv", index_col="Month", parse_dates=True)
df.index.freq = "MS" # set index frequency to Month Start

# print(df.head())
# print(df.tail())
# the data shows we have info from beginning of 1949 to end of 1960 -> so let's say we try to predict the future till 3 years ahead (till end of 1963)

train_data = df.iloc[:109] #.loc[:'1957-01-01']
test_data = df.iloc[108:] # one lay over

fitted_model = ExponentialSmoothing(train_data['Thousands of Passengers'], 
                                    trend='mul', 
                                    seasonal='mul', 
                                    seasonal_periods=12
                                    ).fit()
test_predictions = fitted_model.forecast(36) # forecast 3 years to the future
# print(test_predictions)
# ignore the warnings
# train_data['Thousands of Passengers'].plot(legend=True, label='TRAIN', figsize=(12, 8))
# test_data['Thousands of Passengers'].plot(legend=True, label="TEST")
# test_predictions.plot(legend=True, label="PREDICTION")
# plt.show()

abs_error = mean_absolute_error(test_data, test_predictions)
print(abs_error)
sqr_error = mean_squared_error(test_data, test_predictions)
print(sqr_error)
rmse = np.sqrt(sqr_error)
print(rmse)

final_model = ExponentialSmoothing(df['Thousands of Passengers'], trend = 'mul', seasonal='mul', seasonal_periods=12).fit()

forecast_prediction = final_model.forecast(36)
df['Thousands of Passengers'].plot(figsize=(12, 8))
forecast_prediction.plot()
plt.show()