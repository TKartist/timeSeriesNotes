# forecasting time series data with Keras and RNN
import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../../data/Alcohol_Sales.csv", index_col="DATE", parse_dates=True)
df.index.freq = "MS"
df.columns = ['Sales']

# df.plot(figsize=(16, 10))
# plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

results = seasonal_decompose(df['Sales'])

train = df.iloc[:313]
test = df.iloc[313:]

# we have to scale or normalize data for Neural Network because large difference cause confusion for neurons
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train) # find the maximum, we find what to scale to basically
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# Neurons want certain format (batches) for time series data, so we convert it to the format with the following lib
from keras.preprocessing.sequence import TimeseriesGenerator

n_input = 12 # so we will use 12 months to predict the 13th month
n_features = 1 # number of columns

generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1) # takes in a sequence of data-points gathered at equal
# intervals, along with time series parameters such as stride, length of history, etc., to produce batches
# for training/validation

# essentially, len(scaled_train) - len(generator) = n_input
# it is because "n_input" values are used to predict the next time-series value
# hence if n_input = 2, and train data array is 313, we can try predicting 311 times

# you can try manipulating the TimeseriesGenerator to see improvement of the performance

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features))) # LSTM layer
# add 150 neurons to LSTM (can be bigger or smaller but we have to find out manually by trying out different values)
model.add(Dense(1)) # output layer
model.compile(optimizer='adam', loss='mse')

model.fit_generator(generator, epochs=25)
# plt.plot(range(len(model.history.history['loss'])), model.history.history['loss'])
# plt.show()

# Forecasting with RNN model
test_predictions = [] # holding predictions
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    # predict one timestep ahead of historical 12 points
    current_pred = model.predict(current_batch)[0] # result of predict is [[x]] so by calling [0], we get [x]

    # store prediction
    test_predictions.append(current_pred)

    # update current batch to include the prediction
    # prev shape of current batch (1, 12, 1)
    # by calling current_batch[:,1:,:], we maintain all shape except the middle one
    # second shape dimension will be from 1 to 12 instead of 0 to 12, ignoring first val of 2nd dimension
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

true_pred = scaler.inverse_transform(test_predictions)
print(true_pred)
print(n_input)
test['Predictions'] = true_pred
test.plot(figsize=(12, 8))
plt.show()

# saving trained model
model.save('firstRNNModel.h5')

# loading model
from keras.models import load_model
model1 = load_model('firstRNNModel.h5')