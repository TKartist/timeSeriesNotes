import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import warnings
warnings.filterwarnings('ignore')
## y = mx + b + noise

m = 2
b = 3
x = np.linspace(0, 50, 100)
np.random.seed(101)
noise = np.random.normal(loc=0, scale=4, size=len(x))
y = 2 * x + b + noise
# plt.plot(x,y,'*')
# plt.show()

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=1,activation='relu')) # 4 = no. neurons,
# relu = rectified linear unit
# input dim = input dimension and it is 1 because there is only x for y 
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mse',optimizer='adam')
print(model.summary())

z = model.fit(x,y,epochs=400)
# if you still see the data decreasing after the end of epochs, you can increase them
# until it doesn't decrease anymore

loss = model.history.history['loss']
epochs = range(len(loss))
# plt.plot(epochs, loss)
# plt.show()

x_for_predictions = np.linspace(0, 50, 100)
y_pred = model.predict(x_for_predictions)
plt.plot(x,y,'*')
plt.plot(x_for_predictions, y_pred, 'r')
plt.show()
