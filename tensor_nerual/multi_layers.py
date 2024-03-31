import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# data generation
# y = x^2 + 5 * x + 3
x = np.arange(-80, 81, 0.5)
y = x ** 2 + 5 * x + 3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# model tow
model2 = keras.Sequential()
model2.add(keras.layers.Dense(units=256, activation='relu', input_shape=[1]))
model2.add(keras.layers.Dense(units=256, activation='relu'))
model2.add(keras.layers.Dense(units=256, activation='relu'))
model2.add(keras.layers.Dense(units=256, activation='relu'))
model2.add(keras.layers.Dense(units=1))

optimizer = keras.optimizers.RMSprop(0.0001)
loss = keras.losses.MeanSquaredError()
metric = tf.metrics.RootMeanSquaredError()

model2.compile(optimizer=optimizer, loss=loss, metrics=[metric])
model2.fit(x_train, y_train, batch_size=1, epochs=100, validation_data=(x_test, y_test))

model2.compile(optimizer='adam', loss='mean_squared_error')

hist2 = model2.fit(x_train, y_train, batch_size=64, epochs=1000, validation_data=(x_test, y_test))
