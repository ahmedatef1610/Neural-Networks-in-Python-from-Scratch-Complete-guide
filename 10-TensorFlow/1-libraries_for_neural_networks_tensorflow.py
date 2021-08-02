# -*- coding: utf-8 -*-
"""Libraries for Neural Networks - TensorFlow.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QZrnbFIfhzvBToRJY93DIsKdEO5LdvCr

# Libraries for Neural Networks - TensorFlow (image classification)

## Loading the dataset
"""

import tensorflow
tensorflow.__version__

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0], cmap='gray')
plt.title('Class: ' + str(y_train[0]))

X_train.shape

X_train = X_train.reshape(60000, 28 * 28)
X_train.shape

X_train[0]

X_test.shape

X_test = X_test.reshape(10000, 28 * 28)
X_test.shape

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train[0]

X_train /= 255
X_test /= 255

X_train.max()

X_train.min()

y_train

y_train = np_utils.to_categorical(y_train)

y_train[0]

y_test = np_utils.to_categorical(y_test)
y_test

"""## Neural network (training)"""

(784 + 10) / 2

# 784 -> 397 -> 397 -> 10
network = Sequential()
network.add(Dense(input_shape = (784,), units = 397, activation = 'relu'))
network.add(Dense(units=397, activation='relu'))
network.add(Dense(units = 10, activation='softmax'))

network.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

history = network.fit(X_train, y_train, batch_size=128, epochs=10)

"""## Neural network (evaluation)"""

history.history.keys()

plt.plot(history.history['loss'])

plt.plot(history.history['accuracy'])

accuracy_test = network.evaluate(X_test, y_test)

predictions = network.predict(X_test)

predictions[0]

np.argmax(predictions[0])

plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
plt.title('Class: ' + str(y_test[0]))