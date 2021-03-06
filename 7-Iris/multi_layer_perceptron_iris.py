# -*- coding: utf-8 -*-
"""Multi-layer Perceptron Iris.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YlSHya96xlVRyXdqVhviX7N9SiOnpH-p

# Multi-layer perceptron Iris dataset 

![alt text](https://drive.google.com/uc?id=1xgZhek0467AtlfupqvovcjoFIJ2dB4in)

## Load the dataset
"""

from sklearn import datasets

iris = datasets.load_iris()

iris.data

iris.feature_names

iris.target

iris.target_names

inputs = iris.data[0:100]

len(inputs)

inputs.shape

outputs = iris.target[0:100]
outputs

len(outputs)

outputs.shape

outputs = outputs.reshape(-1, 1)
outputs.shape

"""## Complete neural network"""

import numpy as np

def sigmoid(sum):
  return 1 / (1 + np.exp(-sum))

def sigmoid_derivative(sigmoid):
  return sigmoid * (1 - sigmoid)

weights0 = 2 * np.random.random((4, 5)) - 1
weights1 = 2 * np.random.random((5,1)) - 1

weights0

epochs = 3000
learning_rate = 0.01

error = []

for epoch in range(epochs):
  input_layer = inputs
  sum_synapse0 = np.dot(input_layer, weights0)
  hidden_layer = sigmoid(sum_synapse0)

  sum_synapse1 = np.dot(hidden_layer, weights1)
  output_layer = sigmoid(sum_synapse1)

  error_output_layer = outputs - output_layer
  average = np.mean(abs(error_output_layer))
  
  if epoch % 1000 == 0:
    print('Epoch: ' + str(epoch + 1) + ' Error: ' + str(average))
    error.append(average)
  
  derivative_output = sigmoid_derivative(output_layer)
  delta_output = error_output_layer * derivative_output
  
  weights1T = weights1.T
  delta_output_weight = delta_output.dot(weights1T)
  delta_hidden_layer = delta_output_weight * sigmoid_derivative(hidden_layer)
  
  hidden_layerT = hidden_layer.T
  input_x_delta1 = hidden_layerT.dot(delta_output)
  weights1 = weights1 + (input_x_delta1 * learning_rate)
  
  input_layerT = input_layer.T
  input_x_delta0 = input_layerT.dot(delta_hidden_layer)
  weights0 = weights0 + (input_x_delta0 * learning_rate)

import matplotlib.pyplot as plt
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.plot(error)

def calculate_output(instance):
  hidden_layer = sigmoid(np.dot(instance, weights0))
  output_layer = sigmoid(np.dot(hidden_layer, weights1))
  return output_layer[0]

inputs[0], outputs[0]

round(calculate_output(inputs[0]))

iris.target_names

iris.target_names[int(round(calculate_output(inputs[0])))]

inputs[99], outputs[99]

iris.target_names[int(round(calculate_output(inputs[99])))]