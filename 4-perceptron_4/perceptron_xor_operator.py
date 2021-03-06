# -*- coding: utf-8 -*-
"""Perceptron XOR operator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_PBzFEhneXXShf_upoV2igdFsXh6dZi1

# Perceptron XOR operator

## Inputs, outputs and weights
"""

import numpy as np

inputs = np.array([[0,0], [0,1], [1,0], [1,1]])

outputs = np.array([0, 1, 1, 0])

weights = np.array([0.0, 0.0])

learning_rate = 0.1

"""## Step function"""

def step_function(sum):
  if (sum >= 1):
    return 1
  return 0

"""## Calculate output"""

def calculate_output(instance):
  s = instance.dot(weights)
  return step_function(s)

"""## Train"""

def train():
  total_error = 1
  while (total_error != 0):
    total_error = 0
    for i in range(len(outputs)):
      prediction = calculate_output(inputs[i])
      error = abs(outputs[i] - prediction)
      total_error += error
      if error > 0:
        for j in range(len(weights)):
          weights[j] = weights[j] + (learning_rate * inputs[i][j] * error)
          print('Weight updated: ' + str(weights[j]))  
    print('Total error: ' + str(total_error))

train()