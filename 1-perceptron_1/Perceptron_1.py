# -*- coding: utf-8 -*-
"""Perceptron 1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16JLRDhznF-gnFllZUasPL8blX8JZSqAV

# Perceptron 1

![alt text](https://drive.google.com/uc?id=1o5DhBusTaqWFEfAxC7v20-jemnKBninM)

## Inputs and weights
"""

inputs = [35, 25]

type(inputs)

inputs[0]

inputs[1]

weights = [-0.8, 0.1]

"""## Sum function"""

def sum(inputs, weights):
  s = 0
  for i in range(2):
    #print(i)
    #print(inputs[i])
    #print(weights[i])
    s += inputs[i] * weights[i]
  return s

s = sum(inputs, weights)

s

"""## Step function"""

def step_function(sum):
  if (sum >= 1):
    return 1
  return 0

"""## Final result"""

step_function(s)