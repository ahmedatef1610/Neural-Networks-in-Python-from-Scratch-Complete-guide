# Perceptron_1

import numpy as np

inputs = np.array([35, 25])
weights = np.array([-0.8, 0.1])
##########################################################################
# Sum function
def sum(inputs, weights):
      return inputs.dot(weights)

s = sum(inputs, weights)
##########################################################################
# Step function
def step_function(sum):
    if (sum >= 1):
        return 1
    return 0
##########################################################################
# Final result
step_function(s)