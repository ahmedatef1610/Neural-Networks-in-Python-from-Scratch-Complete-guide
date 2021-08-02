# Perceptron_3 AND

import numpy as np

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 0, 0, 1])
weights = np.array([0.0, 0.0])
learning_rate = 0.1
##########################################################################
##########################################################################
# Sum function
def sum(inputs, weights):
    return inputs.dot(weights)
##########################################################################
# Step function
def step_function(sum):
    if (sum >= 1):
        return 1
    return 0

# # Step function
# def step_function(sum):
#     if (sum.all() >= 1):
#         return 1
#     return 0
##########################################################################
# Calculate output
def calculate_output(instance):
    s = instance.dot(weights)
    return step_function(s)

calculate_output(np.array([[1, 1]]))
##########################################################################
##########################################################################
# Train
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
                    print(f'Weight {j+1} updated: {str(weights[j])}')
        print('Total error: ' + str(total_error))


train()
##########################################################################
# Classification

calculate_output(np.array([0, 0]))

calculate_output(np.array([0, 1]))

calculate_output(np.array([1, 0]))

calculate_output(np.array([1, 1]))
##########################################################################
