# SLP_homework homework_solution_salary_increase

import numpy as np

inputs = np.array([[18,2], [20,3], [21, 4], [35,15], [36,16], [38, 18]])
##########################################################################
# MinMax scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
inputs = scaler.fit_transform(inputs)
outputs = np.array([0, 0, 0, 1, 1, 1])
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
##########################################################################
# Calculate output
def calculate_output(instance):
    s = sum(instance,weights)
    return step_function(s)
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
# Graph
import pandas as pd

df1 = pd.DataFrame(data=inputs, columns=["age", "educational"])
df1

df2 = pd.DataFrame(data=outputs, columns=["class"])
df2

df3 = pd.concat([df1, df2], axis=1)
df3

import seaborn as sns

sns.relplot(x="age", y="educational", data = df3, hue = "class")

# must be Linearly separable

##########################################################################
# Classification

inputs
weights

calculate_output(np.array([0.,0.]))
calculate_output(np.array([1.,1.]))
##########################################################################

test_inputs = np.array([[17,5], [25,8], [45,10], [31,20]])
test_inputs = scaler.transform(test_inputs)
test_inputs

for i in range(len(test_inputs)):
    #print(test_inputs[i])
    print(calculate_output(test_inputs[i]))
##########################################################################
