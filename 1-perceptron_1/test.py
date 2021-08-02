# Perceptron_1

inputs = [35, 25]
weights = [-0.8, 0.1]
##########################################################################
# Sum function
def sum(inputs, weights):
    s = 0
    for i in range(2):
        s += inputs[i] * weights[i]
    return s

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