import math
inputs = [0.5, -1.2, 2.0]
weights = [0.3, 0.8, -0.5]
bias = 0.1
print(inputs, weights, bias)

weighted_sum = 0
for i in range(len(inputs)):
    weighted_sum += inputs[i] * weights[i]
weighted_sum += bias
print(weighted_sum)

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def relu(z):
    return max(0, z)
print(relu(weighted_sum))

def neuron(inputs, weights, bias):
    weighted_sum = 0
    for i in range(len(inputs)):
        weighted_sum += inputs[i] * weights[i]
    weighted_sum += bias
    return sigmoid(weighted_sum)
print(neuron(inputs, weights, bias))
