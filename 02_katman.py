import math
inputs = [0.5, -1.2, 2.0]

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def relu(z):
    return max(0, z)

def neuron(inputs, weights, bias):
    weighted_sum = 0
    for i in range(len(inputs)):
        weighted_sum += inputs[i] * weights[i]
    weighted_sum += bias
    return sigmoid(weighted_sum)

layer_weights = [[0.3, 0.8, -0.5], [-0.7, 0.5, -0.9], [0.2, -0.1, 0.4]]
layer_biases = [0.1, 6, -6]
layer_outputs = []

### tur sayısı = len(layer_weights)
for i in range(len(layer_weights)):
    activation = neuron(inputs, layer_weights[i], layer_biases[i])
    layer_outputs.append(activation)
print(layer_outputs)
