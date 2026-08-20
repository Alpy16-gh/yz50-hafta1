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
layer_biases = [0.1, -3, 5]
layer_outputs = []

for i in range(len(layer_weights)):
    activation = neuron(inputs, layer_weights[i], layer_biases[i])
    layer_outputs.append(activation)
print(layer_outputs)

targets = [0, 0, 1]
def loss(outputs, targets):
    total_loss = 0
    for i in range(len(outputs)):
        total_loss += (outputs[i] - targets[i]) ** 2
    return total_loss 
print(loss(layer_outputs, targets))
