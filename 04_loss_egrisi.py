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

def loss_for_bias(b):
    biases = [b, -3, 5]
    outputs = []
    for i in range(len(layer_weights)):
        activation = neuron(inputs, layer_weights[i], biases[i])
        outputs.append(activation)
    return loss(outputs, targets)
print(loss_for_bias(0.1))

bias_values = []
loss_values = []
for i in range(-20,21):
    b = i / 2
    loss_values.append(loss_for_bias(b))
    bias_values.append(b)
print(bias_values)
print(loss_values)

import matplotlib.pyplot as plt
plt.plot(bias_values, loss_values)
plt.xlabel('Bias Value')
plt.ylabel('Loss Value')
plt.show() 