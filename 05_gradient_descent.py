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

#türev = (f(x+h) - f(x)) / h

def slope(b):
    h = 0.0001
    loss1 = loss_for_bias(b)
    loss2 = loss_for_bias(b + h)
    return (loss2 - loss1) / h

print(slope(2.5))
print(slope(-8))

#1
print('--- #1 ---')
bias = 2.5
learning_rate = 1
for i in range(100):
    bias = bias - learning_rate * slope(bias)
    print(bias, loss_for_bias(bias))

#2
print('--- #2 ---')
bias = 2.5
learning_rate = 50
for i in range(100):
    bias = bias - learning_rate * slope(bias)
    print(bias, loss_for_bias(bias))

#3
print('--- #3 ---')
bias = 8
learning_rate = 1
for i in range(100):
    bias = bias - learning_rate * slope(bias)
    print(bias, loss_for_bias(bias))


print('--- #4: gradient ascent ---')
bias = 2.5
learning_rate = 1
for i in range(100):
    bias = bias + learning_rate * slope(bias)
    print(bias, loss_for_bias(bias))