import numpy as np
from src.ann.neural_network import NeuralNetwork


class DummyArgs:
    input_dim = 2
    hidden_dims = [2]
    output_dim = 2
    num_layers = 1
    activation = "relu"
    loss = "cross_entropy"
    optimizer = "sgd"
    learning_rate = 0.01
    weight_decay = 0.0
    weight_init = "random"


# -------------------------
# Initialize model
# -------------------------
args = DummyArgs()
model = NeuralNetwork(args)

# Manually set weights
model.layers[0].W = np.array([[1.0, 2.0],
                              [3.0, 4.0]])
model.layers[0].b = np.array([0.0, 0.0])

# Input
X = np.array([[1.0, 1.0]])

# True label (class 0)
y_true = np.array([[1.0, 0.0]])


# -------------------------
# FORWARD CHECK
# -------------------------
print("Forward Output:")
out = model.forward(X)
print(out)

# Expected before softmax:
# [4, 6]
# If softmax inside loss, output may be logits.


# -------------------------
# GRADIENT CHECK
# -------------------------
epsilon = 1e-5
layer = model.layers[0]

# Run backward once to compute analytical gradient
y_pred = model.forward(X)
loss = model.loss_fn.forward(y_true,y_pred)
model.backward(y_true, y_pred)

analytical_grad = layer.grad_W[0, 0]

# Numerical gradient
original = layer.W[0, 0]

# W + epsilon
layer.W[0, 0] = original + epsilon
loss_plus = model.loss_fn.forward(y_true, model.forward(X))

# W - epsilon
layer.W[0, 0] = original - epsilon
loss_minus = model.loss_fn.forward(y_true, model.forward(X))

# Restore weight
layer.W[0, 0] = original

numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

print("\nGradient Check for W[0,0]:")
print("Analytical Gradient :", analytical_grad)
print("Numerical Gradient  :", numerical_grad)
print("Difference          :", abs(analytical_grad - numerical_grad))