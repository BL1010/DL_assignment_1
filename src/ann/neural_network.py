import numpy as np
from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh
from ann.objective_functions import MSE, CrossEntropy
from ann.optimizers import SGD, Momentum, RMSProp, Adam, Nadam, NAG


class NeuralNetwork:

    def __init__(self, args):

        activation_map = {
            "relu": ReLU(),
            "sigmoid": Sigmoid(),
            "tanh": Tanh()
        }

        loss_map = {
            "cross_entropy": CrossEntropy(),
            "mse": MSE()
        }

        optimizer_map = {
            "sgd": SGD,
            "momentum": Momentum,
            "rmsprop": RMSProp,
            "adam": Adam,
            "nadam": Nadam,
            "nag": NAG
        }

        self.layers = []

        input_dim = getattr(args, "input_dim", 784)
        output_dim = getattr(args, "output_dim", 10)

        activation = getattr(args, "activation", "relu")
        loss = getattr(args, "loss", "cross_entropy")
        optimizer = getattr(args, "optimizer", "sgd")
        learning_rate = getattr(args, "learning_rate", 0.01)
        weight_decay = getattr(args, "weight_decay", 0.0)
        weight_init = getattr(args, "weight_init", "xavier")

        if hasattr(args, "hidden_dims"):
            hidden_dims = args.hidden_dims
        else:
            hidden_size = getattr(args, "hidden_size", 128)
            num_layers = getattr(args, "num_layers", 1)
            hidden_dims = [hidden_size] * num_layers

        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 2):
            self.layers.append(
                Dense(
                    dims[i],
                    dims[i + 1],
                    activation=activation_map[activation],
                    weight_init=weight_init
                )
            )

        self.layers.append(
            Dense(
                dims[-2],
                dims[-1],
                activation=None,
                weight_init=weight_init
            )
        )

        self.loss_fn = loss_map[loss]
        self.optimizer = optimizer_map[optimizer](learning_rate)
        self.weight_decay = weight_decay

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, logits):
        dA = self.loss_fn.backward(y_true, logits)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update_weights(self):
        params = []
        grads = []

        for layer in self.layers:
            layer.grad_W += self.weight_decay * layer.W
            params.extend([layer.W, layer.b])
            grads.extend([layer.grad_W, layer.grad_b])

        self.optimizer.update(params, grads)

    # ✅ REQUIRED FOR GRADER
    def set_weights(self, weights):
        layer_idx = 0
        for i in range(0, len(weights), 2):
            self.layers[layer_idx].W = weights[i]
            self.layers[layer_idx].b = weights[i + 1]
            layer_idx += 1

    # ✅ REQUIRED FOR SAVING BEST MODEL
    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.W)
            weights.append(layer.b)
        return weights