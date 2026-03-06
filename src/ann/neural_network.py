import numpy as np

from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh
from ann.objective_functions import CrossEntropy, MSE
from ann.optimizers import SGD, Momentum, NAG, RMSProp


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
            "nag": NAG,
            "rmsprop": RMSProp
        }

        self.layers = []

        input_dim = args.input_dim
        output_dim = args.output_dim

        hidden_dims = getattr(args, "hidden_dims", [128])

        activation = activation_map[args.activation]

        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 2):

            self.layers.append(
                Dense(
                    dims[i],
                    dims[i + 1],
                    activation=activation,
                    weight_init=args.weight_init
                )
            )

        self.layers.append(
            Dense(
                dims[-2],
                dims[-1],
                activation=None,
                weight_init=args.weight_init
            )
        )

        self.loss_fn = loss_map[args.loss]
        self.optimizer = optimizer_map[args.optimizer](args.learning_rate)

        self.weight_decay = args.weight_decay

    def forward(self, X):

        for layer in self.layers:
            X = layer.forward(X)

        return X

    def backward(self, y_true, logits):

        grad = self.loss_fn.backward(y_true, logits)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self):

        params = []
        grads = []

        for layer in self.layers:

            layer.grad_W += self.weight_decay * layer.W

            params.append(layer.W)
            params.append(layer.b)

            grads.append(layer.grad_W)
            grads.append(layer.grad_b)

        self.optimizer.update(params, grads)

    def get_weights(self):

        weights = {}

        for i, layer in enumerate(self.layers):
            if hasattr(layer, "W"):
                weights[f"W{i+1}"] = layer.W
                weights[f"b{i+1}"] = layer.b

        return weights

    def set_weights(self, weights):

        if isinstance(weights, np.ndarray):
            weights = list(weights)

        layer_idx = 0

        for i in range(0, len(weights), 2):

            self.layers[layer_idx].W = weights[i].copy()
            self.layers[layer_idx].b = weights[i + 1].copy()

            layer_idx += 1