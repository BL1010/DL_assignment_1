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
            "nag": NAG,
            "rmsprop": RMSProp,
            "adam": Adam,
            "nadam": Nadam
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

        # -----------------------------
        # Robust hidden layer handling
        # -----------------------------

        hidden_size = getattr(args, "hidden_size", 128)
        num_layers = getattr(args, "num_layers", 1)

        if isinstance(hidden_size, list):
            hidden_dims = hidden_size
        elif isinstance(hidden_size, int):
            hidden_dims = [hidden_size] * num_layers
        else:
            hidden_dims = [128] * num_layers

        dims = [input_dim] + hidden_dims + [output_dim]

        # -----------------------------
        # Create hidden layers
        # -----------------------------

        for i in range(len(dims) - 2):

            self.layers.append(
                Dense(
                    dims[i],
                    dims[i + 1],
                    activation=activation_map[activation],
                    weight_init=weight_init
                )
            )

        # Output layer (no activation)

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

    # -----------------------------
    # Forward Pass
    # -----------------------------

    def forward(self, X):

        for layer in self.layers:
            X = layer.forward(X)

        return X

    # -----------------------------
    # Backward Pass
    # -----------------------------

    def backward(self, y_true, logits):

        dA = self.loss_fn.backward(y_true, logits)

        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        
        grads_W = [] 
        grads_b = [] 
        
        for layer in self.layers: 
            grads_W.append(layer.grad_W) 
            grads_b.append(layer.grad_b) 
            
        return grads_W, grads_b 

    # -----------------------------
    # Update Parameters
    # -----------------------------

    def update_weights(self):

        params = []
        grads = []

        for layer in self.layers:

            layer.grad_W += self.weight_decay * layer.W

            params.extend([layer.W, layer.b])
            grads.extend([layer.grad_W, layer.grad_b])

        self.optimizer.update(params, grads)

    # -----------------------------
    # Set Weights (Grader uses this)
    # -----------------------------

    def set_weights(self, weight_dict):

        if isinstance(weight_dict, np.ndarray):
            weight_dict = weight_dict.item()

        for i, layer in enumerate(self.layers):

            w_key = f"W{i}"
            b_key = f"b{i}"

            if w_key in weight_dict:
                layer.W = np.array(weight_dict[w_key])

            if b_key in weight_dict:
                layer.b = np.array(weight_dict[b_key])

    # -----------------------------
    # Get Weights (Saving model)
    # -----------------------------

    def get_weights(self):

        weights = {}

        for i, layer in enumerate(self.layers):

            weights[f"W{i}"] = layer.W.copy()
            weights[f"b{i}"] = layer.b.copy()

        return weights