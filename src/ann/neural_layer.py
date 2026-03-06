import numpy as np


class Dense:

    def __init__(self, in_dim, out_dim, activation=None, weight_init="xavier"):

        if weight_init == "xavier":
            limit = np.sqrt(6 / (in_dim + out_dim))
            self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        elif weight_init == "zeros":
            self.W = np.zeros((in_dim, out_dim))
        else:
            self.W = np.random.randn(in_dim, out_dim) * 0.01

        self.b = np.zeros(out_dim)

        self.activation = activation

    def forward(self, x):

        self.x = x
        z = x @ self.W + self.b

        if self.activation is not None:
            self.z = z
            return self.activation.forward(z)

        return z

    def backward(self, grad):

        if self.activation is not None:
            grad = self.activation.backward(grad)

        self.grad_W = self.x.T @ grad / self.x.shape[0]
        self.grad_b = np.mean(grad, axis=0)

        dx = grad @ self.W.T

        return dx