import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, grad):
        pass


class ReLU(Activation):

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        dx = grad.copy()
        dx[self.x <= 0] = 0
        return dx


class Sigmoid(Activation):

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)


class Tanh(Activation):

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out ** 2)