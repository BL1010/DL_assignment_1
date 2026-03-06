import numpy as np


class SGD:

    def __init__(self, lr):
        self.lr = lr

    def update(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Momentum:

    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}

    def update(self, params, grads):

        for i, (p, g) in enumerate(zip(params, grads)):

            if i not in self.v:
                self.v[i] = np.zeros_like(p)

            self.v[i] = self.beta * self.v[i] - self.lr * g
            p += self.v[i]


class NAG:

    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}

    def update(self, params, grads):

        for i, (p, g) in enumerate(zip(params, grads)):

            if i not in self.v:
                self.v[i] = np.zeros_like(p)

            v_prev = self.v[i].copy()

            self.v[i] = self.beta * self.v[i] - self.lr * g

            p += -self.beta * v_prev + (1 + self.beta) * self.v[i]


class RMSProp:

    def __init__(self, lr, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = {}

    def update(self, params, grads):

        for i, (p, g) in enumerate(zip(params, grads)):

            if i not in self.s:
                self.s[i] = np.zeros_like(p)

            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * (g ** 2)

            p -= self.lr * g / (np.sqrt(self.s[i]) + self.eps)