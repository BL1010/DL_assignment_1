from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):

    def __init__(self, lr):
        self.lr = lr

    @abstractmethod
    def update(self, params, grads):
        pass


# ------------------------------------------------
# SGD
# ------------------------------------------------
class SGD(Optimizer):

    def update(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


# ------------------------------------------------
# Momentum (Heavy Ball)
# ------------------------------------------------
class Momentum(Optimizer):

    def __init__(self, lr, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = {}

    def update(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.v:
                self.v[i] = np.zeros_like(p)

            self.v[i] = self.beta * self.v[i] - self.lr * g
            p += self.v[i]


# ------------------------------------------------
# Nesterov Accelerated Gradient (NAG)
# ------------------------------------------------
class NAG(Optimizer):

    def __init__(self, lr, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = {}

    def update(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.v:
                self.v[i] = np.zeros_like(p)

            v_prev = self.v[i].copy()
            self.v[i] = self.beta * self.v[i] - self.lr * g
            p += -self.beta * v_prev + (1 + self.beta) * self.v[i]


# ------------------------------------------------
# RMSProp
# ------------------------------------------------
class RMSProp(Optimizer):

    def __init__(self, lr, beta=0.9, eps=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.s = {}

    def update(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.s:
                self.s[i] = np.zeros_like(p)

            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * (g ** 2)
            p -= self.lr * g / (np.sqrt(self.s[i]) + self.eps)


# ------------------------------------------------
# Adam
# ------------------------------------------------
class Adam(Optimizer):

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):

        self.t += 1

        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ------------------------------------------------
# Nadam (Adam + Nesterov momentum)
# ------------------------------------------------
class Nadam(Optimizer):

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):

        self.t += 1

        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            m_nesterov = (
                self.beta1 * m_hat +
                (1 - self.beta1) * g / (1 - self.beta1 ** self.t)
            )

            p -= self.lr * m_nesterov / (np.sqrt(v_hat) + self.eps)