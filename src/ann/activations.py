from abc import ABC, abstractmethod
import numpy as np
import wandb


class Activation(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass


# -------------------------
# ReLU
# -------------------------
class ReLU(Activation):

    def forward(self, x): 
        self.x = x
        A = np.maximum(0,x) 
        
        zero_fraction = np.mean(A == 0) 
        
        return A

    def backward(self, x):
        grad = np.zeros_like(x)
        grad[x>0] = 1 
        return grad 


# -------------------------
# Sigmoid
# -------------------------
class Sigmoid(Activation):

    def forward(self, x):
        self.a =  1 / (1 + np.exp(-x))
        return self.a

    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)


# -------------------------
# Tanh
# -------------------------
class Tanh(Activation):

    def forward(self, x):
        self.a = np.tanh(x)
        return self.a

    def backward(self, x):
        t = np.tanh(x)
        return 1 - t ** 2


