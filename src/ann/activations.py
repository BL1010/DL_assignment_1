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
        #wandb.log({"relu_zero_fraction":zero_fraction})
        return A

    def backward(self, x):
        return (x > 0).astype(float)


# -------------------------
# Sigmoid
# -------------------------
class Sigmoid(Activation):

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)


# -------------------------
# Tanh
# -------------------------
class Tanh(Activation):

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        t = np.tanh(x)
        return 1 - t ** 2


# -------------------------
# Softmax (Batch Safe)
# -------------------------
class Softmax(Activation):

    def function(self, x: np.ndarray) -> np.ndarray:
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_vals = np.exp(shifted_x)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Do not use Softmax derivative explicitly. "
            "Use CrossEntropy + Softmax combined gradient: y_pred - y_true."
        )