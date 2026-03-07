from abc import ABC, abstractmethod
import numpy as np


class Objective(ABC):

    @abstractmethod
    def forward(self, y_true, y_pred) -> float:
        pass

    @abstractmethod
    def backward(self, y_true, y_pred):
        pass


# -----------------------------------
# Cross Entropy (Softmax compatible)
# -----------------------------------
class CrossEntropy(Objective):

    def forward(self, y_true, logits):
        epsilon = 1e-9
        
        if y_true.ndim == 1: 
            y_true = np.eye(logits.shape[1])[y_true]
        shift = logits - np.max(logits,axis = 1,keepdims = True)
        exp_vals = np.exp(shift)
        probs = exp_vals / np.sum(exp_vals,axis = 1,keepdims = True)
        m = y_true.shape[0] 
        
        loss = -np.sum(y_true*np.log(probs+epsilon))/m 
        return loss

    def backward(self, y_true, logits):
        
        if y_true.ndim == 1: 
            y_true = np.eye(logits.shape[1])[y_true]
        shift = logits- np.max(logits,axis=1,keepdims=True)
        exp_vals = np.exp(shift) 
        probs= exp_vals / np.sum(exp_vals,axis = 1, keepdims = True)
        m = y_true.shape[0]
        return (probs - y_true)


# -----------------------------------
# Mean Squared Error
# -----------------------------------
class MSE(Objective):

    def forward(self, y_true, y_pred):
        m = y_true.shape[0]
        return np.sum((y_pred - y_true) ** 2) / (2 * m)

    def backward(self, y_true, y_pred):
        m = y_true.shape[0]
        return (y_pred - y_true) / m