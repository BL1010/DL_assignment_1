from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dA):
        pass


class Dense(Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 activation=None,
                 weight_init: str = "xavier"):
        
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.activation = activation 
        if weight_init == 'xavier': 
            self.W = np.random.randn(input_dim,output_dim)*np.sqrt(1/input_dim)
        elif weight_init == 'random': 
            self.W = np.random.randn(input_dim,output_dim)*0.01
        elif weight_init == "zeros": 
            self.W = np.zeros((input_dim,output_dim))
        else: 
            raise ValueError("Initialization not implemented")
        
        self.b = np.zeros((1,output_dim)) 
        
        self.X = None 
        self.Z = None 
        self.grad_W = None 
        self.grad_b = None



    def forward(self, X):

        self.X = X
        self.Z = X @ self.W + self.b

        if self.activation: 
            return self.activation.forward(self.Z)
        return self.Z

    def backward(self, dA):

        m = self.X.shape[0]

        if self.activation:
            dZ = dA * self.activation.backward(self.Z)
        else:
            dZ = dA

        m = self.X.shape[0]
        self.grad_W = self.X.T @ dZ
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) 

        return dZ @ self.W.T

       