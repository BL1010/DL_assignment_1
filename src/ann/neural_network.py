import numpy as np
from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Softmax, Tanh
from ann.objective_functions import MSE, CrossEntropy
from ann.optimizers import SGD, Momentum, Adam, Nadam, RMSProp, NAG
import wandb 

class NeuralNetwork: 
    
    def __init__(self,args): 
        activation_map = {
            "relu":ReLU(), 
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
            "rmsprop": RMSProp, 
            "adam": Adam, 
            "nadam": Nadam,
            "nag":NAG
        }
        
        self.layers = [] 
        dims = [args.input_dim]+ args.hidden_dims + [args.output_dim] 
        
        for i in range(len(dims)-2): 
            self.layers.append(
                Dense(dims[i],dims[i+1],
                      activation=activation_map[args.activation],
                      weight_init=args.weight_init)
            )
        self.layers.append(
            Dense(dims[-2],dims[-1],
                  activation = None,
                  weight_init=args.weight_init)
        )
        
        self.loss_fn = loss_map[args.loss] 
        self.optimizer = optimizer_map[args.optimizer](args.learning_rate) 
        
        self.weight_decay = args.weight_decay 
        
    def forward(self,X): 
        for layer in self.layers: 
            X = layer.forward(X) 
        return X 
    
    def backward(self,y_true,logits): 
        dA = self.loss_fn.backward(y_true,logits) 
        
        for layer in reversed(self.layers): 
            dA = layer.backward(dA) 
            
        
        
                
            
    def update_weights(self): 
        params = [] 
        grads = [] 
        
        for layer in self.layers: 
            layer.grad_W+= self.weight_decay * layer.W
            params.extend([layer.W,layer.b]) 
            grads.extend([layer.grad_W,layer.grad_b]) 
        self.optimizer.update(params,grads) 
        