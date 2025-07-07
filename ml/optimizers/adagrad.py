import numpy as np
from .base import Optimizer
from ml.base import Trainable

class Adagrad(Optimizer):
    def __init__(self, learning_rate: float=1., decay: float=0., epsilon: float=1e-7) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate  = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.caches = {}
        

    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))
    
    def update_params(self, trainable: Trainable) -> None:
        params = trainable.parameters
        grads = trainable.gradients
        
        for name, param in params.items():
            grad = grads[name]
            
            if name not in self.caches:
                self.caches[name] = np.zeros_like(param)
            
            self.caches[name] += grad ** 2 
            v = self.caches[name]
                
            update = self.current_learning_rate * grad / (np.sqrt(v) + self.epsilon)
            new_param = param - update
            trainable.set_parameter(name, new_param)
    
    def post_update_params(self) -> None:
        self.iterations += 1