import numpy as np
from .base import Optimizer
from ml.base import Trainable

class RMSprop(Optimizer):
    def __init__(self, learning_rate: float=0.001, decay: float=0, epsilon: float=1e-7, rho: float=0.9) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
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
            
            self.caches[name] = self.rho * self.caches[name] + (1 - self.rho) * grad ** 2
            
            update = self.current_learning_rate * grad / (np.sqrt(self.caches[name]) + self.epsilon)
            new_param = param - update
            trainable.set_parameter(name, new_param)

    def post_update_params(self) -> None:
        self.iterations += 1
