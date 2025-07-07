import numpy as np
from ml.base import Trainable
from .base import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate: float=0.1, decay: float=0, epsilon: float=1e-7, beta_1: float=0.9, beta_2: float=0.999) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2 
        self.caches = {}
        self.momentums = {}
        
    def pre_update_params(self) -> None:
        if self.decay: 
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))
        
    def update_params(self, trainable:Trainable) -> None:        
        params = trainable.parameters
        grads = trainable.gradients
                
        for name, param in params.items():
            grad = grads[name]     
            if name not in self.momentums:
                self.momentums[name] = np.zeros_like(param)
                self.caches[name] = np.zeros_like(param)
                
            m = self.momentums[name] = self.beta_1 * self.momentums[name] + (1 - self.beta_1) * grad
            v = self.caches[name] = self.beta_2 * self.caches[name] + (1 - self.beta_2) * (grad ** 2)
            
            m_hat = m / (1 - self.beta_1 ** (self.iterations + 1))
            v_hat = v / (1 - self.beta_2 ** (self.iterations + 1))
            
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            new_param = param - update
            
            trainable.set_parameter(name, new_param)
         
    def post_update_params(self) -> None:
        self.iterations += 1