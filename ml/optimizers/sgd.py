import numpy as np
from .base import Optimizer
from ml.base import Trainable

class SGD(Optimizer):
    def __init__(self, learning_rate: float=1., decay: float=0., momentum: float=0.) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate  = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.momentums = {}

    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))
    
    def update_params(self, trainable: Trainable) -> None:
        params = trainable.parameters
        grads = trainable.gradients

        for name, param in params.items():
            grad = grads[name]
            
            if self.momentum:
                if name not in self.momentums:
                    self.momentums[name] = np.zeros_like(param)
                
                update = self.momentum * self.momentums[name] - self.current_learning_rate * grad
                self.momentums[name] = update
            else:
                update = -self.current_learning_rate * grad
            
            new_param = param + update
            trainable.set_parameter(name, new_param)
    
    def post_update_params(self) -> None:
        self.iterations += 1