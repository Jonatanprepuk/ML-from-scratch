
import numpy as np
from ml.losses import Loss
from ml.optimizers import Optimizer
from .linear import LinearRegression
from ml.base import Trainable 

class LinearRegressionGradient(LinearRegression, Trainable):
    def __init__(self):
        self.m = 0.0
        self.c = 0.0
            
    def set(self, *, loss: Loss = None, optimizer: Optimizer=None) -> None:
        self.loss_function = loss
        self.optimizer = optimizer
        
    def backward(self, dvalues: np.ndarray, X:np.ndarray, y: np.ndarray) -> None:
        self.loss_function.backward(dvalues, y)
        dvalues = self.loss_function.dinputs
        self.dm = np.mean(dvalues * X)
        self.dc = np.mean(dvalues)
    
    def fit(self, X: np.ndarray, y: np.ndarray, *, epochs: int=1, batch_size: int=None, print_every: int=1) -> None:
        train_steps = 1
        
        if batch_size is not None:
            train_steps = len(X) // batch_size
            
            if train_steps * batch_size < len(X):
                train_steps += 1
        
        for epoch in range(1, epochs+1):
            
            self.loss_function.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step + 1) * batch_size]
                    batch_y = y[step*batch_size:(step + 1) * batch_size]
                    
                
                output = self.predict(batch_X)
                loss = self.loss_function.calculate(output, batch_y)
            
                self.backward(output, batch_X, batch_y)

                self.optimizer.pre_update_params()
                self.optimizer.update_params(self)
                self.optimizer.post_update_params()
                # self.m -= self.dm * self.learning_rate
                # self.c -= self.dc * self.learning_rate

                if not step % print_every: 
                    print(f'step: {step},' +
                        f' loss: {loss}') 
            
            acumulated_loss = self.loss_function.calculate_accumulated()
            print(f'epoch: {epoch},' +
                  f' Acumulated loss: {acumulated_loss}')
    
    @property
    def parameters(self) -> dict[str, np.ndarray]: #TODO Returnerar inte np.ndarray
        return {'m': self.m, 'c' : self.c}
    
    @property
    def gradients(self) -> dict[str, np.ndarray]:
        return {'m': self.dm, 'c' : self.dc}
    
    def set_parameter(self, name, value):
        return setattr(self, name, value)
