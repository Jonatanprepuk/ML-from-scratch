import numpy as np 
from .base import ActivationLayer

class SELU(ActivationLayer):
    def __init__(self, alpha: float=1.67326, lam: float=1.0507):
        self.alpha = alpha
        self.lam = lam 
        
    def forward(self, inputs: np.ndarray, training: np.ndarray) -> None:
        self.inputs = inputs
        self.output = self.lam * np.where(
            inputs > 0,
            inputs,
            self.alpha * (np.exp(inputs) - 1)
        )  
    
    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = self.lam * np.where(
            self.inputs > 0,
            dvalues * self.lam,
            dvalues * (self.lam * self.alpha * np.exp(self.inputs))
        )
    
    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        #Don't use SELU as the final layer
        return outputs
         
    