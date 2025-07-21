import numpy as np 
from .base import ActivationLayer

class LeakyReLU(ActivationLayer):
    def __init__(self, *, alpha: float=0.01):
        self.alpha = alpha
        
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] *= self.alpha
        
    def predictions(self, outputs):
        return outputs