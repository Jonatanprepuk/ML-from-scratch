import numpy as np 
from .base import ActivationLayer

class Tanh(ActivationLayer):
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        self.inputs = inputs
        self.output = np. tanh(inputs) # (np.exp(inputs) -np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output ** 2)
    
    def predictions(self, outputs):
        return (outputs > 0) * 1
    