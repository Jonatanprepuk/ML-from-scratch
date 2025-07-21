import numpy as np 
from .base import ActivationLayer

class Sigmoid(ActivationLayer):
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        return (outputs > 0.5) * 1