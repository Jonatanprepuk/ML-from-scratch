import numpy as np
from .base import Layer


class Dropout(Layer):
    def __init__(self, rate: float) -> None:
        self.rate = 1 - rate 
    
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        if not training:
            self.output = inputs.copy()
            return 
        
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues * self.binary_mask