import numpy as np 
from .base import ActivationLayer

class ReLU(ActivationLayer):
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    
    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        return outputs