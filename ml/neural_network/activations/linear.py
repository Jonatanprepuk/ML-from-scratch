import numpy as np
from .base import ActivationLayer

class Linear(ActivationLayer):
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        self.inputs = inputs 
        self.output = inputs
    
    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues.copy()

    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        return outputs