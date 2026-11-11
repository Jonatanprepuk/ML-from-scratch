import numpy as np 
from .base import ActivationLayer

class Softmax(ActivationLayer):
    def forward(self, inputs, training: bool) -> None:
        self.inputs = inputs
        exp_val = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        probabilities = exp_val / np.sum(exp_val, axis=-1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = self.output * (dvalues - np.sum(self.output * dvalues, axis=-1, keepdims=True))
    
    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        return np.argmax(outputs, axis=-1)