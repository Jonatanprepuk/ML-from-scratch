import numpy as np
from .base import Layer

class Flatten(Layer):
    def forward(self, inputs: np.ndarray, training) -> None:
        self.inputs = inputs
        self.output = inputs.reshape(inputs.shape[0], -1)
        
    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues.reshape(self.inputs.shape)