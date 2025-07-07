import numpy as np 
from .base import ActivationLayer

class Softmax(ActivationLayer):
    def forward(self, inputs, training: bool) -> None:
        self.inputs = inputs
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    
    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        return np.argmax(outputs, axis=1)