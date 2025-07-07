
import numpy as np
from .base import Layer

class LayerInput(Layer):
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        self.output = inputs
    
    def backward(self, dvalues: np.ndarray) -> None:
        pass
    



