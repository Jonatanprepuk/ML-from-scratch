from typing import Any

import numpy as np
from ml.base import Trainable
from .base import Layer
from ml.neural_network import Softmax


class Attention(Layer, Trainable):
    
    def __init__(self, d_model: int, d_k) -> None:
        self.weights_Q = np.random.randn(d_model, d_k)
        self.weights_K = np.random.randn(d_model, d_k)
        self.weights_V = np.random.randn(d_model, d_k)
        self.d_k = d_k
        self.softmax = Softmax()
   
    def forward(self, inputs: np.ndarray, training: bool) -> None:
       self.inputs = inputs
       self.Q = inputs @ self.weights_Q
       self.K = inputs @ self.weights_K
       self.V = inputs @ self.weights_V
       
       self.S = (self.Q @ np.transpose(self.K, (0, 2, 1))) / np.sqrt(self.d_k)
       
       self.softmax.forward(self.S, training=training)
       self.A = self.softmax.output
              
       self.output = self.A @ self.V 
       
   
    def backward(self, dvalues: np.ndarray) -> None:
        pass
    
    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return super().parameters
    
    @property
    def gradients(self) -> dict[str, np.ndarray]:
        return super().gradients
    
    def set_parameter(self, name: str, value: np.ndarray) -> None:
        pass 

if __name__ == "__main__":
    attention = Attention(d_model=32, d_k=16)
    x = np.random.rand(4, 10, 32)
    print(x.shape)
    attention.forward(x, training=False)
    
    print(np.sum(attention.A, axis=-1))
    