import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        raise NotImplementedError
    @abstractmethod
    def backward(self, dvalues: np.ndarray) -> None:
        raise NotImplementedError
    
    