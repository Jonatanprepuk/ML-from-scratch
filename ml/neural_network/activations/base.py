import numpy as np
from ..layers import Layer
from abc import abstractmethod

class ActivationLayer(Layer):
    @abstractmethod
    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError