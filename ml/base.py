import numpy as np
from abc import ABC, abstractmethod

class Trainable(ABC):
    @property
    @abstractmethod
    def parameters(self) -> dict[str, np.ndarray]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def gradients(self) -> dict[str, np.ndarray]:
        raise NotImplementedError
    
    @abstractmethod
    def set_parameter(self, name:str, value: np.ndarray) -> None:
        raise NotImplementedError