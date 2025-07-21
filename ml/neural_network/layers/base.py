import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    """
    Abstract base class for all neural network layers.

    Defines the interface that all concrete layer implementations must follow.

    Methods:
        forward(inputs, training): Computes the forward pass.
        backward(dvalues): Computes the backward pass.
    """
    @abstractmethod
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Performs the forward pass of the layer.

        Args:
            inputs (np.ndarray): Input data to the layer.
            training (bool): Indicates whether the forward pass is during training.
        """
        raise NotImplementedError
    @abstractmethod
    def backward(self, dvalues: np.ndarray) -> None:
        """
        Performs the backward pass of the layer.

        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the output of this layer.
        """
        raise NotImplementedError
    
    