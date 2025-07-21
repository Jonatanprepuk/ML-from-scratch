import numpy as np
from .base import Layer


class Dropout(Layer):
    """
    Dropout layer for regularization during training.

    Randomly sets a fraction of input units to zero during the forward pass,
    and rescales the remaining activations to maintain their expected value.

    Attributes:
        rate (float): The probability of keeping a neuron active (1 - dropout_rate).
        binary_mask (np.ndarray): The dropout mask used during training.
    """
    def __init__(self, rate: float) -> None:
        self.rate = 1 - rate 
    
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Performs the forward pass with dropout.

        During training, randomly drops neurons and rescales the rest.
        During inference, passes inputs unchanged.

        Args:
            inputs (np.ndarray): Input data.
            training (bool): Whether the model is in training mode.
        """
        if not training:
            self.output = inputs.copy()
            return 
        
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues: np.ndarray) -> None:
        """
        Performs the backward pass of dropout.

        Multiplies the incoming gradient by the dropout mask used in the forward pass.

        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the output.
        """
        self.dinputs = dvalues * self.binary_mask