import numpy as np
from .base import Layer

class Flatten(Layer):
    """
    Layer that flattens the input tensor into a 2D array.

    Converts input of shape (batch_size, ..., ...) into (batch_size, features)
    by collapsing all dimensions except the batch size.

    Attributes:
        inputs (np.ndarray): Stored input for use in the backward pass.
        output (np.ndarray): Flattened output.
    """
    def forward(self, inputs: np.ndarray, training) -> None:
        """
        Performs the forward pass by flattening the input tensor.

        Args:
            inputs (np.ndarray): Input array of shape (batch_size, ..., ...).
            training (bool): Whether the model is in training mode (not used here but kept for compatibility).
        """
        self.inputs = inputs
        self.output = inputs.reshape(inputs.shape[0], -1)
        
    def backward(self, dvalues: np.ndarray) -> None:
        """
        Performs the backward pass by reshaping gradients back to the original input shape.

        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the flattened output.
        """
        self.dinputs = dvalues.reshape(self.inputs.shape)