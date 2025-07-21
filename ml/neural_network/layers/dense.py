import numpy as np
from ml.base import Trainable
from .base import Layer

class Dense(Layer, Trainable):
    """
    A fully connected (dense) layer for a neural network.

    This layer performs a linear transformation of the input: 
    output = input * weights + biases. It also supports L1 and L2 
    regularization for both weights and biases.
    """
    
    def __init__(self, n_inputs: int, n_neurons: int, 
                 weight_regularizer_l1: float = 0, weight_regularizer_l2: float = 0,
                 bias_regularizer_l1: float = 0, bias_regularizer_l2: float = 0) -> None:

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
        self.input_shape = (None, n_inputs)
        self.output_shape = (None, n_neurons)
        
        self.name =f'Dense_{id(self)}'

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Performs the forward pass of the dense layer.

        Computes the output as the dot product of inputs and weights, plus biases:
        output = inputs · weights + biases

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, n_inputs).
            training (bool): Indicates whether the layer is in training mode (not used here but included for compatibility).
        """
        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Performs the backward pass of the dense layer.

        Computes gradients with respect to weights, biases, and inputs.
        Applies L1 and L2 regularization to the gradients if specified.

        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the output of this layer, 
                                  of shape (batch_size, n_neurons).
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1 
            self.dweights += self.weight_regularizer_l1 * dL1
        
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)
    
    @property
    def parameters(self) -> dict[str, np.ndarray]:
        """
        Returns a dictionary containing the layer's parameters.

        Returns:
            dict[str, np.ndarray]: Dictionary with keys '<name>_weights' and '<name>_biases'.
        """
        return {f'{self.name}_weights': self.weights, f'{self.name}_biases': self.biases}
    
    @property
    def gradients(self) -> dict[str, np.ndarray]:
        """
        Returns a dictionary containing the gradients of the layer's parameters.

        Returns:
            dict[str, np.ndarray]: Dictionary with keys '<name>_weights' and '<name>_biases'.
        """
        return {f'{self.name}_weights' : self.dweights, f'{self.name}_biases' : self.dbiases}
    
    def set_parameter(self, name, value):
        """
        Sets the value of a parameter (weight or bias) by name.

        This is typically used when loading saved weights. The method extracts
        the actual attribute name from the provided key and updates it.

        Args:
            name (str): Name of the parameter to set.
            value (np.ndarray): New value for the parameter.
        """
        if '_' in name:
            name = name.split('_', 2)[2]
            
        setattr(self, name, value)
        
