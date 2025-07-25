import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    def remember_trainable_layers(self, trainable_layers: np.ndarray) -> None:
        self.trainable_layers = trainable_layers

    def regularization_loss(self) -> np.ndarray:
        regularization_loss = 0

        for layer in self.trainable_layers:

            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss
    
    def calculate(self, output: np.ndarray, y: np.ndarray, *, include_regularization: bool=False)  -> np.ndarray:
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization: bool=False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    def new_pass(self) -> None:
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray)  -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        raise NotImplementedError
        
    