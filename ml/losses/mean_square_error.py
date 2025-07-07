import numpy as np
from .base import Loss

class MeanSquareError(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        sample_loss = np.mean((y_true - y_pred) ** 2, axis=-1)

        return sample_loss
    
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples