import numpy as np 
from .base import Accuracy

class AccuracyCategorical(Accuracy):
    def __init__(self, *, binary: bool=False):
        self.binary = binary
    
    def init(self, y):
        pass

    def compare(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        
        return predictions == y