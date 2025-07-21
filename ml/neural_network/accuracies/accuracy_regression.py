import numpy as np
from .base import Accuracy

class AccuracyRegression(Accuracy):
    def __init__(self) -> None:
        self.precision = None
    
    def init(self, y, reinit: bool=False) -> None:
        if self.precision is None or reinit:
            self.precision = np.std(y)/250
        
    def compare(self, predictions, y) -> np.ndarray:
        return np.absolute(predictions-y) < self.precision