import numpy as np
from abc import ABC, abstractmethod

class Accuracy(ABC):
    def calculate(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy
    
    def calculate_accumulated(self) -> np.ndarray:
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    
    def new_pass(self) -> None:
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    @abstractmethod
    def init(self, y, reinit: bool=False) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def compare(self, predictions, y) -> np.ndarray:
        raise NotImplementedError