import numpy as np

class KNNClassifier:
    def __init__(self, k: int = 3, distance=None):
        self.k = k
        self.distance = distance
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X, self.y = X, y
        
    def predict(self, X:np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        distances = self.distance(self.X, X)
        idx_k = np.argsort(distances, axis=0)[:self.k, :]
        
        n_test = X.shape[0]
        y_pred = np.empty(n_test, dtype=self.y.dtype)
    
        for j in range(n_test):
            neighbors = idx_k[:, j]
            votes = self.y[neighbors]
            
            y_pred[j] = np.bincount(votes).argmax()
            
        return y_pred
    
    def calculate_accuracy(self, X: np.ndarray, y:np.ndarray) -> float:
        predictions = self.predict(X)
        tot = len(y)
        
        correct = np.count_nonzero(predictions == y)
        
        return correct / tot