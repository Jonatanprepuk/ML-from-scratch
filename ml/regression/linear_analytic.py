import numpy as np
from ml.regression.linear import LinearRegression

class LinearRegressionAnalytic(LinearRegression):
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        num = np.sum((X - X.mean()) * (y - y.mean()))
        den = np.sum(np.square(X - X.mean()))
        
        self.m = num / den
        self.c = y.mean() - self.m * X.mean()
    
    