import numpy as np 
from abc import ABC, abstractmethod

class LinearRegression(ABC):
    
    @abstractmethod
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        raise NotImplementedError 
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.multiply(self.m, X) + self.c
    
    def summary(self, X: np.ndarray, y:np.ndarray) -> None:
        observations = len(X)
        
        y_pred = self.predict(X)
        residuals = y_pred - y 
        
        variance_residuals = np.var(residuals)
        variance_y = np.var(y)
        r_squared = 1 - (variance_residuals / variance_y)
        
        mse = np.mean((residuals) ** 2)
        
        root_mse = np.sqrt(mse)
        SE_m = np.sqrt((np.sum(residuals ** 2) / (observations-2))\
            / np.sum((X - np.mean(X)) ** 2))
        
        SE_c = np.sqrt(np.sum(residuals ** 2) / (observations-2)\
            * (1/observations + (np.mean(X) ** 2) / np.sum((X - np.mean(X)) ** 2)))

        print('======= Linear Regression Summary =======\n\n' + 
              f'{'Number of observations:':30}{observations:10}\n'+
              f'{'Dependent variable:':30}{'y':>10}\n'+
              f'{'Independent variable:':30}{'x':>10}\n'+
              f'-----------------------------------------\n\n'+
              f'Model Fit:\n' + 
              f'{'R-squared:':30}{r_squared:10.3f}\n' + 
              f'{'Mean Square Error (MSE):':30}{mse:10.3f}\n'+
              f'{'Root Mean Square Error:':30}{root_mse:10.3f}\n'+
              f'-----------------------------------------\n\n'+
              f'Coefficients:\n' + 
              f'{'Intercept (c):':30}{self.c:10.3f}\n' + 
              f'{'Standard Error:':30}{SE_c:10.3f}\n\n' +
              f'{'Slope (m):':30}{self.m:10.3f}\n' +
              f'{'Standard Error:':30}{SE_m:10.3f}'
              )