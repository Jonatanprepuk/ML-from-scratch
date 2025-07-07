import numpy as np

def linear_data(*, start:float=0, end:float=10, 
                samples:int=100, noise_scalar:float=0, slope:float=1,
                intercept:float=0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a 1D linear dataset.
    
    Parameters
    ----------
    start : float, optional 
        Starting value for x (default is 0).
    end : float, optional
        End value for x (default is 10).
    samples : int, optional
        Number of samples to generate (default is 100).
    noise_scalar : float, optional
        Standard deviation of the Gaussian noise added to the y values (default is 0 (no noise)).
    slope : float 
        Slope of the linear dataset (default is 1).
    intercept : float
        Intercept of the linear dataset (default is 0). 
        
    Returns
    -------
    x : np.ndarray, shape (samples, 1)
        The input x values.
    y : np.ndarray, shape (samples, 1)
        The output y values (slope * x + intercept + noise). 
    """
    x = np.linspace(start, end, samples).reshape(-1,1)
    y = slope * x + intercept + np.random.randn(*x.shape) * noise_scalar
    return x, y
