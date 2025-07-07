import numpy as np

def sine_data(*, start:float=0, end:float=2*np.pi, 
              samples:int=100,noise_scalar:float=0, amplitude:float=1, 
              frequency:float=1) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a 1D sine wave dataset.

    Parameters
    ----------
    start : float, optional
        Starting value for x (default is 0).
    end : float, optional
        End value for x (default is 2 * pi).
    samples : int, optional
        Number of samples to generate (default is 100).
    noise_scalar : float, optional
        Standard deviation of the Gaussian noise added to the sine values (default is 0 (no noise)).
    amplitude : float, optional
        Amplitude of the sine wave (default is 1).
    frequency : float, optional
        Frequency multiplier for the sine wave (default is 1).
        
    Returns
    -------
    x : np.ndarray, shape (samples, 1)
        The input x values.
    y : np.ndarray, shape (samples, 1)
        The output y values (amplitude * sin(frequency * x) + noise).
    """
    x = np.linspace(start, end, samples).reshape(-1,1)
    y = amplitude * np.sin(frequency * x) + np.random.randn(*x.shape) * noise_scalar
    return x, y
