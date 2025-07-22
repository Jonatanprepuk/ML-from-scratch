import numpy as np

def euclidean(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = X.reshape(-1, X.shape[-1])
    y = y.reshape(-1, y.shape[-1])

    diffs = X[:, None, :] - y[None, :, :]

    return np.linalg.norm(diffs, axis=2)

def manhattan(X: np.ndarray, y: np.ndarray) -> np.ndarray:

    X = X.reshape(-1, X.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    return np.sum(np.abs(X[:, None, :] - y[None, :, :]), axis=2)