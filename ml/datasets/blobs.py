import numpy as np

def blobs_data(*, classes:int=1, low:float=0.0, high:float=10.0,
               samples_per_class:int=100, noise_scalar:float=1, 
               seed:int=None, test_split:float=0.0,
               return_centers:bool=False) ->(
                                                tuple[np.ndarray, np.ndarray] | 
                                                tuple[np.ndarray, np.ndarray, np.ndarray] | 
                                                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | 
                                                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                                            ):
    """
    Generates clustered blob data with guaranteed minimum distance between cluster centers.

    Parameters
    ----------
    classes : int
        Number of clusters/classes.
    low : float
        Minimum value for cluster center coordinates.
    high : float
        Maximum value for cluster center coordinates.
    samples_per_class : int
        Number of points per cluster.
    noise_scalar : float
        Standard deviation for cluster spread.
    seed : int or None, optional
        Seed for reproducibility.
    test_split : float, optional
        Fraction of the dataset to be used as test data (between 0 and 1). If 0, no split is done.
    return_centers : bool, optional
        If True, also returns the cluster center coordinates.

    Returns
    -------
    If test_split == 0 and return_centers == False:
        X : np.ndarray, shape (samples_total, 2)
            Generated points.
        y : np.ndarray, shape (samples_total,)
            Labels for each point (cluster index).

    If test_split == 0 and return_centers == True:
        X : np.ndarray, shape (samples_total, 2)
            Generated points.
        y : np.ndarray, shape (samples_total,)
            Labels for each point (cluster index).
        centers : np.ndarray, shape (classes, 2)
            The coordinates of the cluster centers.

    If test_split > 0 and return_centers == False:
        X_train : np.ndarray, shape (n_train_samples, 2)
            Training set points.
        y_train : np.ndarray, shape (n_train_samples,)
            Training set labels.
        X_test : np.ndarray, shape (n_test_samples, 2)
            Test set points.
        y_test : np.ndarray, shape (n_test_samples,)
            Test set labels.

    If test_split > 0 and return_centers == True:
        X_train : np.ndarray, shape (n_train_samples, 2)
            Training set points.
        y_train : np.ndarray, shape (n_train_samples,)
            Training set labels.
        X_test : np.ndarray, shape (n_test_samples, 2)
            Test set points.
        y_test : np.ndarray, shape (n_test_samples,)
            Test set labels.
        centers : np.ndarray, shape (classes, 2)
            The coordinates of the cluster centers.
    """
    if seed is not None:
        np.random.seed(seed)
        
    centers = []
    min_dist = 4 * noise_scalar
    
    if min_dist > (high - low):
        raise ValueError("min_dist is greater than the allowed range. Reduce noise_scalar or increase the range.")
    
    while len(centers) < classes:
        center = np.random.uniform(low, high, 2)
        if not centers:
            centers.append(center)
            continue
        dists = np.linalg.norm(np.array(centers) - center, axis=1)
        if np.all(dists > min_dist):
            centers.append(center)
        
    X = []
    y = []
    
    for i, c in enumerate(centers):
        points = np.random.randn(samples_per_class, 2) * noise_scalar + c
        X.append(points)
        y.append(np.full(samples_per_class, i))
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    if test_split:
        split = int(len(X) * (1 - test_split))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        if return_centers:
            return X_train, y_train, X_test, y_test, np.array(centers)
        
        return X_train, y_train, X_test, y_test
    
    if return_centers:
        return X, y, np.array(centers)
    
    return X, y