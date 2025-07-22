import numpy as np

class GaussianNB:
    def __init__(self):
        self.class_stats = {}
        self.class_priors = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.classes = np.unique(y)
        
        for c in self.classes:
            X_c = X[y == c]
            self.class_stats[c] = (X_c.mean(axis=0), X_c.std(axis=0))
            self.class_priors[c] = np.log(X_c.shape[0] / X.shape[0])
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        results = []
        for x in X:
            class_logprobs = {}
            for c in self.classes:
                mu, sigma = self.class_stats[c]
                log_likelihood = np.sum(np.log(self._normal_dist(x, mu, sigma)))
                log_prior = self.class_priors[c]
                class_logprobs[c] = log_prior + log_likelihood
                
            results.append(max(class_logprobs, key=class_logprobs.get))
        return np.array(results)
    
    def evaluate(self, X, y) -> float:
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
    
    def _normal_dist(self, x:np.ndarray, mu:np.ndarray, sigma:np.ndarray) -> np.ndarray:
        sigma = np.where(sigma == 0, 1e-6, sigma)
        return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu)/sigma)**2)
