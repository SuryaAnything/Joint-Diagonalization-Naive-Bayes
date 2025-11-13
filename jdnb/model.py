import numpy as np
from numpy.linalg import qr
from .utils import class_stats, jd_loss_and_grad, orthonormalize

class JDNB:
    """
    Joint-Diagonalization Naive Bayes (JD-NB) Classifier.

    This classifier learns an orthonormal rotation W to jointly diagonalize
    the class covariance matrices before fitting a standard Gaussian Naive Bayes
    in the rotated feature space.
    """
    def __init__(self, lr=1e-2, iters=400, random_state=0):
        self.lr = lr
        self.iters = iters
        self.random_state = random_state
        self.W = None
        self.classes_ = None
        self.mus_ = None
        self.vars_ = None
        self.priors_ = None

    def fit(self, X, y):
        """
        Fit the JD-NB model.
        
        1. Estimates class statistics (means, covariances).
        2. Learns the rotation matrix W via gradient descent.
        3. Transforms stats and stores them for Naive Bayes.
        """
        np.random.seed(self.random_state)
        classes, mus, Sigmas, priors = class_stats(X, y)
        d = X.shape[1]
        W = np.eye(d)

        print(f"Training JDNB ({d}-dim data)...")
        for t in range(self.iters):
            loss, G = jd_loss_and_grad(W, Sigmas)
            W -= self.lr * G
            W = orthonormalize(W)
            if t % 100 == 0 or t == self.iters - 1:
                print(f"[Iter {t:03d}] JD-Loss = {loss:.6f}")

        self.W = W
        Z = X @ W
        
        # Get stats in the *rotated* space
        _, mus_z, Sigmas_z, priors_z = class_stats(Z, y)
        
        self.classes_ = classes
        self.mus_ = {c: mus_z[c] for c in classes}
        self.vars_ = {c: np.clip(np.diag(Sigmas_z[c]), 1e-8, None) for c in classes}
        self.priors_ = priors_z
        return self

    def _log_gauss_diag(self, Z, mu, var):
        """Log-likelihood of a diagonal Gaussian."""
        # This is the core of Naive Bayes
        return -0.5 * (
            np.sum(np.log(2 * np.pi * var)) + np.sum(((Z - mu) ** 2) / var, axis=1)
        )

    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.W is None:
            raise RuntimeError("Model must be fitted before prediction.")
            
        # Rotate the input data into the new space
        Z = X @ self.W
        scores = []
        for c in self.classes_:
            mu = self.mus_[c]
            var = self.vars_[c]
            # Calculate log-posterior (up to a constant)
            s = np.log(self.priors_[c]) + self._log_gauss_diag(Z, mu, var)
            scores.append(s)
            
        scores = np.vstack(scores).T
        
        # Convert log-posteriors to probabilities (numerically stable softmax)
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        """Predict class labels."""
        # The predicted class is the one with the highest probability
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]