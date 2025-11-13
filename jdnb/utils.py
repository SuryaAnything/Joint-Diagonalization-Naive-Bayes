import numpy as np
from numpy.linalg import qr

def offdiag(M):
    """Returns the off-diagonal elements of a matrix M (A - diag(A))."""
    return M - np.diag(np.diag(M))

def orthonormalize(W):
    """Orthonormalizes the columns of W using QR decomposition."""
    Q, _ = qr(W)
    return Q

def class_stats(X, y):
    """
    Calculates class-wise means, covariances, and priors.
    Adds a small ridge to covariances for numerical stability.
    """
    classes = np.unique(y)
    mus, Sigmas, priors = {}, {}, {}
    n_samples, n_features = X.shape
    
    for c in classes:
        Xc = X[y == c]
        n_c = Xc.shape[0]
        
        mus[c] = Xc.mean(axis=0)
        
        # Calculate empirical covariance
        S = np.cov(Xc, rowvar=False, bias=True)
        
        # Add regularization (ridge) as described in your code
        # 1e-2 of trace / num_features as regularization
        tau = 1e-2 * np.trace(S) / n_features
        if tau < 1e-8: # Handle zero-variance features
            tau = 1e-8
            
        Sigmas[c] = (1 - 1e-2) * S + 1e-2 * tau * np.eye(n_features)
        priors[c] = n_c / n_samples
        
    return classes, mus, Sigmas, priors

def jd_loss_and_grad(W, Sigmas):
    """
    Calculates the Joint-Diagonalization loss and its gradient.
    
    Loss = sum_c || offdiag(W.T @ Sigma_c @ W) ||_F^2
    """
    loss = 0.0
    G = np.zeros_like(W)
    
    for S in Sigmas.values():
        A = W.T @ S @ W
        O = offdiag(A)
        loss += np.sum(O * O)
        G += 4 * (S @ W @ O)
        
    return loss, G

def offdiag_energy(Sigmas):
    """Calculates the total off-diagonal energy for a set of covariances."""
    return sum(np.sum(offdiag(S)**2) for S in Sigmas.values())

def kl_divergence_cov(Sigma1, Sigma2):
    """KL divergence between two diagonal covariance approximations."""
    d1 = np.diag(Sigma1)
    d2 = np.diag(Sigma2)
    # Epsilon for numerical stability
    d1 = np.clip(d1, 1e-10, None)
    d2 = np.clip(d2, 1e-10, None)
    return np.sum(d1 * np.log(d1 / d2) - d1 + d2)