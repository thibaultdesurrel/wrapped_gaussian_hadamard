import numpy as np

from pyriemann.utils.tangentspace import exp_map_riemann
from pyriemann.utils.base import sqrtm


def unvect_In(Y):
    """Unvectorize an array of matrices of size (m, D) into an array of symmetric matrices of size (m, n, n).
    It is the implementation of the invese of the function vect_In.
    """
    m, D = Y.shape
    n = int((np.sqrt(1 + 8 * D) - 1) / 2)

    diag_elements = Y[:, :n]
    other_elements = Y[:, n:] / np.sqrt(2)

    # Initialize an empty array for the resulting matrices
    X = np.zeros((m, n, n))

    # Assign diagonal elements
    X[np.arange(m)[:, None], np.arange(n), np.arange(n)] = diag_elements

    # Get indices for the upper triangular part (excluding the diagonal)
    idx = np.triu_indices(n, k=1)

    # Assign upper triangular elements
    X[np.arange(m)[:, None, None], idx[0], idx[1]] = other_elements[:, None, :]

    # Since the matrix is symmetric, assign the transposed elements as well
    X[np.arange(m)[:, None, None], idx[1], idx[0]] = other_elements[:, None, :]

    return X


def unvect_sigma(X, Sigma):
    """Unvectorize at point Sigma an array of matrices of size (m, D) into an array of symmetric matrices of size (m, n, n)
    It is the implementation of the invese of the function vect_sigma.
    """
    sigma_sqrt = sqrtm(Sigma)
    return sigma_sqrt @ unvect_In(X) @ sigma_sqrt


def sample_wrapped_gaussian(num, p, mu, Gamma, verbosity=1):
    """Sample num points of the wrapped Gaussian with parameters X_bar, mu and Gamma.

    Args:
        num (int): Number of points to generate
        X_bar (ndarray of size (dim, dim)): point on the SPD manifold
        mu (ndarray of size (dim*(dim-1)/2)): mean of the multivariate gaussian on the tangent space of SPD matrices at X_bar
        Gamma (ndarray of size (dim*(dim-1)/2, dim*(dim-1)/2)): Covariance matrix of the multivariate gaussian on the tangent space of SPD matrices at X_bar

    Returns:
        ndarray of size (num, dim, dim): The sampled SPD matrices
    """
    t = np.random.multivariate_normal(size=num, mean=mu, cov=Gamma)

    return exp_map_riemann(unvect_sigma(t, p), p, Cm12=True)
