import numpy as np
import matplotlib.pyplot as plt

from pyriemann.utils.base import sqrtm

import pymanopt
from pymanopt.manifolds import SymmetricPositiveDefinite, Euclidean, Product
from pymanopt.optimizers import SteepestDescent
import autograd.numpy as anp
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import log_map_riemann
from sklearn.covariance import ShrunkCovariance


def multitransp(A):
    """Vectorized matrix transpose.

    ``A`` is assumed to be an array containing ``M`` matrices, each of which
    has dimension ``N x P``.
    That is, ``A`` is an ``M x N x P`` array. Multitransp then returns an array
    containing the ``M`` matrix transposes of the matrices in ``A``, each of
    which will be ``P x N``.
    """
    if A.ndim == 2:
        return A.T
    return anp.transpose(A, (0, 2, 1))


def multihconj(A):
    """Vectorized matrix conjugate transpose."""
    return anp.conjugate(multitransp(A))


def multilogm(A, *, positive_definite=False):
    """Vectorized matrix logarithm."""

    w, v = anp.linalg.eigh(A)
    w = anp.expand_dims(anp.log(w), axis=-1)
    logmA = v @ (w * multihconj(v))
    # if anp.isrealobj(A):
    return anp.real(logmA)
    # return logmA


def log_anp(point_a, point_b, full=True):
    c = anp.linalg.cholesky(point_a)
    c_inv = anp.linalg.inv(c)
    logm = multilogm(
        c_inv @ point_b @ multihconj(c_inv),
        positive_definite=True,
    )
    if full:
        return c @ logm @ multihconj(c)
    else:
        return logm


def _matrix_operator_anp(C, operator):
    # print(C.shape)
    """Matrix function."""
    # if not isinstance(C, anp.ndarray) or C.ndim < 2:
    # raise ValueError("Input must be at least a 2D ndarray")
    if anp.isinf(C).any() or anp.isnan(C).any():
        raise ValueError(
            "Matrices must be positive definite. Add "
            "regularization to avoid this error."
        )
    eigvals, eigvecs = anp.linalg.eigh(C)
    eigvals = operator(eigvals)
    if C.ndim >= 3:
        eigvals = anp.expand_dims(eigvals, -2)
    D = (eigvecs * eigvals) @ anp.swapaxes(anp.conj(eigvecs), -2, -1)
    return D


def invsqrtm_anp(C):
    def isqrt(x):
        return 1.0 / anp.sqrt(x)

    return _matrix_operator_anp(C, isqrt)


def vect_In_anp(X):
    """Vectorization of the upper triangular part of a set of matrices at the identity.
    This function is compatible with autograd.
    """
    m, n = X.shape[:2]
    idx = anp.triu_indices(n, k=1)
    diag_elements = X[anp.arange(m)[:, None], anp.eye(n, dtype=bool)]
    other_elements = anp.sqrt(2) * X[:, idx[0], idx[1]]
    return anp.hstack((diag_elements, other_elements))


def vect_sigma_anp(X, Sigma):
    """Vectorization of the upper triangular part of a set of matrices at Sigma.
    This function is compatible with autograd.
    """
    sigma_invsqrt = invsqrtm_anp(Sigma)
    return vect_In_anp(sigma_invsqrt @ X @ sigma_invsqrt)


def create_cost_estimation(sample, manifold, diagonal):
    """Create the cost function for the estimation of the parameters of a wrapped Gaussian distribution.
    We want to maximize the log-likelihood of the sample under the model of a wrapped Gaussian distribution.
    As our framework only does minimization, we return the negative log-likelihood.

    Args:
        sample (N, dim, dim): The sample of N SPD matrices of size dim x dim.
        manifold (pymanopt.manifold): The manifold on which the optimization is performed.
        diagonal (bool): If the covariance matrix is diagonal or not.

    Returns:
        The cost function to minimize.
    """
    d = sample.shape[1]  # Dimension of the SPD matrices
    dim_TS = int(d * (d + 1) / 2)  # Dimension of the tangent space

    @pymanopt.function.autograd(manifold)
    def cost(p, mu, Sigma):
        if diagonal:
            # If the covariance matrix is diagonal, as we only store its diagonal, we recover the full matrix
            Sigma = anp.diag(Sigma**2)

        # We compute Vect_p(Log_p(Sample))
        p_invsqrt = invsqrtm_anp(p)
        logm_sample = multilogm(p_invsqrt @ sample @ p_invsqrt)
        log_vect = vect_In_anp(logm_sample)

        # Density of the multivariate normal on the tangent space
        expo = anp.exp(
            -anp.sum(
                ((log_vect - mu) @ anp.linalg.inv(Sigma)) * (log_vect - mu), axis=1
            )
            / 2.0
        )
        const_gauss = 1.0 / anp.sqrt(
            anp.power(2.0 * anp.pi, dim_TS) * anp.linalg.det(Sigma)
        )
        h_ = const_gauss * expo

        # The Jacobian of the exponential
        eigenvalues, _ = anp.linalg.eigh(logm_sample)

        # Use broadcasting to compute all eigenvalue differences
        diff_matrix = eigenvalues[:, :, anp.newaxis] - eigenvalues[:, anp.newaxis, :]

        # Take only the differences for i < j
        triu_indices = anp.triu_indices(d, k=1)
        diff_triu = diff_matrix[:, triu_indices[0], triu_indices[1]]

        # Compute sinh((lambda_i - lambda_j) / 2) / (lambda_i - lambda_j)
        sinh_term = anp.sinh(diff_triu / 2) / diff_triu

        # Calculate the product of all terms for each matrix
        # Without the constant factor that will vanish in the log
        J_ = anp.prod(sinh_term, axis=1)

        # return the negative log-likelihood
        return -anp.sum(anp.log(h_ / anp.abs(J_)))

    return cost


def estimation_wrapped_gaussian(
    X,
    verbosity=1,
    max_iterations=10000,
    max_time=1000,
    initial_point=None,
    optimizer=SteepestDescent,
    minimal=True,
    diagonal=False,
):
    """Estimation of the parameters of a wrapped Gaussian distribution based on a sample X of SPD matrices.

    Args:
        X (N, dim, dim): A sample of N SPD matrices of size dim x dim.
        verbosity (int, optional): The level of verbosity of the optimizer. Defaults to 1.
        max_iterations (int, optional): The maximum number of iteration of the optimization. Defaults to 10000.
        max_time (int, optional): The maximum time of the optimization in seconds. Defaults to 60.
        initial_point (tuple, optional): The inital parameters from which the optimization will start. Defaults to None.
        optimizer (pymanopt.optimizer, optional):The optimizer chosen for the optimization. Defaults to SteepestDescent.
        minimal (bool, optional): If the parameters returned are minimal. Defaults to True.
        diagonal (bool, optional): If the covariance matrix Sigma to estimate is diagonal. Defaults to False.

    Returns:
        pymanopt.OptimizerResult: The result of the optimization.
    """
    d = X.shape[1]  # Dimension of the SPD matrices
    dim_TS = int(d * (d + 1) / 2)  # Dimension of the tangent space
    nu = np.concatenate((np.ones(d), np.zeros(int(dim_TS - d))))

    if diagonal:
        # If the covariance matrix is diagonal, we only need to estimate its diagonal
        manifold_prod = Product(
            [
                SymmetricPositiveDefinite(n=d),
                Euclidean(dim_TS),
                Euclidean(dim_TS),
            ]
        )
    else:
        # If the covariance matrix is full, we need to estimate all its entries
        manifold_prod = Product(
            [
                SymmetricPositiveDefinite(n=d),
                Euclidean(dim_TS),
                SymmetricPositiveDefinite(dim_TS),
            ]
        )

    if initial_point is None:
        # If no initial point is provided, we use the Riemannian mean as the initial p
        p_init = mean_riemann(X)

        # For mu and Sigma, we send the points to the tangent space at p_init and compute the mean and covariance there.
        X_TS_vect = vect_sigma_anp(log_map_riemann(X, p_init, C12=True), p_init)
        mu_init = anp.mean(X_TS_vect, axis=0)
        Sigma_init = ShrunkCovariance().fit(X_TS_vect).covariance_

        # We compute the minimal representation of the inital parameters
        s = np.sum(mu_init[:d]) / d
        p_init = np.exp(s) * p_init
        mu_init = mu_init - s * nu

        if diagonal:
            # If the covariance matrix is diagonal, we keep only its diagonal
            Sigma_init = anp.sqrt(anp.diag(Sigma_init))

        initial_point = [p_init, mu_init, Sigma_init]

    # Create the cost function
    cost = create_cost_estimation(X, manifold_prod, diagonal)
    problem = pymanopt.Problem(manifold_prod, cost)
    optimizer = optimizer(
        verbosity=verbosity,
        log_verbosity=2,
        max_iterations=max_iterations,
        max_time=max_time,
    )
    # Run the optimization
    res = optimizer.run(problem, initial_point=initial_point)

    if minimal:
        # We return the minimal representation of the parameters
        s = np.sum(res.point[1][:d]) / d
        res.point[0] = np.exp(s) * res.point[0]
        res.point[1] = res.point[1] - s * nu

    return res


def log_likelihood(sample, p, mu, Sigma):
    """Computes the negative log-likelihood of a sample of SPD matrices under the wrapped Gaussian WG(p, mu, Sigma).

    Args:
        sample (N, dim, dim): The sample of N SPD matrices of size dim x dim.
        p (dim, dim): The parameter p of the wrapped Gaussian distribution.
        mu (dim*(dim-1)/2): The parameter mu of the wrapped Gaussian distribution.
        Sigma (dim*(dim-1)/2, dim*(dim-1)/2): The parameter Sigma of the wrapped Gaussian distribution.

    Returns:
        n, float: The negative log-likelihood of the sample under the wrapped Gaussian distribution.
    """
    n = sample.shape[1]  # Dimension of the sample matrices
    d = int(n * (n + 1) / 2)  # Dimension of the tangent space
    p_invsqrt = invsqrtm_anp(p)
    logm_sample = multilogm(p_invsqrt @ sample @ p_invsqrt)
    log_vect = vect_In_anp(logm_sample)  # Vect_p(Log_p(Sample))

    # Density of the multivariate normal on the tangent space
    expo = anp.exp(
        -anp.sum(((log_vect - mu) @ anp.linalg.inv(Sigma)) * (log_vect - mu), axis=1)
        / 2.0
    )
    const_norm = 1.0 / anp.sqrt(anp.power(2.0 * anp.pi, d) * anp.linalg.det(Sigma))
    h_ = const_norm * expo

    # The Jacobian of the exponential
    eigenvalues, _ = anp.linalg.eigh(logm_sample)

    # Use broadcasting to compute all eigenvalue differences
    diff_matrix = eigenvalues[:, :, anp.newaxis] - eigenvalues[:, anp.newaxis, :]

    # Take only the differences for i < j
    triu_indices = anp.triu_indices(n, k=1)
    diff_triu = diff_matrix[:, triu_indices[0], triu_indices[1]]

    # Compute sinh((lambda_i - lambda_j) / 2) / (lambda_i - lambda_j)
    sinh_term = anp.sinh(diff_triu / 2) / diff_triu

    # Calculate the product of all terms for each matrix
    # Without the constant factor that will vanish in the log
    J_ = anp.prod(sinh_term, axis=1)

    # return the negative log-likelihood
    return -anp.sum(anp.log(h_ / anp.abs(J_)))


def create_cost_Ho_WDA(sample, manifold, labels, diagonal):
    """Create the cost function for the estimation of the parameters for the Ho_WDA.
    In this case, we learn one p and one mu per class, but the covariance matrix Sigma is shared among all classes.

    Args:
        sample (N, dim, dim): The sample of N SPD matrices of size dim x dim.
        manifold (pymanopt.manifold): The manifold on which the optimization is performed.
        labels (n, ): The labels of the sample.
        diagonal (bool): If the covariance matrix Sigma is diagonal or not.

    Returns:
        The cost function to minimize.
    """
    n_classes = len(np.unique(labels))
    all_labels = np.unique(labels)
    n = sample.shape[1]  # Dimension of the sample matrices
    d = int(n * (n + 1) / 2)  # Dimension of the tangent space

    @pymanopt.function.autograd(manifold)
    # def cost(p, mu, Sigma):
    def cost(p, mu, Sigma):
        mu = mu.reshape((n_classes, d))

        if diagonal:
            # If the covariance matrix is diagonal, as we only store its diagonal, we recover the full matrix
            Sigma = anp.diag(Sigma**2)

        # We compute the log-likelihood of each sample under its estimated WG and add them up.
        total_log_likelihood = 0
        for i in range(n_classes):
            X = sample[labels == all_labels[i]]
            total_log_likelihood += log_likelihood(X, p[i], mu[i], Sigma)
        return total_log_likelihood

    return cost


def estimation_Ho_WDA(
    X,
    labels,
    verbosity=2,
    max_iterations=10000,
    max_time=1000,
    initial_point=None,
    optimizer=SteepestDescent,
    minimal=True,
    diagonal=False,
):
    """Estimation of the parameters of a wrapped Gaussian distribution for the Ho_WDA.
    We learn one p and one mu per class, but the covariance matrix Sigma is shared among all classes.

    Args:
        X (N, dim, dim): A sample of N SPD matrices of size dim x dim.
        labels (n, ): The labels of the sample.
        verbosity (int, optional): The level of verbosity of the optimizer. Defaults to 1.
        max_iterations (int, optional): The maximum number of iteration of the optimization. Defaults to 10000.
        initial_point (tuple, optional): The inital parameters from which the optimization will start. Defaults to None.
        optimizer (pymanopt.optimizer, optional):The optimizer chosen for the optimization. Defaults to SteepestDescent.
        minimal (bool, optional): If the parameters returned are minimal. Defaults to True.
        diagonal (bool, optional): If the covariance matrix Sigma to estimate is diagonal. Defaults to False.

    Returns:
        pymanopt.OptimizerResult: The result of the optimization.
    """
    d = X.shape[1]
    dim_TS = int(d * (d + 1) / 2)
    n_classes = len(np.unique(labels))
    if diagonal:
        # If the covariance matrices are diagonal, we only need to estimate their diagonal
        manifold_prod = Product(
            [
                SymmetricPositiveDefinite(n=d, k=n_classes),
                Euclidean(dim_TS * n_classes),
                Euclidean(dim_TS),
            ]
        )
    else:
        # Otherwise, we need to estimate all the entries of the covariance matrices of each class
        manifold_prod = Product(
            [
                SymmetricPositiveDefinite(n=d, k=n_classes),
                Euclidean(dim_TS * n_classes),
                SymmetricPositiveDefinite(n=dim_TS),
            ]
        )

    if initial_point is None:
        # If no initial point is provided, we use the Riemannian mean as the initial p for each class
        # We also compute the mean of each class in the tangent space at the Riemannian mean.
        all_p_init = []
        all_mu_init = []

        for i in range(n_classes):
            p_init = mean_riemann(X[labels == labels[i]])
            all_p_init.append(p_init.tolist())

            X_TS_vect = vect_sigma_anp(
                log_map_riemann(X[labels == labels[i]], p_init, C12=True), p_init
            )
            mu_init = np.mean(X_TS_vect, axis=0)

            nu = np.concatenate((np.ones(d), np.zeros(int(dim_TS - d))))
            s = np.sum(mu_init[:d]) / d
            all_p_init.append((np.exp(s) * p_init).tolist())
            all_mu_init += list(mu_init - s * nu)

        # For the commun Sigma, we compute the mean of the tangent space at the Riemannian mean of all the samples.
        mean_all = mean_riemann(X)
        X_TS_vect = vect_sigma_anp(log_map_riemann(X, mean_all, C12=True), mean_all)
        Sigma_init = ShrunkCovariance().fit(X_TS_vect).covariance_
        if diagonal:
            # If the covariance matrix is diagonal, we keep only its diagonal
            Sigma_init = np.sqrt(np.diag(Sigma_init))

        initial_point = [
            np.array(all_p_init),
            np.array(all_mu_init),
            np.array(Sigma_init),
        ]

    cost = create_cost_Ho_WDA(X, manifold_prod, labels, diagonal)
    problem = pymanopt.Problem(manifold_prod, cost)
    optimizer = optimizer(
        verbosity=verbosity,
        log_verbosity=2,
        max_iterations=max_iterations,
        max_time=max_time,
    )
    res = optimizer.run(problem, initial_point=initial_point)
    res.point[1] = res.point[1].reshape((n_classes, dim_TS))
    if minimal:
        # We return the minimal representation of the parameters
        nu = np.concatenate((np.ones(d), np.zeros(int(dim_TS - d))))

        for i in range(n_classes):
            s = np.sum(res.point[1][i, :d]) / d
            res.point[0][i] = np.exp(s) * res.point[0][i]
            res.point[1][i] = res.point[1][i] - s * nu
    return res
