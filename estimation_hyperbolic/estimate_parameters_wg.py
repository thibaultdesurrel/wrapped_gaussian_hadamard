from autograd import numpy as anp
from hyperbolic_manifold import Hyperbolic
from pymanopt.manifolds import Product, Euclidean, SymmetricPositiveDefinite
from base import vect_p, lorentz_ps
import pymanopt
from pymanopt.optimizers import SteepestDescent


def log_map(z, mu):
    alpha = -lorentz_ps(mu, z)
    if anp.any(alpha**2 <= 1):
        raise ValueError(
            "The log map is not defined for points outside the hyperbolic space."
        )
    coeff = anp.arccosh(alpha) / anp.sqrt(alpha**2 - 1)
    return coeff[:, None] * (z - alpha[:, None] * mu)


def jacobian_exp(u, p, manifold):
    """
    Compute the Jacobian of the exponential map at point x with respect to p.
    """
    d = u.shape[1]  # Dimension of the hyperbolic space

    # norm_u = anp.array([manifold.norm(p, u[i]) for i in range(u.shape[0])])
    ps_u = lorentz_ps(u, u)
    if anp.any(ps_u <= 0):
        raise ValueError(
            "The norm of u is not defined for points outside the hyperbolic space."
            + str(ps_u)
        )
    norm_u = anp.sqrt(ps_u)

    return (anp.sinh(norm_u) / norm_u) ** (d - 1)


def compute_log_likelihood(x, p, mu, Sigma, manifold):
    """
    Compute the (negative) log-likelihood of points under a wrapped Gaussian on the hyperbolic space H^d.

    Parameters
    ----------
    x : (N, d): The sample of N samples from the hyperbolic space H^d.
    p : anp.ndarray
        Base point of the wrapped Gaussian on the hyperbolic manifold.
        Shape (d,).
    mu : anp.ndarray
        Mean vector of the Gaussian in the tangent space at p. Shape (D,), where D
        is the tangent-space dimension.
    Sigma : anp.ndarray
        Covariance matrix of the Gaussian in the tangent space at p. Shape (D, D).
    manifold : (pymanopt.manifold)
        The manifold on which the optimization is performed.
    Returns
    -------
    float
        The scalar negative log-likelihood summed over all samples:
        -sum( log p(x_i | p, mu, Sigma) )
    """

    log_p_Xi = log_map(x, anp.tile(p, (x.shape[0], 1)))  # FASTER

    ps = lorentz_ps(log_p_Xi, anp.tile(p, (x.shape[0], 1)))

    if not anp.allclose(ps, 0):
        # Project log_p_Xi onto the tangent space
        log_p_Xi_new = anp.array(
            [manifold.projection(p, log_p_Xi[i]) for i in range(log_p_Xi.shape[0])]
        )
        print(
            "Needed to project log_p_Xi onto the tangent space.",
        )
        log_p_Xi = log_p_Xi_new

    vect_p_log_p_Xi = vect_p(log_p_Xi, p)

    diff = vect_p_log_p_Xi - mu  # shape (N, d)
    Sigma_inv = anp.linalg.inv(Sigma)

    mahalanobis = anp.einsum("ni,ij,nj->n", diff, Sigma_inv, diff)  # shape (N,)

    denominator = anp.abs(jacobian_exp(log_p_Xi, p, manifold))

    # return the negative log-likelihood
    return -anp.sum(
        -0.5 * mahalanobis - anp.log(denominator) - anp.log(anp.linalg.det(Sigma)) / 2
    )


def create_cost_full(sample, manifold_product, manifold_hyperbolic):
    """Create the cost function for the estimation of the parameters of a wrapped Gaussian distribution.
    We want to maximize the log-likelihood of the sample under the model of a wrapped Gaussian distribution.
    As our framework only does minimization, we return the negative log-likelihood.

    Args:
        sample (N, d): The sample of N samples from the hyperbolic space H^d.
        manifold_product (pymanopt.manifold): The manifold on which the optimization is performed.
        manifold_hyperbolic (pymanopt.manifold): The hyperbolic space H^d.

    Returns:
        The cost function to minimize.
    """

    @pymanopt.function.autograd(manifold_product)
    def cost_function(p, mu, Sigma):
        log_likelihood = compute_log_likelihood(
            sample, p, mu, Sigma, manifold_hyperbolic
        )
        return log_likelihood

    return cost_function


def estimation_param(
    samples,
    verbosity=2,
    max_iterations=100,
    max_time=60,
    initial_point=None,
    optimizer=SteepestDescent,
):
    """Estimation of the parameters of a wrapped Gaussian distribution based on a sample of points on the hyperbolic space H^d.

    Args:
        X (N, d): The sample of N samples from the hyperbolic space H^d.
        verbosity (int, optional): The level of verbosity of the optimizer. Defaults to 2.
        max_iterations (int, optional): The maximum number of iteration of the optimization. Defaults to 100.
        max_time (int, optional): The maximum time of the optimization in seconds. Defaults to 60.
        initial_point (tuple, optional): The inital parameters from which the optimization will start. Defaults to None.
        optimizer (pymanopt.optimizer, optional):The optimizer chosen for the optimization. Defaults to SteepestDescent.

    Returns:
        pymanopt.OptimizerResult: The result of the optimization.
    """
    d = samples.shape[1] - 1  # Dimension of the hyperbolic space

    manifold_product = Product(
        [Hyperbolic(d), Euclidean(d), SymmetricPositiveDefinite(d)]
    )
    manifold_hyperblic = Hyperbolic(d)
    cost_function = create_cost_full(samples, manifold_product, manifold_hyperblic)
    problem = pymanopt.Problem(manifold_product, cost_function)
    optimizer = optimizer(
        verbosity=verbosity,
        max_iterations=max_iterations,
        max_time=max_time,
        log_verbosity=2,
    )
    solution = optimizer.run(problem=problem, initial_point=initial_point)

    return solution
