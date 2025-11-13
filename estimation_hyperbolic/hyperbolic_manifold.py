import autograd.numpy as np

from pymanopt.manifolds.manifold import Manifold


def _intrinsic_to_extrinsic_coordinates(point):
    """Convert intrinsic to extrinsic coordinates.

    Convert the parameterization of a point in hyperbolic space
    from its intrinsic coordinates, to its extrinsic coordinates
    in Minkowski space.

    From geomstats.geometry.hyperbolic.Hyperbolic.

    Parameters
    ----------
    point : array-like, shape=[..., dim]
        Point in hyperbolic space in intrinsic coordinates.

    Returns
    -------
    point_extrinsic : array-like, shape=[..., dim + 1]
        Point in hyperbolic space in extrinsic coordinates.


    """
    coord_0 = np.sqrt(1.0 + np.sum(point**2, axis=-1))
    return np.concatenate([coord_0[..., None], point], axis=-1)


class Hyperbolic(Manifold):
    r"""The hyperbloid manifold.

    Args:
        n: The dimension of the hyperbolic manifold.
    """

    def __init__(self, n: int):
        self._n = n

        if n < 1:
            raise ValueError(f"Need n >= 1. Value given was n = {n}")

        name = f"Hyperbolic space H^{n}"

        dimension = n
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return self.dim / 8

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return -tangent_vector_a[0] * tangent_vector_b[0] + np.dot(
            tangent_vector_a[1:], tangent_vector_b[1:]
        )

    def projection(self, point, vector):
        """
        Projects vector in the ambient space on the tangent space.
        Formula from https://juliamanifolds.github.io/Manifolds.jl/v0.5/manifolds/hyperbolic.html#ManifoldsBase.project-Tuple{Hyperbolic,Any,Any}
        """

        proj = (
            -self.inner_product(point, point, point) * vector
            + self.inner_product(point, point, vector) * point
        )
        return proj

    to_tangent_space = projection

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def random_point(self):
        samples = 2.0 * (np.random.rand(self.dim) - 0.5)
        return _intrinsic_to_extrinsic_coordinates(samples)

    def random_tangent_vector(self, point):
        vector = np.random.normal(size=point.shape)
        return self.projection(point, vector)

    def zero_vector(self, point):
        return np.zeros_like(point)

    def dist(self, point_a, point_b):
        return np.arccosh(-self.inner_product(point_a, point_a, point_b))

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        """Convert a Euclidean gradient to a Riemannian gradient on the hyperbolic manifold.

        Code inspired from the Matlab version of manopt: https://github.com/NicolasBoumal/manopt/blob/master/manopt/manifolds/hyperbolic/hyperbolicfactory.m#L196
        It comes from the paper: Gradient descent in hyperbolic space, Wilson & Leimeister, 2018 https://arxiv.org/pdf/1805.08207
        """
        # Flip the sign of the first coordinate of the gradient (time-like direction)
        modified_egrad = np.copy(euclidean_gradient)
        modified_egrad[0] = -modified_egrad[0]

        # Project onto the tangent space
        rgrad = self.projection(point, modified_egrad)

        return rgrad

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        # Flip the sign of the first coordinate (time-like direction)
        modified_egrad = np.copy(euclidean_gradient)
        modified_egrad[0] = -modified_egrad[0]

        modified_ehess = np.copy(euclidean_hessian)
        modified_ehess[0] = -modified_ehess[0]

        # Compute Minkowski inner product between point and gradient
        inner = self.inner_product(point, point, modified_egrad)

        # Apply correction and project
        rhess = self.projection(point, modified_ehess + tangent_vector * inner)

        return rhess

    def exp(self, point, tangent_vector):
        norm_u = self.norm(point, tangent_vector)
        if np.allclose(norm_u, 0):
            # print("Warning: norm_u is zero in exp function, returning point")
            return point
        res = np.cosh(norm_u) * point + np.sinh(norm_u) * tangent_vector / norm_u

        if not np.allclose(self.inner_product(res, res, res), -1):
            # raise ValueError(
            #    "Exponential map output is not on the hyperbolic manifold."
            #    + str(self.inner_product(res, res, res))
            # )
            print("Exponential map output is not on the hyperbolic manifold.")
            # res = -res / self.inner_product(res, res, res)
        return res

    retraction = exp

    def log(self, point_a, point_b):
        """
        Computes the logarithmic map (log map) from point_a to point_b on the hyperbolic manifold.
        The log map returns the tangent vector at point_a that points towards point_b, representing
        the shortest path (geodesic) between the two points in the hyperbolic space.
        Parameters
        ----------
        point_a : np.ndarray
            The base point on the hyperbolic manifold (starting point of the log map).
        point_b : np.ndarray
            The target point on the hyperbolic manifold.
        Returns
        -------
        np.ndarray
            The tangent vector at point_a pointing towards point_b.
        Notes
        -----
        This implementation assumes the use of the Lorentz model of hyperbolic geometry.
        """

        alpha = -self.inner_product(point_a, point_a, point_b)
        coeff = np.arccosh(alpha) / np.sqrt(alpha**2 - 1)
        return coeff * (point_b - alpha * point_a)

    def pair_mean(self, point_a, point_b):
        return self.exp(point_a, 0.5 * self.log(point_a, point_b))

    def transport(self, point_a, point_b, tangent_vector_a):
        alpha = -self.inner_product(point_a, point_a, point_b)
        lhs = point_b - alpha * point_a
        return tangent_vector_a + self.inner_product(point_a, lhs, tangent_vector_a) * (
            point_a + point_b
        ) / (alpha + 1)
