import autograd.numpy as anp


# def lorentz_ps(z_1, z_2):
#     return -z_1[:, 0] * z_2[:, 0] + anp.sum(z_1[:, 1:] * z_2[:, 1:], axis=1)


def lorentz_ps_simple(z1, z2):
    # Suppose to be faster
    return -z1[0] * z2[0] + anp.dot(z1[1:], z2[1:])


def lorentz_ps(z1, z2):
    # Ensure both inputs are 2D (n, d)
    if z1.ndim == 1:
        z1 = z1[None, :]
    if z2.ndim == 1:
        z2 = z2[None, :]

    return anp.einsum("ij,ij->i", z1, z2) - 2 * z1[:, 0] * z2[:, 0]


def parallel_transport(nu, mu, v):
    n_samples = v.shape[0]
    alpha = -lorentz_ps_simple(nu, mu)
    lhs = anp.vstack([mu - alpha * nu] * n_samples)
    return v + lorentz_ps(lhs, v)[:, None] * (nu + mu) / (alpha + 1)


def exp_map(u, mu):
    norm_u = anp.sqrt(lorentz_ps(u, u))
    return (
        anp.cosh(norm_u)[:, None] * mu + anp.sinh(norm_u)[:, None] * u / norm_u[:, None]
    )


def log_map(z, mu):
    alpha = -lorentz_ps(anp.array([mu]), z)
    coeff = anp.arccosh(alpha) / anp.sqrt(alpha**2 - 1)
    return coeff[:, None] * (z - alpha[:, None] * mu)


def unvect_p0(v):
    zeros = anp.zeros((v.shape[0], 1))
    return anp.hstack((zeros, v))


def unvect_p(v, p):
    p0 = anp.zeros_like(p)
    p0[0] = 1
    return parallel_transport(p0, p, unvect_p0(v))


def vect_p0(v):
    return v[:, 1:]


def parallel_transport_to_p0(nu, v):
    # alpha = -<nu, p0> = nu[0]
    alpha = nu[0]
    e0 = anp.concatenate(([1.0], anp.zeros_like(nu[1:])))
    lhs = e0 - alpha * nu
    proj = lorentz_ps(lhs, v)
    return v + proj[:, None] * (nu + e0) / (alpha + 1)


def vect_p(v, p):
    p0 = anp.zeros_like(p)
    p0[0] = 1
    # print("p0:", p0)
    # print("v:", v)
    # print("parallel transport:", parallel_transport(p0, p, v))
    # print("p", p)
    return vect_p0(parallel_transport_to_p0(p, v))
    # return vect_p0(parallel_transport(p, p0, v))
