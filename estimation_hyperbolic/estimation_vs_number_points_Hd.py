import numpy as np

from sample_hyperbolic_wg import sample_hyperbolic_wg
from estimate_parameters_wg import estimation_param

from hyperbolic_manifold import Hyperbolic
from pyriemann.utils.distance import distance_riemann
from pyriemann.datasets import generate_random_spd_matrix
from pymanopt.optimizers import ConjugateGradient


import pandas as pd

np.random.seed(42)

all_dim = [3, 15, 45]
all_num = np.linspace(50, 1000, 20)
rep = 5

results = []
for d in all_dim:
    print("=============================================")
    print(f"================= Dim: {d} =================")
    print("=============================================")

    manifold = Hyperbolic(d)
    for r in range(rep):
        print(f"================= Rep: {r} =================")
        p = manifold.random_point()
        mu = np.random.uniform(low=0, high=0.2, size=d)
        Sigma = generate_random_spd_matrix(n_dim=d, mat_mean=0.01, mat_std=0.02) / 5
        samples = sample_hyperbolic_wg(int(all_num[-1]), p, mu, Sigma)

        for i in range(all_num.shape[-1]):
            num = int(all_num[i])
            print("Number of points: ", num)
            X = samples[:num]
            res_optim = estimation_param(
                X,
                verbosity=1,
                max_iterations=20000,
                max_time=6000,
                optimizer=ConjugateGradient,
            )

            results.append(
                {
                    "algo": "MLE",
                    "rep": r,
                    "Dimension": d,
                    "num": num,
                    "error_p": manifold.dist(p, res_optim.point[0]),
                    "error_mu": np.linalg.norm(mu - res_optim.point[1]),
                    "error_sigma": distance_riemann(Sigma, res_optim.point[2]),
                }
            )

results_dataframe = pd.DataFrame(results)
results_dataframe.to_pickle("results_vs_number_points_big_dim.pkl")
