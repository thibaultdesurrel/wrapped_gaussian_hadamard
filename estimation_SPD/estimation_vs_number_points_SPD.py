import numpy as np
import matplotlib.pyplot as plt


from sampling_wrapped_gaussian_SPD import sample_wrapped_gaussian
from estimation_autograd_SPD import estimation_wrapped_gaussian, vect_sigma_anp

from pyriemann.utils.distance import distance_riemann
from pyriemann.datasets import generate_random_spd_matrix
from pymanopt.optimizers import ConjugateGradient

import pandas as pd
from tqdm import tqdm

np.random.seed(42)

all_dim = np.array([2, 5, 10])
all_num = np.linspace(100, 10000, 20)
rep = 5

results = []

for d in tqdm(all_dim):
    dim_TS = int(d * (d + 1) / 2)
    for r in range(rep):
        print(f"================= Rep: {r} =================")
        p = generate_random_spd_matrix(n_dim=int(d), mat_mean=0.1, mat_std=1)
        mu = np.random.uniform(low=0, high=0.1, size=dim_TS)
        Sigma = (
            generate_random_spd_matrix(n_dim=dim_TS, mat_mean=0.01, mat_std=0.02) / 5
        )

        s = np.sum(mu[:d]) / d
        p_mininal = np.exp(s) * p
        mu_minimal = mu - s * np.concatenate((np.ones(d), np.zeros(int(dim_TS - d))))

        all_X = sample_wrapped_gaussian(int(all_num[-1]), p, mu, Sigma)

        for i in range(all_num.shape[-1]):
            num = int(all_num[i])
            print("Number of points: ", num)
            X = all_X[:num]
            res_optim = estimation_wrapped_gaussian(
                X,
                verbosity=1,
                max_iterations=20000,
                optimizer=ConjugateGradient,
                minimal=True,
            )

            results.append(
                {
                    "algo": "MLE",
                    "rep": r,
                    "Dimension": d,
                    "num": num,
                    "error_p": distance_riemann(p_mininal, res_optim.point[0]),
                    "error_mu": np.linalg.norm(mu_minimal - res_optim.point[1]),
                    "error_sigma": distance_riemann(Sigma, res_optim.point[2]),
                }
            )

results_dataframe = pd.DataFrame(results)
results_dataframe.to_pickle("results_vs_number_points.pkl")
