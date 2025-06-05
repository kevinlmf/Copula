import numpy as np
from scipy.stats import kendalltau

def compute_kendalls_tau(Y):
    d = Y.shape[1]
    tau = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            t, _ = kendalltau(Y[:, i], Y[:, j])
            tau[i, j] = tau[j, i] = t
    return tau
