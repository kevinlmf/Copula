import numpy as np
from copulas.multivariate import GaussianMultivariate

def fit_parametric_gaussian_copula(Y):
    model = GaussianMultivariate()
    model.fit(Y)
    Y_sim = model.sample(len(Y))
    return model, model.to_dict(), Y_sim
