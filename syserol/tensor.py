"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.random import random_kruskal
from tensorly.decomposition import parafac


def R2X(reconstructed, original):
    """ Calculates R2X of two tensors. Missing values should be indicated as nan. """
    return 1.0 - np.nanvar(reconstructed - original) / np.nanvar(original)


def perform_decomposition(tensor, r, weightFactor=2, fixed=None):
    """ Perform PARAFAC decomposition. """
    tensor = np.copy(tensor)
    mask = np.isfinite(tensor).astype(int)
    tensor[mask == 0] = 0.0

    if fixed is not None:
        init = random_kruskal(tensor.shape, rank=r, normalise_factors=False)
        init.factors[0] = fixed[0]
        init.factors[1] = fixed[1]
        fixed_modes = [0, 1]
    else:
        init = "random"
        fixed_modes = []

    weights, factors = parafac(tensor, r, mask=mask, tol=1.0e-10, n_iter_max=6000, orthogonalise=True, normalize_factors=False, init=init, fixed_modes=fixed_modes)
    assert np.all(np.isfinite(factors[0]))
    assert np.all(np.isfinite(weights))

    factors[weightFactor] *= weights[np.newaxis, :]  # Put weighting in designated factor

    return factors


def find_R2X(values, factors):
    """Compute R2X from CP. Note that the inputs values and factors are in numpy."""
    return R2X(tl.kruskal_to_tensor((np.ones(factors[0].shape[1]), factors)), values)


def impute(tensor, r):
    """ Decompose and then reconstruct tensor without missingness. """
    factors = perform_decomposition(tensor, r)
    recon = tl.kruskal_to_tensor((np.ones(factors[0].shape[1]), factors))

    return recon
