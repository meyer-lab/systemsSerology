"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, coupled_matrix_tensor_3d_factorization


def R2X(reconstructed, original):
    """ Calculates R2X of two tensors. Missing values should be indicated as nan. """
    return 1.0 - np.nanvar(reconstructed - original) / np.nanvar(original)


def perform_decomposition(tensor, r, weightFactor=2):
    """ Perform PARAFAC decomposition. """
    tensor = np.copy(tensor)
    mask = np.isfinite(tensor).astype(int)
    tensor[mask == 0] = 0.0

    weights, factors = parafac(tensor, r, mask=mask, orthogonalise=True, n_iter_max=4000, normalize_factors=True, init="random")
    assert np.all(np.isfinite(factors[0]))
    assert np.all(np.isfinite(weights))

    factors[weightFactor] *= weights[np.newaxis, :]  # Put weighting in designated factor
    
    print("R2X: " + str(find_R2X(tensor, factors)))

    return factors


def perform_CMTF(tensor, matrix, r):
    """ Perform CMTF decomposition. """
    tensor = np.copy(tensor)
    mask = np.isfinite(tensor).astype(int)
    tensor[mask == 0] = 0.0

    matrix = np.copy(matrix)
    mask_matrix = np.isfinite(matrix).astype(int)
    matrix[mask_matrix == 0] = 0.0

    tensorFac, matrixFac = coupled_matrix_tensor_3d_factorization(tensor, matrix, r, mask_3d=mask, mask_matrix=mask_matrix, init="random")

    return tensorFac, matrixFac


def find_R2X(values, factors):
    """Compute R2X from CP. Note that the inputs values and factors are in numpy."""
    return R2X(tl.kruskal_to_tensor((np.ones(factors[0].shape[1]), factors)), values)


def impute(tensor, r):
    """ Decompose and then reconstruct tensor without missingness. """
    factors = perform_decomposition(tensor, r)
    recon = tl.kruskal_to_tensor((np.ones(factors[0].shape[1]), factors))

    return recon
