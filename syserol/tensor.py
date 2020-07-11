"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from .cmtf import coupled_matrix_tensor_3d_factorization


def perform_CMTF(tensorIn, matrixIn, r):
    """ Perform CMTF decomposition. """
    tensor = np.copy(tensorIn)
    mask = np.isfinite(tensor).astype(int)
    tensor[mask == 0] = 0.0

    matrix = np.copy(matrixIn)
    mask_matrix = np.isfinite(matrix).astype(int)
    matrix[mask_matrix == 0] = 0.0

    # Initialize by running PARAFAC on the 3D tensor
    kruskal = parafac(tensor, r, mask=mask, orthogonalise=True, normalize_factors=False, linesearch=True)
    tensor = tensor*mask + tl.kruskal_to_tensor(kruskal, mask=1 - mask)
    assert np.all(np.isfinite(tensor))

    # Now run CMTF
    tensorFac, matrixFac = coupled_matrix_tensor_3d_factorization(tensor, matrix, r, mask_3d=mask, mask_matrix=mask_matrix, init=kruskal)

    tensorErr = np.nanvar(tl.kruskal_to_tensor(tensorFac) - tensorIn)
    matrixErr = np.nanvar(tl.kruskal_to_tensor(matrixFac) - matrixIn)

    R2XX = 1.0 - (tensorErr + matrixErr) / (np.nanvar(tensorIn) + np.nanvar(matrixIn))

    print("CMTF R2X: " + str(R2XX))

    return tensorFac, matrixFac, R2XX
