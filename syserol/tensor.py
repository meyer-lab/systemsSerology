"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.kruskal_tensor import KruskalTensor
from tensorly.decomposition import parafac


def calcR2X(data, factor):
    """ Calculate R2X. """
    tensorErr = np.nanvar(tl.kruskal_to_tensor(factor) - data)
    return 1.0 - tensorErr / np.nanvar(data)


def cmtf(Y, mask_matrix, init):
    """ Calculate the glycosylation matrix components corresponding to the patient components from the tensor. """
    assert tl.is_tensor(Y)

    # initialize values
    A = init.factors[0]

    # alternating least squares
    for iteration in range(10 ** 4):
        V = np.linalg.lstsq(A, Y, rcond=-1)[0]

        # Perform masking
        Y = Y * mask_matrix + A @ V * (1 - mask_matrix)

        error_new = np.linalg.norm(Y - A @ V)
        if iteration > 2 and (np.absolute(error_old - error_new) <= 1e-12):
            break

        error_old = error_new

    return KruskalTensor((None, [A, np.transpose(V)]))


def perform_CMTF(tensorIn, matrixIn, r):
    """ Perform CMTF decomposition. """
    tensor = np.copy(tensorIn)
    mask = np.isfinite(tensor).astype(int)
    tensor[mask == 0] = 0.0

    matrix = np.copy(matrixIn)
    mask_matrix = np.isfinite(matrix).astype(int)
    matrix[mask_matrix == 0] = 0.0

    # Initialize by running PARAFAC on the 3D tensor
    parafacSettings = {'orthogonalise': 100, 'tol': 1e-08, 'normalize_factors': False, 'n_iter_max': 600, 'linesearch': True}
    tensorFac = parafac(tensor, r, mask=mask, **parafacSettings)

    tensor = tensor * mask + tl.kruskal_to_tensor(tensorFac, mask=1 - mask)
    assert np.all(np.isfinite(tensor))

    # Now run CMTF
    matrixFac = cmtf(matrix, mask_matrix=mask_matrix, init=tensorFac)

    # Solve for factors on remaining glycosylation matrix variation
    matrixResid = matrixIn - tl.kruskal_to_tensor(matrixFac)
    matrixResid[mask_matrix == 0] = 0.0

    matrixFacExt = parafac(matrixResid, 6, mask=mask_matrix, **parafacSettings)
    ncp = matrixFacExt.rank

    # Incorporate PCA into factorization
    tensorFac.factors[0] = np.concatenate((tensorFac.factors[0], matrixFacExt.factors[0]), axis=1)
    tensorFac.factors[1] = np.pad(tensorFac.factors[1], ((0, 0), (0, ncp)), constant_values=0.0)
    tensorFac.factors[2] = np.pad(tensorFac.factors[2], ((0, 0), (0, ncp)), constant_values=0.0)
    tensorFac.rank += ncp
    tensorFac.weights = np.pad(tensorFac.weights, (0, ncp), constant_values=1.0)
    matrixFac.factors[0] = tensorFac.factors[0]
    matrixFac.factors[1] = np.concatenate((matrixFac.factors[1], matrixFacExt.factors[1]), axis=1)
    matrixFac.weights = np.pad(matrixFac.weights, (0, ncp), constant_values=1.0)
    matrixFac.rank += ncp

    tensor_R2XX = calcR2X(tensorIn, tensorFac)
    matrix_R2XX = calcR2X(matrixIn, matrixFac)
    print("CMTF Tensor R2X: " + str(tensor_R2XX))
    print("CMTF Matrix R2X: " + str(matrix_R2XX))

    return tensorFac, matrixFac, tensor_R2XX, matrix_R2XX
