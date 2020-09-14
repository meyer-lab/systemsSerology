"""
Tensor decomposition methods
"""
from copy import deepcopy
import numpy as np
import tensorly as tl
from tensorly.kruskal_tensor import KruskalTensor, kruskal_normalise
from tensorly.decomposition import parafac
from .dataImport import createCube, load_cache


def calcR2X(data, factor):
    """ Calculate R2X. """
    tensorErr = np.nanvar(tl.kruskal_to_tensor(factor) - data)
    return 1.0 - tensorErr / np.nanvar(data)


def reorient_factors(factors):
    """ This function ensures that factors are negative on at most one direction. """
    for jj in range(len(factors) - 1):
        # Calculate the sign of the current factor in each component
        means = np.sign(np.mean(factors[jj]**3, axis = 0))

        # Update both the current and last factor
        factors[jj] *= means[np.newaxis, :]
        factors[-1] *= means[np.newaxis, :]
    return factors


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


def perform_CMTF(tensorIn=None, matrixIn=None, r=6):
    """ Perform CMTF decomposition. """
    cacheMissing = load_cache()

    if tensorIn is None:
        tensorIn, matrixIn = createCube()

    tensor = np.copy(tensorIn)
    mask = np.isfinite(tensor).astype(int)
    tensor[mask == 0] = 0.0

    if cacheMissing is not None:
        tensor = tensor * mask + tl.kruskal_to_tensor(cacheMissing, mask=1 - mask)

    matrix = np.copy(matrixIn)
    mask_matrix = np.isfinite(matrix).astype(int)
    matrix[mask_matrix == 0] = 0.0

    # Initialize by running PARAFAC on the 3D tensor
    parafacSettings = {'orthogonalise': True, 'tol': 1e-9, 'n_iter_max': 1000}
    tensorFac = parafac(tensor, r, mask=mask, **parafacSettings)
    tensorFac.factors = reorient_factors(tensorFac.factors)

    # Now run CMTF
    matrixFac = cmtf(matrix, mask_matrix=mask_matrix, init=tensorFac)

    # Solve for factors on remaining glycosylation matrix variation
    matrixResid = matrix - tl.kruskal_to_tensor(matrixFac)
    matrixFacExt = parafac(matrixResid, r, mask=mask_matrix, **parafacSettings)
    matrixFacExt.factors = reorient_factors(matrixFacExt.factors)
    ncp = matrixFacExt.rank

    # Go back to tensor
    tensorResid = tensor - tl.kruskal_to_tensor(tensorFac)
    tensorFacTwo = deepcopy(tensorFac)
    tensorFacTwo.factors[0] = matrixFacExt.factors[0]
    tensorFacExt = parafac(tensorResid, r, init=tensorFacTwo, fixed_modes=[0])
    # TODO: Fix tensorly so that orthogonalize only applies to non-fixed modes
    # TODO: Explodes when we pass the missingness mask

    # Incorporate PCA into factorization
    for ii in range(3):
        tensorFac.factors[ii] = np.concatenate((tensorFac.factors[ii], tensorFacExt.factors[ii]), axis=1)

    tensorFac.rank += ncp
    matrixFac.rank += ncp
    tensorFac.weights = np.concatenate((tensorFac.weights, tensorFacExt.weights))
    matrixFac.factors[0] = tensorFac.factors[0]
    matrixFac.factors[1] = np.concatenate((matrixFac.factors[1], matrixFacExt.factors[1]), axis=1)
    matrixFac.weights = np.concatenate((matrixFac.weights, matrixFacExt.weights))

    tensor_R2XX = calcR2X(tensorIn, tensorFac)
    matrix_R2XX = calcR2X(matrixIn, matrixFac)
    tensorFac = kruskal_normalise(tensorFac)
    matrixFac = kruskal_normalise(matrixFac)

    # Reorient the later tensor factors
    tensorFac.factors[1::] = reorient_factors(tensorFac.factors[1::])

    return tensorFac, matrixFac, tensor_R2XX, matrix_R2XX
