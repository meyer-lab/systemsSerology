"""
Tensor decomposition methods
"""
import numpy as np
from scipy.linalg import khatri_rao
import tensorly as tl
from statsmodels.multivariate.pca import PCA
from tensorly.decomposition._cp import initialize_cp
from tensorly.decomposition import tucker
from tensorly.cp_tensor import CPTensor
from .dataImport import createCube

tl.set_backend('numpy')


def calcR2X(tensorIn, matrixIn, tensorFac, matrixFac):
    """ Calculate R2X. """
    if isinstance(tensorFac, CPTensor):
        tErr = np.nanvar(tl.cp_to_tensor(tensorFac) - tensorIn)
    else:
        tErr = np.nanvar(tl.tucker_to_tensor(tensorFac) - tensorIn)

    mErr = np.nanvar(tl.cp_to_tensor(matrixFac) - matrixIn)
    return 1.0 - (tErr + mErr) / (np.nanvar(tensorIn) + np.nanvar(matrixIn))


import numpy as np
from scipy.sparse.linalg import svds
from functools import partial


def emsvd(Y, k, tol=1E-3, maxiter=3000):
    """
    Approximate SVD on data with missing values via expectation-maximization

    Inputs:
    -----------
    Y:          (nobs, ndim) data matrix, missing values denoted by NaN/Inf
    k:          number of singular values/vectors to find (default: k=ndim)
    tol:        convergence tolerance on change in trace norm
    maxiter:    maximum number of EM steps to perform (default: no limit)

    Returns:
    -----------
    Y_hat:      (nobs, ndim) reconstructed data matrix
    mu_hat:     (ndim,) estimated column means for reconstructed data
    U, s, Vt:   singular values and vectors (see np.linalg.svd and 
                scipy.sparse.linalg.svds for details)
    """
    # initialize the missing values to their respective column means
    Y = np.copy(Y)
    valid = np.isfinite(Y)
    Y = np.nan_to_num(Y)
    y_prev = np.copy(Y)

    for ii in range(maxiter):
        # SVD on filled-in data
        U, s, Vt = svds(Y, k=k)

        # impute missing values
        Y[~valid] = (U.dot(np.diag(s)).dot(Vt))[~valid]

        # test convergence using relative change in trace norm
        if np.linalg.norm(Y - y_prev) < tol:
            break

        y_prev = np.copy(Y)

    return U


def perform_CMTF(tOrig=None, mOrig=None, r=10):
    """ Perform CMTF decomposition. """
    if tOrig is None:
        tOrig, mOrig = createCube()

    r = int(r)
    tOrig = np.copy(tOrig)

    tMat = np.reshape(np.copy(tOrig), (181, -1))
    tMat = tMat[:, ~np.all(np.isnan(tMat), axis=0)]
    tMat = np.hstack((tMat, mOrig))
    factorOne = emsvd(tMat, r)

    tFac = tucker(np.nan_to_num(tOrig), r, n_iter_max=1)
    tFac.factors[0] = factorOne

    mFac = CPTensor(initialize_cp(np.nan_to_num(mOrig), r))
    mFac.factors[0] = tFac.factors[0]
    selPat = np.all(np.isfinite(mOrig), axis=1)
    mFac.factors[1] = np.linalg.lstsq(mFac.factors[0][selPat, :], mOrig[selPat, :], rcond=None)[0].T

    for m in [1, 2]:
        tFac.factors[m] = emsvd(tl.unfold(np.copy(tOrig), m), r)

    tmask = np.isnan(tOrig)

    for ii in range(5):
        tOrig[tmask] = tl.tucker_to_tensor(tFac)[tmask]
        tFac.core = tl.tenalg.multi_mode_dot(tOrig, tFac.factors, modes=[0, 1, 2], transpose=True)

    for ii in range(5):
        print(calcR2X(tOrig, mOrig, tFac, mFac))
        tFac = tucker(tOrig, r, n_iter_max=2, init=tFac, fixed_factors=[0])
        tOrig[tmask] = tl.tucker_to_tensor(tFac)[tmask]

    R2X = calcR2X(tOrig, mOrig, tFac, mFac)
    print(R2X)

    return tFac, mFac, R2X
