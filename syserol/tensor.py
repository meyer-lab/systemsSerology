"""
Tensor decomposition methods
"""
import os
from os.path import join, dirname
import pickle
import numpy as np
from scipy.linalg import khatri_rao
import tensorly as tl
from tensorly.decomposition._cp import initialize_cp
from .dataImport import createCube

tl.set_backend("numpy")
path_here = dirname(dirname(__file__))


def calcR2X(tensorIn, matrixIn, tensorFac, matrixFac):
    """ Calculate R2X. """
    tErr = np.nanvar(tl.cp_to_tensor(tensorFac) - tensorIn)
    mErr = np.nanvar(tl.cp_to_tensor(matrixFac) - matrixIn)
    return 1.0 - (tErr + mErr) / (np.nanvar(tensorIn) + np.nanvar(matrixIn))


def reorient_factors(tensorFac, matrixFac):
    """ This function ensures that factors are negative on at most one direction. """
    for jj in range(1, len(tensorFac)):
        # Calculate the sign of the current factor in each component
        means = np.sign(np.mean(tensorFac[jj], axis=0))

        # Update both the current and last factor
        tensorFac[0] *= means[np.newaxis, :]
        matrixFac[0] *= means[np.newaxis, :]
        matrixFac[1] *= means[np.newaxis, :]
        tensorFac[jj] *= means[np.newaxis, :]
    return tensorFac, matrixFac


def delete_component(cp_tensor, compNum):
    """ Delete the indicated component. """
    assert compNum < cp_tensor.rank

    cp_tensor.weights = np.delete(cp_tensor.weights, compNum)
    cp_tensor.rank -= 1
    for i, fac in enumerate(cp_tensor.factors):
        cp_tensor.factors[i] = np.delete(fac, compNum, axis=0)

    return cp_tensor


def censored_lstsq(A: np.ndarray, B: np.ndarray, uniqueInfo) -> np.ndarray:
    """Solves least squares problem subject to missing data.

    Note: uses a for loop over the columns of B, leading to a
    slower but more numerically stable algorithm

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """
    X = np.empty((A.shape[1], B.shape[1]))
    unique, uIDX = uniqueInfo

    for i in range(unique.shape[1]):
        uI = (uIDX == i)
        uu = np.squeeze(unique[:, i])

        Bx = B[uu, :]
        X[:, uI] = np.linalg.lstsq(A[uu, :], Bx[:, uI], rcond=None)[0]
    return X.T


def cp_normalize(cp_tensor):
    cp_tensor.factors[0] *= cp_tensor.weights
    cp_tensor.weights = np.ones(cp_tensor.rank)

    for i, factor in enumerate(cp_tensor.factors):
        scales = np.linalg.norm(factor, ord=np.inf, axis=0)
        cp_tensor.weights *= scales
        cp_tensor.factors[i] /= scales

    return cp_tensor


def perform_CMTF(tOrig=None, mOrig=None, r=10):
    """ Perform CMTF decomposition. """
    filename = join(path_here, "syserol/data/" + str(r) + ".pkl")

    if (tOrig is None) and (r > 2):
        pick = True
        if os.path.exists(filename):
            with open(filename, 'rb') as p:
                return pickle.load(p)
    else:
        pick = False

    if tOrig is None:
        tOrig, mOrig = createCube()

    tFac = initialize_cp(np.nan_to_num(tOrig), r)

    # Everything from the original mFac will be overwritten
    mFac = initialize_cp(np.nan_to_num(mOrig), r)

    # Pre-unfold
    selPat = np.all(np.isfinite(mOrig), axis=1)
    unfolded = [tl.unfold(tOrig, i) for i in range(3)]
    unfolded[0] = np.hstack((unfolded[0], mOrig))

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    R2X = -1.0
    mFac.factors[0] = tFac.factors[0]
    mFac.factors[1] = np.linalg.lstsq(mFac.factors[0][selPat, :], mOrig[selPat, :], rcond=None)[0].T

    for ii in range(8000):
        # Solve for the subject matrix
        kr = khatri_rao(tFac.factors[1], tFac.factors[2])
        kr2 = np.vstack((kr, mFac.factors[1]))

        tFac.factors[0] = censored_lstsq(kr2, unfolded[0].T, uniqueInfo[0])
        mFac.factors[0] = tFac.factors[0]

        # PARAFAC on other antigen modes
        for m in [1, 2]:
            kr = khatri_rao(tFac.factors[0], tFac.factors[3 - m])
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m])

        # Solve for the glycan matrix fit
        mFac.factors[1] = np.linalg.lstsq(mFac.factors[0][selPat, :], mOrig[selPat, :], rcond=None)[0].T

        if ii % 20 == 0:
            R2X_last = R2X
            R2X = calcR2X(tOrig, mOrig, tFac, mFac)

        if R2X - R2X_last < 1e-9:
            break

    tFac = cp_normalize(tFac)
    mFac = cp_normalize(mFac)

    # Reorient the later tensor factors
    tFac.factors, mFac.factors = reorient_factors(tFac.factors, mFac.factors)

    if pick:
        with open(filename, 'wb') as p:
            pickle.dump((tFac, mFac, R2X), p)

    return tFac, mFac, R2X
