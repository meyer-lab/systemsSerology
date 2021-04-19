"""
Tensor decomposition methods
"""
import os
from os.path import join, dirname
import pickle
import numpy as np
from scipy.linalg import khatri_rao
import tensorly as tl
from tensorly.decomposition._nn_cp import initialize_nn_cp
from copy import deepcopy
from .dataImport import createCube

tl.set_backend("numpy")
path_here = dirname(dirname(__file__))


def calcR2X(tIn, mIn, tFac):
    """ Calculate R2X. """
    tErr = np.nanvar(tl.cp_to_tensor(tFac) - tIn)
    mErr = np.nanvar(tFac.factors[0] @ tFac.mFactor.T - mIn)
    return 1.0 - (tErr + mErr) / (np.nanvar(tIn) + np.nanvar(mIn))


def reorient_factors(tFac):
    """ This function ensures that factors are negative on at most one direction. """
    # Flip the subjects to be positive
    subjMeans = np.sign(np.mean(tFac.factors[0], axis=0))
    tFac.factors[0] *= subjMeans[np.newaxis, :]
    tFac.factors[1] *= subjMeans[np.newaxis, :]
    tFac.mFactor *= subjMeans[np.newaxis, :]

    # Flip the receptors to be positive
    rMeans = np.sign(np.mean(tFac.factors[1], axis=0))
    tFac.factors[1] *= rMeans[np.newaxis, :]
    tFac.factors[2] *= rMeans[np.newaxis, :]
    return tFac

def totalVar(tFac):
    return np.nanvar(tl.cp_to_tensor(tFac)) + np.nanvar(tFac.factors[0] @ tFac.mFactor.T)

def sort_factors(tFac):
    rr = tFac.rank
    tensor = deepcopy(tFac)
    vars = np.array([totalVar(delete_component(tFac, np.delete(np.arange(rr), i))) for i in np.arange(rr)])
    order = np.flip(np.argsort(vars))

    tensor.weights = tensor.weights[order]
    tensor.mWeights = tensor.mWeights[order]
    tensor.mFactor = tensor.mFactor[:, order]
    for i, fac in enumerate(tensor.factors):
        tensor.factors[i] = fac[:, order]
    assert np.all(tl.cp_to_tensor(tFac) - tl.cp_to_tensor(tensor) < 0.01)
    return tensor

def delete_component(tFac, compNum):
    """ Delete the indicated component. """
    tensor = deepcopy(tFac)
    if isinstance(compNum, int):
        assert compNum < tensor.rank
        tensor.rank -= 1
    elif isinstance(compNum, list) or isinstance(compNum, np.ndarray):
        compNum = np.unique(compNum)
        assert all(i < tensor.rank for i in compNum)
        tensor.rank -= len(compNum)
    else:
        raise TypeError

    tensor.weights = np.delete(tensor.weights, compNum)
    tensor.mWeights = np.delete(tensor.mWeights, compNum)
    tensor.mFactor = np.delete(tensor.mFactor, compNum, axis=1)
    for i, fac in enumerate(tensor.factors):
        tensor.factors[i] = np.delete(fac, compNum, axis=1)
        assert tensor.factors[i].shape[1] == tensor.rank

    return tensor


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
        uI = uIDX == i
        uu = np.squeeze(unique[:, i])

        Bx = B[uu, :]
        X[:, uI] = np.linalg.lstsq(A[uu, :], Bx[:, uI], rcond=None)[0]
    return X.T


def cp_normalize(tFac):
    tFac.factors[0] *= tFac.weights
    tFac.weights = np.ones(tFac.rank)
    tFac.mWeights = np.ones(tFac.rank)

    for i, factor in enumerate(tFac.factors):
        scales = np.linalg.norm(factor, ord=np.inf, axis=0)
        tFac.weights *= scales
        if i == 0:
            tFac.mWeights *= scales

        tFac.factors[i] /= scales

    # Handle matrix
    scales = np.linalg.norm(tFac.mFactor, ord=np.inf, axis=0)
    tFac.mWeights *= scales
    tFac.mFactor /= scales

    return tFac


def perform_CMTF(tOrig=None, mOrig=None, r=10):
    """ Perform CMTF decomposition. """
    filename = join(path_here, "syserol/data/" + str(r) + ".pkl")

    if (tOrig is None) and (r > 2):
        pick = True
        if os.path.exists(filename):
            with open(filename, "rb") as p:
                return pickle.load(p)
    else:
        pick = False

    if tOrig is None:
        tOrig, mOrig = createCube()

    tFac = initialize_nn_cp(np.nan_to_num(tOrig, nan=np.nanmean(tOrig)), r)

    # Pre-unfold
    selPat = np.all(np.isfinite(mOrig), axis=1)
    unfolded = [tl.unfold(tOrig, i) for i in range(3)]
    unfolded[0] = np.hstack((unfolded[0], mOrig))

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    R2X = -1.0
    tFac.mFactor = np.linalg.lstsq(tFac.factors[0][selPat, :], mOrig[selPat, :], rcond=None)[0].T

    for ii in range(8000):
        # Solve for the subject matrix
        kr = khatri_rao(tFac.factors[1], tFac.factors[2])
        kr2 = np.vstack((kr, tFac.mFactor))

        tFac.factors[0] = censored_lstsq(kr2, unfolded[0].T, uniqueInfo[0])

        # PARAFAC on other antigen modes
        for m in [1, 2]:
            kr = khatri_rao(tFac.factors[0], tFac.factors[3 - m])
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m])

        # Solve for the glycan matrix fit
        tFac.mFactor = np.linalg.lstsq(tFac.factors[0][selPat, :], mOrig[selPat, :], rcond=None)[0].T

        if ii % 20 == 0:
            R2X_last = R2X
            R2X = calcR2X(tOrig, mOrig, tFac)

        if R2X - R2X_last < 1e-9:
            break

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)
    tFac.R2X = R2X

    if pick:
        with open(filename, "wb") as p:
            pickle.dump(tFac, p)

    return sort_factors(tFac)
