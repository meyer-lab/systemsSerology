"""
Tensor decomposition methods
"""
import os
from os.path import join, dirname
import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import khatri_rao
import tensorly as tl
from tensorly.cp_tensor import unfolding_dot_khatri_rao
from tensorly.decomposition._nn_cp import initialize_nn_cp
from copy import deepcopy
from .dataImport import createCube

tl.set_backend('numpy')
path_here = dirname(dirname(__file__))


def buildGlycan(tFac):
    """ Build the glycan matrix from the factors. """
    return (tFac.mWeights * tFac.factors[0]) @ tFac.mFactor.T


def calcR2X(tFac, tIn=None, mIn=None):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    assert (tIn is not None) or (mIn is not None)

    vTop = 0.0
    vBottom = 0.0

    if tIn is not None:
        vTop += np.nanvar(tl.cp_to_tensor(tFac) - tIn)
        vBottom += np.nanvar(tIn)
    if mIn is not None:
        vTop += np.nanvar(buildGlycan(tFac) - mIn)
        vBottom += np.nanvar(mIn)

    return 1.0 - vTop / vBottom


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

def sort_factors(tFac):
    """ Sort the components from the largest variance to the smallest. """
    rr = tFac.rank
    tensor = deepcopy(tFac)
    totalVar = lambda tFac: np.nanvar(tl.cp_to_tensor(tFac)) + np.nanvar(tFac.factors[0] @ tFac.mFactor.T)
    vars = np.array([totalVar(delete_component(tFac, np.delete(np.arange(rr), i))) for i in np.arange(rr)])
    order = np.flip(np.argsort(vars))

    tensor.weights = tensor.weights[order]
    tensor.mWeights = tensor.mWeights[order]
    tensor.mFactor = tensor.mFactor[:, order]
    for i, fac in enumerate(tensor.factors):
        tensor.factors[i] = fac[:, order]

    np.testing.assert_allclose(tl.cp_to_tensor(tFac), tl.cp_to_tensor(tensor))
    np.testing.assert_allclose(buildGlycan(tFac), buildGlycan(tensor))
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
    """ Normalize the factors using the inf norm. """
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


def perform_CMTF(tOrig=None, mOrig=None, r=5):
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

    tFac = initialize_nn_cp(np.nan_to_num(tOrig), r, nntype="nndsvd")

    # Pre-unfold
    selPat = np.all(np.isfinite(mOrig), axis=1)
    unfolded = [tl.unfold(tOrig, i) for i in range(3)]
    unfolded[0] = np.hstack((unfolded[0], mOrig))

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    tFac.R2X = -1.0
    tFac.mFactor = np.linalg.lstsq(tFac.factors[0][selPat, :], mOrig[selPat, :], rcond=None)[0].T
    tFac.mWeights = np.ones(r)

    for ii in range(20):
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

        if ii % 2 == 0:
            R2X_last = tFac.R2X
            tFac.R2X = calcR2X(tFac, tOrig, mOrig)
            assert tFac.R2X > 0.0

        if tFac.R2X - R2X_last < 1e-9:
            break

    # Refine with direct optimization
    tFac = fit_refine(tFac, tOrig, mOrig)

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)
    tFac = sort_factors(tFac)

    if pick:
        with open(filename, "wb") as p:
            pickle.dump(tFac, p)

    return tFac


def cp_to_vec(tFac):
    vec = np.concatenate([tFac.factors[i].flatten() for i in range(3)])
    return np.concatenate((vec, tFac.mFactor.flatten()))


def buildTensors(pIn, tensor, matrix, r):
    """ Use parameter vector to build kruskal tensors. """
    assert tensor.shape[0] == matrix.shape[0]
    nA = tensor.shape[0]*r
    nB = tensor.shape[1]*r
    nC = tensor.shape[2]*r
    A = np.reshape(pIn[:nA], (tensor.shape[0], r))
    B = np.reshape(pIn[nA:nA+nB], (tensor.shape[1], r))
    C = np.reshape(pIn[nA+nB:nA+nB+nC], (tensor.shape[2], r))
    tFac = tl.cp_tensor.CPTensor((None, [A, B, C]))
    tFac.mFactor = np.reshape(pIn[nA+nB+nC:], (matrix.shape[1], r))
    tFac.mWeights = np.ones(r)
    return tFac


def cost(pIn, tOrig, mOrig, r):
    tFac = buildTensors(pIn, tOrig, mOrig, r)
    return -calcR2X(tFac, tOrig, mOrig)


def grad(pIn, tOrig, mOrig, r):
    tFac = buildTensors(pIn, tOrig, mOrig, r)
    tDiff = np.nan_to_num(tOrig - tl.cp_to_tensor(tFac))
    mDiff = np.nan_to_num(mOrig - buildGlycan(tFac))
    totalVar = np.nanvar(tOrig) + np.nanvar(mOrig)

    nT = np.sqrt(tOrig.size)
    nM = np.sqrt(mOrig.size)

    gtFac = deepcopy(tFac)
    gtFac.factors = [-unfolding_dot_khatri_rao(tDiff, tFac, ii) / nT for ii in range(3)]

    mCP = (None, [tFac.factors[0], tFac.mFactor])
    gtFac.factors[0] += -unfolding_dot_khatri_rao(mDiff, mCP, 0) / nM
    gtFac.mFactor = -unfolding_dot_khatri_rao(mDiff, mCP, 1) / nM

    return cp_to_vec(gtFac) / totalVar


from statsmodels.tools.numdiff import approx_fprime


def fit_refine(tFac, tOrig, mOrig):
    """ Refine the factorization with direct optimization. """
    r = tFac.rank

    x0 = cp_to_vec(tFac)
    print(tFac.R2X)

    ndx = approx_fprime(x0, lambda x: cost(x, tOrig, mOrig, r), centered=True)
    fdx = grad(x0, tOrig, mOrig, r)

    print(fdx / ndx)

    res = minimize(cost, x0, jac=grad, args=(tOrig, mOrig, r), options={"disp": True, "maxiter": 300})

    tFac = buildTensors(res.x, tOrig, mOrig, r)
    tFac.R2X = calcR2X(tFac, tOrig, mOrig)
    print(tFac.R2X)

    return tFac
