"""
Tensor decomposition methods
"""
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad
from jax.config import config
from scipy.optimize import minimize
from scipy.linalg import khatri_rao
import tensorly as tl
from tensorly.decomposition._nn_cp import initialize_nn_cp
from copy import deepcopy
from .dataImport import createCube


tl.set_backend('numpy')
config.update("jax_enable_x64", True)


def buildGlycan(tFac):
    """ Build the glycan matrix from the factors. """
    return tFac.factors[0] @ tFac.mFactor.T


def calcR2X(tFac, tIn=None, mIn=None):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    assert (tIn is not None) or (mIn is not None)

    vTop = 0.0
    vBottom = 0.0

    if tIn is not None:
        tMask = np.isfinite(tIn)
        vTop += jnp.sum(jnp.square(tl.cp_to_tensor(tFac) * tMask - np.nan_to_num(tIn)))
        vBottom += np.sum(np.square(np.nan_to_num(tIn)))
    if mIn is not None:
        mMask = np.isfinite(mIn)
        recon = tFac if isinstance(tFac, np.ndarray) else buildGlycan(tFac)
        vTop += jnp.sum(jnp.square(recon * mMask - np.nan_to_num(mIn)))
        vBottom += np.sum(np.square(np.nan_to_num(mIn)))

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
    def totalVar(tFac): return np.nanvar(tl.cp_to_tensor(tFac)) + np.nanvar(buildGlycan(tFac))
    vars = np.array([totalVar(delete_component(tFac, np.delete(np.arange(rr), i))) for i in np.arange(rr)])
    order = np.flip(np.argsort(vars))

    tensor.weights = tensor.weights[order]
    tensor.mFactor = tensor.mFactor[:, order]
    for i, fac in enumerate(tensor.factors):
        tensor.factors[i] = fac[:, order]

    np.testing.assert_allclose(tl.cp_to_tensor(tFac), tl.cp_to_tensor(tensor))
    np.testing.assert_allclose(buildGlycan(tFac), buildGlycan(tensor))
    return tensor


def delete_component(tFac, compNum):
    """ Delete the indicated component. """
    tensor = deepcopy(tFac)
    compNum = np.array(compNum, dtype=int)

    # Assert that component # don't exceed range, and are unique
    assert np.amax(compNum) < tensor.rank
    assert np.unique(compNum).size == compNum.size

    tensor.rank -= compNum.size
    tensor.weights = np.delete(tensor.weights, compNum)
    tensor.mFactor = np.delete(tensor.mFactor, compNum, axis=1)
    tensor.factors = [np.delete(fac, compNum, axis=1) for fac in tensor.factors]
    return tensor


def censored_lstsq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solves least squares problem subject to missing data.

    Note: uses a for loop over the missing patterns of B, leading to a
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
    # Calculate the missingness patterns
    unique, uIDX = np.unique(np.isfinite(B), axis=1, return_inverse=True)

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
            tFac.mFactor *= scales

        tFac.factors[i] /= scales

    return tFac


def perform_CMTF(tOrig=None, mOrig=None, r=5):
    """ Perform CMTF decomposition. """
    if tOrig is None:
        tOrig, mOrig = createCube()

    tFac = initialize_nn_cp(np.nan_to_num(tOrig), r, nntype="nndsvd")

    # Pre-unfold
    selPat = np.all(np.isfinite(mOrig), axis=1)
    unfolded = [tl.unfold(tOrig, i) for i in range(3)]
    unfolded[0] = np.hstack((unfolded[0], mOrig))

    tFac.R2X = -1.0
    tFac.mFactor = np.linalg.lstsq(tFac.factors[0][selPat, :], mOrig[selPat, :], rcond=None)[0].T

    for ii in range(100):
        # Solve for the subject matrix
        kr = khatri_rao(tFac.factors[1], tFac.factors[2])
        kr2 = np.vstack((kr, tFac.mFactor))

        tFac.factors[0] = censored_lstsq(kr2, unfolded[0].T)

        # PARAFAC on other antigen modes
        for m in [1, 2]:
            kr = khatri_rao(tFac.factors[0], tFac.factors[3 - m])
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T)

        # Solve for the glycan matrix fit
        tFac.mFactor = np.linalg.lstsq(tFac.factors[0][selPat, :], mOrig[selPat, :], rcond=None)[0].T

        if ii % 2 == 0:
            R2X_last = tFac.R2X
            tFac.R2X = calcR2X(tFac, tOrig, mOrig)
            assert tFac.R2X > 0.0

        if tFac.R2X - R2X_last < 1e-6:
            break

    # Refine with direct optimization
    tFac = fit_refine(tFac, tOrig, mOrig)

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)

    if r > 1:
        tFac = sort_factors(tFac)

    return tFac


def cp_to_vec(tFac):
    vec = np.concatenate([tFac.factors[i].flatten() for i in range(3)])
    return np.concatenate((vec, tFac.mFactor.flatten()))


def buildTensors(pIn, tensor, matrix, r):
    """ Use parameter vector to build kruskal tensors. """
    assert tensor.shape[0] == matrix.shape[0]
    nN = np.cumsum(np.array(tensor.shape) * r)
    A = jnp.reshape(pIn[:nN[0]], (tensor.shape[0], r))
    B = jnp.reshape(pIn[nN[0]:nN[1]], (tensor.shape[1], r))
    C = jnp.reshape(pIn[nN[1]:nN[2]], (tensor.shape[2], r))
    tFac = tl.cp_tensor.CPTensor((None, [A, B, C]))
    tFac.mFactor = jnp.reshape(pIn[nN[2]:], (matrix.shape[1], r))
    return tFac


def cost(pIn, tOrig, mOrig, r):
    tFac = buildTensors(pIn, tOrig, mOrig, r)
    return -calcR2X(tFac, tOrig, mOrig)


def fit_refine(tFac, tOrig, mOrig):
    """ Refine the factorization with direct optimization. """
    r = tFac.rank
    x0 = cp_to_vec(tFac)

    gF = value_and_grad(cost, 0)

    def gradF(*args):
        value, grad = gF(*args)
        return value, np.array(grad)

    tl.set_backend('jax')
    # TODO: Setup constraint to avoid opposing components
    res = minimize(gradF, x0, method="L-BFGS-B", jac=True, args=(tOrig, mOrig, r), options={"gtol": 1e-10, "ftol": 1e-10})
    tl.set_backend('numpy')

    tFac = buildTensors(res.x, tOrig, mOrig, r)
    tFac.R2X = calcR2X(tFac, tOrig, mOrig)
    return tFac
