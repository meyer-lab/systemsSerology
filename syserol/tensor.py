"""
Tensor decomposition methods
"""
import numpy as np
from scipy.optimize import minimize
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.decomposition._nn_cp import make_svd_non_negative
from copy import deepcopy
from .dataImport import createCube


tl.set_backend('numpy')


def buildGlycan(tFac):
    """ Build the glycan matrix from the factors. """
    return tFac.factors[0] @ tFac.mFactor.T


def calcR2X(tFac, tIn=None, mIn=None):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    assert (tIn is not None) or (mIn is not None)

    vTop, vBottom = 0.0, 0.0

    if tIn is not None:
        tMask = np.isfinite(tIn)
        vTop += np.sum(np.square(tl.cp_to_tensor(tFac) * tMask - np.nan_to_num(tIn)))
        vBottom += np.sum(np.square(np.nan_to_num(tIn)))
    if mIn is not None:
        mMask = np.isfinite(mIn)
        recon = tFac if isinstance(tFac, np.ndarray) else buildGlycan(tFac)
        vTop += np.sum(np.square(recon * mMask - np.nan_to_num(mIn)))
        vBottom += np.sum(np.square(np.nan_to_num(mIn)))

    return 1.0 - vTop / vBottom


def tensor_degFreedom(tFac) -> int:
    """ Calculate the degrees of freedom within a tensor factorization. """
    deg = np.sum([f.size for f in tFac.factors])

    if hasattr(tFac, 'mFactor'):
        deg += tFac.mFactor.size

    return deg


def reorient_factors(tFac):
    """ This function ensures that factors are negative on at most one direction. """
    # Flip the subjects to be positive
    subjMeans = np.sign(np.mean(tFac.factors[0], axis=0))
    tFac.factors[0] *= subjMeans[np.newaxis, :]
    tFac.factors[1] *= subjMeans[np.newaxis, :]

    if hasattr(tFac, 'mFactor'):
        tFac.mFactor *= subjMeans[np.newaxis, :]

    # Flip the receptors to be positive
    rMeans = np.sign(np.mean(tFac.factors[1], axis=0))
    tFac.factors[1] *= rMeans[np.newaxis, :]
    tFac.factors[2] *= rMeans[np.newaxis, :]
    return tFac


def totalVar(tFac):
    """ Total variance of a factorization on reconstruction. """
    varr = tl.cp_norm(tFac)
    if hasattr(tFac, 'mFactor'):
        varr += tl.cp_norm((None, [tFac.factors[0], tFac.mFactor]))
    return varr


def sort_factors(tFac):
    """ Sort the components from the largest variance to the smallest. """
    rr = tFac.rank
    tensor = deepcopy(tFac)
    vars = np.array([totalVar(delete_component(tFac, np.delete(np.arange(rr), i))) for i in np.arange(rr)])
    order = np.flip(np.argsort(vars))

    tensor.weights = tensor.weights[order]
    tensor.factors = [fac[:, order] for fac in tensor.factors]
    np.testing.assert_allclose(tl.cp_to_tensor(tFac), tl.cp_to_tensor(tensor))

    if hasattr(tFac, 'mFactor'):
        tensor.mFactor = tensor.mFactor[:, order]
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

    if hasattr(tFac, 'mFactor'):
        tensor.mFactor = np.delete(tensor.mFactor, compNum, axis=1)

    tensor.factors = [np.delete(fac, compNum, axis=1) for fac in tensor.factors]
    return tensor


def censored_lstsq(A: np.ndarray, B: np.ndarray, uniqueInfo) -> np.ndarray:
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
    # Missingness patterns
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
        if i == 0 and hasattr(tFac, 'mFactor'):
            tFac.mFactor *= scales

        tFac.factors[i] /= scales

    return tFac


def initialize_nn_cp(tensor, matrix, rank):
    r"""Initialize factors used in `parafac`.
    Parameters
    ----------
    tensor : ndarray
    rank : int
    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    factors = []
    for mode in range(tl.ndim(tensor)):
        unfold = tl.unfold(tensor, mode)

        if mode == 0 and (matrix is not None):
            unfold = np.hstack((unfold, matrix))

        # Remove completely missing columns
        unfold = unfold[:, ~np.all(np.isnan(unfold), axis=0)]

        U, S, V = np.linalg.svd(np.nan_to_num(unfold))

        # Apply nnsvd to make non-negative
        U = make_svd_non_negative(tensor, U, S, V, "nndsvd")

        if U.shape[1] < rank:
            # This is a hack but it seems to do the job for now
            pad_part = np.random.rand(U.shape[0], rank - U.shape[1])
            U = tl.concatenate([U, pad_part], axis=1)

        factors.append(U[:, :rank])

    return tl.cp_tensor.CPTensor((None, factors))


def perform_CMTF(tOrig=None, mOrig=None, r=5, ALS=True):
    """ Perform CMTF decomposition. """
    if tOrig is None:
        tOrig, mOrig = createCube()

    tFac = initialize_nn_cp(tOrig, mOrig, r)

    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]
    tFac.R2X = -1.0

    if mOrig is not None:
        uniqueInfoM = np.unique(np.isfinite(mOrig), axis=1, return_inverse=True)
        tFac.mFactor = censored_lstsq(tFac.factors[0], mOrig, uniqueInfoM)
        unfolded[0] = np.hstack((unfolded[0], mOrig))

    if ALS:
        # Precalculate the missingness patterns
        uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

        for ii in range(200):
            # Solve for the subject matrix
            kr = khatri_rao(tFac.factors, skip_matrix=0)

            if mOrig is not None:
                kr = np.vstack((kr, tFac.mFactor))

            tFac.factors[0] = censored_lstsq(kr, unfolded[0].T, uniqueInfo[0])

            # PARAFAC on other antigen modes
            for m in range(1, len(tFac.factors)):
                kr = khatri_rao(tFac.factors, skip_matrix=m)
                tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m])

            # Solve for the glycan matrix fit
            if mOrig is not None:
                tFac.mFactor = censored_lstsq(tFac.factors[0], mOrig, uniqueInfoM)

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
    vec = np.concatenate([f.flatten() for f in tFac.factors])

    # Add matrix if present
    if hasattr(tFac, 'mFactor'):
        vec = np.concatenate((vec, tFac.mFactor.flatten()))

    return vec


def buildTensors(pIn, tensor, matrix, r):
    """ Use parameter vector to build kruskal tensors. """
    nN = np.cumsum(np.array(tensor.shape) * r)
    nN = np.insert(nN, 0, 0)
    factorList = [np.reshape(pIn[nN[i]:nN[i+1]], (tensor.shape[i], r)) for i in range(tensor.ndim)]
    tFac = tl.cp_tensor.CPTensor((None, factorList))

    if matrix is not None:
        assert tensor.shape[0] == matrix.shape[0]
        tFac.mFactor = np.reshape(pIn[nN[3]:], (matrix.shape[1], r))
    return tFac


def fit_refine(tFac, tOrig, mOrig):
    """ Refine the factorization with direct optimization. """
    r = tFac.rank
    x0 = cp_to_vec(tFac)
    R2Xbefore = tFac.R2X

    Z = np.nan_to_num(tOrig)
    normZsqr = np.square(np.linalg.norm(Z))
    if mOrig is not None:
        ZM = np.nan_to_num(mOrig)
        normZsqrM = np.square(np.linalg.norm(ZM))

    def gradF(pIn, tOrig, mOrig, r):
        # Tensor
        tFac = buildTensors(pIn, tOrig, mOrig, r)
        B = np.isfinite(tOrig) * tl.cp_to_tensor(tFac)
        f = 0.5 * normZsqr - tl.tenalg.inner(Z, B) + 0.5 * np.square(np.linalg.norm(B))
        T = Z - B

        Gfactors = [-tl.unfolding_dot_khatri_rao(T, tFac, ii) for ii in range(tOrig.ndim)]
        tFacG = tl.cp_tensor.CPTensor((None, Gfactors))

        # Matrix
        if mOrig is not None:
            BM = np.isfinite(mOrig) * buildGlycan(tFac)
            f += 0.5 * normZsqrM - tl.tenalg.inner(ZM, BM) + 0.5 * np.square(np.linalg.norm(BM))
            TM = ZM - BM

            Mcp = tl.cp_tensor.CPTensor((None, [tFac.factors[0], tFac.mFactor]))
            tFacG.factors[0] -= tl.unfolding_dot_khatri_rao(TM, Mcp, 0)
            tFacG.mFactor = -tl.unfolding_dot_khatri_rao(TM, Mcp, 1)

        grad = cp_to_vec(tFacG)
        return f / 1.0e12, grad / 1.0e12

    res = minimize(gradF, x0, method="L-BFGS-B", jac=True, args=(tOrig, mOrig, r), options={"gtol": 1e-10, "ftol": 1e-10})

    tFac = buildTensors(res.x, tOrig, mOrig, r)
    tFac.R2X = calcR2X(tFac, tOrig, mOrig)

    assert R2Xbefore < tFac.R2X
    return tFac
