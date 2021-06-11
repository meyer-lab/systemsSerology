"""
Tensor decomposition methods
"""
import numpy as np
from scipy.optimize import minimize, Bounds
import tensorly as tl
from tensorly.decomposition._nn_cp import initialize_nn_cp
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


def cp_normalize(tFac):
    """ Normalize the factors using the inf norm. """
    for i, factor in enumerate(tFac.factors):
        scales = np.linalg.norm(factor, ord=np.inf, axis=0)
        tFac.weights *= scales
        if i == 0 and hasattr(tFac, 'mFactor'):
            tFac.mFactor *= scales

        tFac.factors[i] /= scales

    return tFac


def perform_CMTF(tOrig=None, mOrig=None, r=5):
    """ Perform CMTF decomposition. """
    if tOrig is None:
        tOrig, mOrig = createCube()

    tFac = initialize_nn_cp(np.nan_to_num(tOrig), r, nntype="nndsvd")

    if mOrig is not None:
        tFac.mFactor = np.zeros((mOrig.shape[1], r))

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

    # TODO: Setup constraint to avoid opposing components
    bnds = Bounds(np.zeros_like(x0), np.full_like(x0, np.inf))
    res = minimize(gradF, x0, method="L-BFGS-B", jac=True, bounds=bnds, args=(tOrig, mOrig, r), options={"disp": 90, "gtol": 1e-10, "ftol": 1e-10})

    tFac = buildTensors(res.x, tOrig, mOrig, r)
    tFac.R2X = calcR2X(tFac, tOrig, mOrig)
    print(tFac.R2X)
    return tFac
