"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorpack import calcR2X, censored_lstsq, cp_normalize, sort_factors
from tensorpack.SVD_impute import IterativeSVD
from .dataImport import createCube


tl.set_backend('numpy')


def reorient_factors(tFac):
    """ This function ensures that factors are negative on at most one direction. """
    # Flip the subjects to be positive
    rMeans = np.sign(np.mean(tFac.factors[1], axis=0))
    agMeans = np.sign(np.mean(tFac.factors[2], axis=0))
    tFac.factors[0] *= rMeans[np.newaxis, :] * agMeans[np.newaxis, :]
    tFac.factors[1] *= rMeans[np.newaxis, :]
    tFac.factors[2] *= agMeans[np.newaxis, :]

    if hasattr(tFac, 'mFactor'):
        tFac.mFactor *= rMeans[np.newaxis, :] * agMeans[np.newaxis, :]

    return tFac


def totalVar(tFac):
    """ Total variance of a factorization on reconstruction. """
    varr = tl.cp_norm(tFac)
    if hasattr(tFac, 'mFactor'):
        varr += tl.cp_norm((None, [tFac.factors[0], tFac.mFactor]))
    return varr


def initialize_cp(tensor: np.ndarray, matrix: np.ndarray, rank: int):
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
        unfold = unfold[:, np.sum(np.isfinite(unfold), axis=0) > 2]

        # Impute by SVD
        si = IterativeSVD(rank=rank, random_state=1)
        unfold = si.fit_transform(unfold)
        U = si.U

        factors.append(U[:, :rank])

    return tl.cp_tensor.CPTensor((None, factors))


def perform_CMTF(tOrig=None, mOrig=None, r=6):
    """ Perform CMTF decomposition. """
    if tOrig is None:
        tOrig, mOrig = createCube()

    tFac = initialize_cp(tOrig, mOrig, r)

    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]

    if mOrig is not None:
        uniqueInfoM = np.unique(np.isfinite(mOrig), axis=1, return_inverse=True)
        tFac.mFactor = censored_lstsq(tFac.factors[0], mOrig, uniqueInfoM)
        unfolded[0] = np.hstack((unfolded[0], mOrig))

    R2X_last = -np.inf
    tFac.R2X = calcR2X(tFac, tOrig, mOrig)

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    for ii in range(2000):
        # PARAFAC on other antigen modes
        for m in range(1, len(tFac.factors)):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m])

        # Solve for the subject matrix
        kr = khatri_rao(tFac.factors, skip_matrix=0)

        if mOrig is not None:
            kr = np.vstack((kr, tFac.mFactor))

        tFac.factors[0] = censored_lstsq(kr, unfolded[0].T, uniqueInfo[0])

        # Solve for the glycan matrix fit
        if mOrig is not None:
            tFac.mFactor = censored_lstsq(tFac.factors[0], mOrig, uniqueInfoM)

        if ii % 2 == 0:
            R2X_last = tFac.R2X
            tFac.R2X = calcR2X(tFac, tOrig, mOrig)
            assert tFac.R2X > 0.0

        if tFac.R2X - R2X_last < 1e-6:
            print(ii)
            break

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)

    if r > 1:
        tFac = sort_factors(tFac)

    print(tFac.R2X)

    return tFac
