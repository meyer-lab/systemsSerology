"""
Tensor decomposition methods
"""
import numpy as np
from scipy.linalg import khatri_rao
import tensorly as tl
from tensorly.decomposition._nn_cp import initialize_nn_cp
from .dataImport import createCube

tl.set_backend("numpy")


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


def censored_lstsq(A, B):
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
    for i in range(B.shape[1]):
        m = np.isfinite(B[:, i])  # drop rows where mask is zero
        X[:, i] = np.linalg.lstsq(A[m], B[m, i], rcond=None)[0]
    return X.T


def perform_CMTF(tOrig=None, mOrig=None, r=10):
    """ Perform CMTF decomposition. """
    if tOrig is None:
        tOrig, mOrig = createCube()

    tFac = initialize_nn_cp(np.nan_to_num(tOrig, nan=np.nanmean(tOrig)), r)
    mFac = initialize_nn_cp(np.nan_to_num(mOrig, nan=np.nanmean(mOrig)), r)

    # Pre-unfold
    selPat = np.all(np.isfinite(mOrig), axis=1)
    unfolded = tl.unfold(tOrig, 0)
    missing = np.any(np.isnan(unfolded), axis=0)
    unfolded = unfolded[:, ~missing]

    R2X = -1.0
    mFac.factors[0] = tFac.factors[0]
    mFac.factors[1] = np.linalg.lstsq(mFac.factors[0][selPat, :], mOrig[selPat, :], rcond=None)[0].T

    for ii in range(8000):
        # Solve for the subject matrix
        kr = khatri_rao(tFac.factors[1], tFac.factors[2])[~missing, :]
        kr2 = np.vstack((kr, mFac.factors[1]))
        unfolded2 = np.hstack((unfolded, mOrig))

        tFac.factors[0] = censored_lstsq(kr2, unfolded2.T)
        mFac.factors[0] = tFac.factors[0]

        # PARAFAC on other antigen modes
        for m in [1, 2]:
            kr = khatri_rao(tFac.factors[0], tFac.factors[3 - m])
            unfold = tl.unfold(tOrig, m)
            tFac.factors[m] = censored_lstsq(kr, unfold.T)

        # Solve for the glycan matrix fit
        mFac.factors[1] = np.linalg.lstsq(mFac.factors[0][selPat, :], mOrig[selPat, :], rcond=None)[0].T

        if ii % 20 == 0:
            R2X_last = R2X
            R2X = calcR2X(tOrig, mOrig, tFac, mFac)

        if R2X - R2X_last < 1e-6:
            break

    tFac.normalize()
    mFac.normalize()

    # Reorient the later tensor factors
    tFac.factors, mFac.factors = reorient_factors(tFac.factors, mFac.factors)

    return tFac, mFac, R2X
