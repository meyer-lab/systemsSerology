"""
Tensor decomposition methods
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import khatri_rao
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac
from tensorly.cp_tensor import cp_normalize
from .dataImport import createCube

tl.set_backend('numpy')

def calcR2X(tensorIn, matrixIn, tensorFac, matrixFac):
    """ Calculate R2X. """
    tErr = np.nanvar(tl.cp_to_tensor(tensorFac) - tensorIn)
    mErr = np.nanvar(tl.cp_to_tensor(matrixFac) - matrixIn)
    return 1.0 - (tErr + mErr) / (np.nanvar(tensorIn) + np.nanvar(matrixIn))


def perform_CMTF(tensorOrig=None, matrixOrig=None, r=10):
    """ Perform CMTF decomposition. """
    if tensorOrig is None:
        tensorOrig, matrixOrig = createCube()

    tensorIn = tensorOrig.copy()
    tmask = np.isnan(tensorIn)
    tensorIn[tmask] = 0.0
    matrixIn = matrixOrig.copy()
    mmask = np.isnan(matrixIn)
    matrixIn[mmask] = 0.0

    tFac = non_negative_parafac(tensorIn, r, mask=1-tmask, n_iter_max=100)
    mFac = non_negative_parafac(matrixIn, r, mask=1-mmask, n_iter_max=1)

    # Pre-unfold
    selPat = np.all(np.isfinite(matrixOrig), axis=1)
    unfolded = tl.unfold(tensorOrig, 0)
    missing = np.any(np.isnan(unfolded), axis=0)
    unfolded = unfolded[:, ~missing]

    for _ in range(200):
        kr = khatri_rao(tFac.factors[1], tFac.factors[2])[~missing, :]
        kr2 = np.vstack((kr, mFac.factors[1]))
        unfolded2 = np.hstack((unfolded, matrixIn))

        tFac.factors[0] = np.linalg.lstsq(kr2, unfolded2.T, rcond=None)[0].T
        mFac.factors[0] = tFac.factors[0]

        tFac = parafac(tensorIn, r, init=tFac, mask=1-tmask, fixed_modes=[0], n_iter_max=2)

        # Solve for the glycan matrix fit
        mFac.factors[1] = np.linalg.lstsq(mFac.factors[0][selPat, :], matrixOrig[selPat, :], rcond=None)[0].T

        # Fill in glycan matrix
        matrixIn[mmask] = tl.cp_to_tensor(mFac)[mmask]
        tensorIn[tmask] = tl.cp_to_tensor(tFac)[tmask]

    tFac = cp_normalize(tFac)
    mFac = cp_normalize(mFac)

    R2X = calcR2X(tensorOrig, matrixOrig, tFac, mFac)

    return tFac, mFac, R2X
