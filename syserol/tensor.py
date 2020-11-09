"""
Tensor decomposition methods
"""
import numpy as np
from scipy.linalg import khatri_rao
import tensorly as tl
from tensorly.decomposition._cp import initialize_cp
from tensorly.cp_tensor import CPTensor
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
    tensorIn[tmask] = np.nanmean(tensorOrig)
    matrixIn = matrixOrig.copy()
    mmask = np.isnan(matrixIn)
    matrixIn[mmask] = np.nanmean(matrixOrig)

    tFac = CPTensor(initialize_cp(tensorIn, r, non_negative=True))
    mFac = CPTensor(initialize_cp(matrixIn, r, non_negative=True))

    # Pre-unfold
    selPat = np.all(np.isfinite(matrixOrig), axis=1)
    unfolded = tl.unfold(tensorOrig, 0)
    missing = np.any(np.isnan(unfolded), axis=0)
    unfolded = unfolded[:, ~missing]

    R2X_last = R2X = 0.0

    for ii in range(20000):
        # Solve for the patient matrix
        kr = khatri_rao(tFac.factors[1], tFac.factors[2])[~missing, :]
        kr2 = np.vstack((kr, mFac.factors[1]))
        unfolded2 = np.hstack((unfolded, matrixIn))

        tFac.factors[0] = np.linalg.lstsq(kr2, unfolded2.T, rcond=None)[0].T
        mFac.factors[0] = tFac.factors[0]

        # PARAFAC on other antigen modes
        for mode in [1, 2]:
            pinv = np.ones((r, r))
            for i, factor in enumerate(tFac.factors):
                if i != mode:
                    pinv *= np.dot(factor.T, factor)

            mttkrp = tl.unfolding_dot_khatri_rao(tensorIn, (None, tFac.factors), mode)
            tFac.factors[mode] = np.linalg.solve(pinv.T, mttkrp.T).T

        # Solve for the glycan matrix fit
        mFac.factors[1] = np.linalg.lstsq(mFac.factors[0][selPat, :], matrixOrig[selPat, :], rcond=None)[0].T

        # Fill in glycan matrix
        matrixIn[mmask] = tl.cp_to_tensor(mFac)[mmask]
        tensorIn[tmask] = tl.cp_to_tensor(tFac)[tmask]

        if ii % 10 == 0:
            R2X_last = R2X
            R2X = calcR2X(tensorOrig, matrixOrig, tFac, mFac)

        if R2X - R2X_last < 1e-6:
            break

    tFac.normalize()
    mFac.normalize()

    return tFac, mFac, R2X
