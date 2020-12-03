"""
Tensor decomposition methods
"""
import pickle
from os.path import join
import numpy as np
from scipy.linalg import khatri_rao
import tensorly as tl
from tensorly.decomposition._cp import initialize_cp
from tensorly.cp_tensor import CPTensor
from .dataImport import createCube, path_here

tl.set_backend('numpy')


def calcR2X(tensorIn, matrixIn, tensorFac, matrixFac):
    """ Calculate R2X. """
    tErr = np.nanvar(tl.cp_to_tensor(tensorFac) - tensorIn)
    mErr = np.nanvar(tl.cp_to_tensor(matrixFac) - matrixIn)
    return 1.0 - (tErr + mErr) / (np.nanvar(tensorIn) + np.nanvar(matrixIn))


def perform_CMTF(tOrig=None, mOrig=None, r=14, cache=True):
    """ Perform CMTF decomposition. """
    if tOrig is None:
        tOrig, mOrig = createCube()

    tensorIn = tOrig.copy()
    tmask = np.isnan(tensorIn)
    matrixIn = mOrig.copy()
    mmask = np.isnan(matrixIn)

    if cache:
        path = join(path_here, "syserol/data/cache.p")
        tFill, mFill = pickle.load(open(path, "rb" ))
        matrixIn[mmask] = tl.cp_to_tensor(mFill)[mmask]
        tensorIn[tmask] = tl.cp_to_tensor(tFill)[tmask]
    else:
        tensorIn[tmask] = np.nanmean(tOrig)
        matrixIn[mmask] = np.nanmean(mOrig)

    tFac = CPTensor(initialize_cp(tensorIn, r, non_negative=False))
    mFac = CPTensor(initialize_cp(matrixIn, r, non_negative=False))

    # Pre-unfold
    selPat = np.all(np.isfinite(mOrig), axis=1)
    unfolded = tl.unfold(tOrig, 0)
    missing = np.any(np.isnan(unfolded), axis=0)
    unfolded = unfolded[:, ~missing]

    R2X_last = R2X = -1000.0

    for ii in range(90000):
        # Solve for the patient matrix
        kr = khatri_rao(tFac.factors[1], tFac.factors[2])[~missing, :]
        kr2 = np.vstack((kr, mFac.factors[1]))
        unfolded2 = np.hstack((unfolded, matrixIn))

        tFac.factors[0] = np.linalg.lstsq(kr2, unfolded2.T, rcond=None)[0].T
        mFac.factors[0] = tFac.factors[0]

        # PARAFAC on other antigen modes
        for m in [1, 2]:
            pinv = np.dot(tFac.factors[0].T, tFac.factors[0]) * np.dot(tFac.factors[3 - m].T, tFac.factors[3 - m])
            mttkrp = tl.unfolding_dot_khatri_rao(tensorIn, tFac, m)
            tFac.factors[m] = np.linalg.solve(pinv.T, mttkrp.T).T

        # Solve for the glycan matrix fit
        mFac.factors[1] = np.linalg.lstsq(mFac.factors[0][selPat, :], mOrig[selPat, :], rcond=None)[0].T

        # Fill in glycan matrix
        matrixIn[mmask] = tl.cp_to_tensor(mFac)[mmask]
        tensorIn[tmask] = tl.cp_to_tensor(tFac)[tmask]

        if ii % 200 == 0:
            R2X_last = R2X
            R2X = calcR2X(tOrig, mOrig, tFac, mFac)

        if R2X - R2X_last < 1e-9:
            break

    tFac.normalize()
    mFac.normalize()

    return tFac, mFac, R2X
