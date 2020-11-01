"""
Tensor decomposition methods
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import khatri_rao
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import CPTensor, cp_normalize, unfolding_dot_khatri_rao
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

    tensorFac = parafac(tensorIn, r, mask=tmask, n_iter_max=100, orthogonalise=10)
    matrixFac = parafac(matrixIn, r, mask=mmask, n_iter_max=10, orthogonalise=5)

    # Pre-unfold
    selPat = np.all(np.isfinite(matrixOrig), axis=1)
    unfolded = tl.unfold(tensorOrig, 0)
    missing = np.any(np.isnan(unfolded), axis=0)
    unfolded = unfolded[:, ~missing]

    for ii in range(10):
        pinv = np.ones((r, r)) * np.dot(np.conj(tensorFac.factors[1].T), tensorFac.factors[1])
        pinv *= np.dot(np.conj(tensorFac.factors[2].T), tensorFac.factors[2])
        pinv = np.conj(pinv.T)

        kr = khatri_rao(tensorFac.factors[1], tensorFac.factors[2])[~missing, :]
        mttkrp = tl.dot(unfolded, kr)
        tensorFac.factors[0] = tl.solve(pinv, mttkrp.T).T
        matrixFac.factors[0] = tensorFac.factors[0]

        tensorFac = parafac(tensorIn, r, init=tensorFac, mask=tmask, fixed_modes=[0], n_iter_max=10)

        # Solve for the glycan matrix fit
        matrixFac.factors[1] = np.linalg.lstsq(matrixFac.factors[0][selPat, :], matrixOrig[selPat, :], rcond=None)[0].T

        print(calcR2X(tensorOrig, matrixIn, tensorFac, matrixFac))

    tensorFac = cp_normalize(tensorFac)
    matrixFac = cp_normalize(matrixFac)

    R2X = calcR2X(tensorOrig, matrixIn, tensorFac, matrixFac)

    for ii in range(3):
        tensorFac.factors[ii] = np.array(tensorFac.factors[ii])
    for ii in range(2):
        matrixFac.factors[ii] = np.array(matrixFac.factors[ii])

    return tensorFac, matrixFac, R2X
