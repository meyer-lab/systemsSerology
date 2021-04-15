""" Evaluate the ability of CP to impute data. """

import numpy as np
import tensorly as tl
from statsmodels.multivariate.pca import PCA
from .dataImport import createCube
from .tensor import perform_CMTF


def flatten_to_mat(tensor, matrix):
    n = tensor.shape[0]
    tMat = np.reshape(tensor, (n, -1))
    tMat = tMat[:, ~np.all(np.isnan(tMat), axis=0)]
    tMat = np.hstack((tMat, matrix))
    return tMat


def evaluate_missing(comps, numSample=15):
    """ check differences between original and recon values for different number of components. """
    cube, glyCube = createCube()
    np.random.seed(1)  # Avoid random variation in this output

    R2X = np.zeros(comps.shape)
    missingCube = np.copy(cube)
    for _ in range(numSample):
        idxs = np.argwhere(np.isfinite(missingCube))
        _, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
        missingCube[:, j, k] = np.nan

    imputeVals = np.copy(cube)
    imputeVals[np.isfinite(missingCube)] = np.nan

    for ii, nComp in enumerate(comps):
        # reconstruct with some values missing
        tensorR = tl.cp_to_tensor(perform_CMTF(missingCube, glyCube, nComp))
        tensorR[np.isfinite(missingCube)] = np.nan

        # Compare original Cube with reconstructed cube, which was created from the cube with imputed missing values
        R2X[ii] = 1.0 - np.nanvar(tensorR - imputeVals) / np.nanvar(imputeVals)

    return R2X

def evaluate_missing_meas(comps, numSample=15):
    """ check differences between original and recon values for different number of components. """
    cube, glyCube = createCube()
    np.random.seed(1)  # Avoid random variation in this output

    PCAR2X = np.zeros(comps.shape)
    CMTFR2X = np.zeros(comps.shape)
    missingCube = np.copy(cube)
    for _ in range(numSample):
        idxs = np.argwhere(np.isfinite(missingCube))
        i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
        missingCube[i, j, k] = np.nan

    imputeVals = np.copy(cube)
    imputeVals[np.isfinite(missingCube)] = np.nan

    missingMat = flatten_to_mat(missingCube, glyCube)
    imputeMat = np.copy(flatten_to_mat(cube, glyCube))
    imputeMat[np.isfinite(missingMat)] = np.nan

    for ii, nComp in enumerate(comps):
        # reconstruct with some values missing
        tensorR = tl.cp_to_tensor(perform_CMTF(missingCube, glyCube, nComp))
        tensorR[np.isfinite(missingCube)] = np.nan

        # Compare original Cube with reconstructed cube, which was created from the cube with imputed missing values
        CMTFR2X[ii] = 1.0 - np.nanvar(tensorR - imputeVals) / np.nanvar(imputeVals)

        outt = PCA(missingMat, ncomp=nComp, missing="fill-em", standardize=False, demean=False, normalize=False)
        recon = outt.scores @ outt.loadings.T
        PCAR2X[ii] = 1.0 - np.nanvar(recon - imputeMat) / np.nanvar(imputeMat)

    return CMTFR2X, PCAR2X
