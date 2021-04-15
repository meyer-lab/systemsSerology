""" Evaluate the ability of CP to impute data. """

import numpy as np
import tensorly as tl
from .dataImport import createCube
from .tensor import perform_CMTF


def evaluate_missing(comps, numSample=15, indv=False):
    """ check differences between original and recon values for different number of components. """
    cube, glyCube = createCube()
    np.random.seed(1)  # Avoid random variation in this output

    R2X = np.zeros(comps.shape)
    missingCube = np.copy(cube)
    for _ in range(numSample):
        idxs = np.argwhere(np.isfinite(missingCube))
        i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
        if indv == True:
            missingCube[i, j, k] = np.nan
        else:
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
