""" Evaluate the ability of CP to impute data. """

import numpy as np
import tensorly as tl
from .dataImport import createCube
from .tensor import perform_CMTF


def evalMissing(cube, glyCube, numSample=100):
    """ Evaluate how well factorization imputes missing values. """
    cube = np.copy(cube)
    glyCube = np.copy(glyCube)
    orig = []

    indices = list()
    for _ in range(numSample):
        idxs = np.argwhere(np.isfinite(cube))
        i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
        indices.append((np.arange(0, 181),j,k))
        orig.append(cube[:, j, k])
        cube[:, j, k] = np.nan

    return np.array(orig), cube, indices


def evaluate_missing():
    """ check differences between original and recon values for different number of components. """
    Cube, glyCube = createCube()

    Sumsqs = list()
    for nComp in np.arange(1, 17):
        orig, cube, indices = evalMissing(Cube, glyCube, numSample=100)

        factors, _, _ = perform_CMTF(cube, glyCube, nComp)
        tensorR = tl.cp_to_tensor(factors)
        recon = [tensorR[indx[0], indx[1], indx[2]] for indx in indices]
        recon = np.array(recon)

        Sumsqs.append(np.linalg.norm(orig - recon) / np.linalg.norm(orig))

    return Sumsqs
