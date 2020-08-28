""" Evaluate the ability of CP to impute data. """

import numpy as np
import tensorly as tl
from .dataImport import createCube
from .tensor import perform_CMTF


def evalMissing(cube, glyCube, nComp, numSample=100):
    """ Evaluate how well factorization imputes missing values. """
    cube = np.copy(cube)
    glyCube = np.copy(glyCube)
    orig = []

    indices = list()

    for _ in range(numSample):
        idxs = np.argwhere(np.isfinite(cube))
        i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
        indices.append((i, j, k))
        orig.append(cube[i, j, k])
        cube[i, j, k] = np.nan

    factors, _, _, _ = perform_CMTF(cube, glyCube, nComp)
    tensorR = tl.kruskal_to_tensor(factors)

    recon = [tensorR[indx[0], indx[1], indx[2]] for indx in indices]

    return np.array(orig), np.array(recon)


def evaluate_missing():
    """ check differences between original and recon values for different number of components. """
    Cube, glyCube = createCube()

    Sumsqs = list()
    for comp in np.arange(1, 8):
        orig, recon = evalMissing(Cube, glyCube, nComp=comp, numSample=100)

        Sumsqs.append(np.linalg.norm(orig - recon) / np.linalg.norm(orig))

    return Sumsqs
