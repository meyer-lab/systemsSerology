""" Evaluate the ability of CP to impute data. """

import numpy as np
from .dataImport import createCube
from .tensor import impute


def evalMissing(cube, nComp=1, numSample=100):
    """ Evaluate how well factorization imputes missing values. """
    cube = np.copy(cube)
    orig = []

    indices = list()

    for _ in range(numSample):
        idxs = np.argwhere(np.isfinite(cube))
        i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
        
        indices.append((i, j, k))
        orig.append(cube[i, j, k])
        cube[i, j, k] = np.nan

    tensorR = impute(cube, nComp)
    recon = [tensorR[indx[0], indx[1], indx[2]] for indx in indices]

    return np.array(orig), np.array(recon)

def evaluate_missing():
    """ check differences between original and recon values for different number of components. """
    Cube, _ = createCube()

    Sumsqs = list()
    for comp in np.arange(1, 10):
        orig, recon = evalMissing(Cube, nComp=comp, numSample=100)

        Sumsqs.append(np.sum(np.square(orig - recon)))

    return Sumsqs
