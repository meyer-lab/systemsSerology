""" Evaluate the ability of CP to impute data. """

import numpy as np
from .dataImport import createCube
from .tensor import impute


def evalMissing(cube, nComp = 1, numSample = 100):
    """ Evaluate how well factorization imputes missing values. """
    cube = np.copy(cube)
    orig = []
    recon = []

    indices = list()
    idxs = np.argwhere(np.isfinite(cube))

    for ii in range(numSample):
        i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]

        indices.append((i, j, k))
        orig.append(cube[i, j, k])
        cube[i, j, k] = np.nan

    tensorR = impute(cube, nComp)
    
    for ii in range(len(indices)):
        recon.append(tensorR[indices[ii][0], indices[ii][1], indices[ii][2]])

    return np.array(orig), np.array(recon)


def evaluate_missing():
    Cube, GlyCube = createCube()
    
    #check differences between original and recon values for different number of components
    Averages = list()
    Sums = list()
    for comp in np.arange(1,10):
        orig, recon = evalMissing(Cube, nComp = comp, numSample = 100)

        Diff = np.absolute(orig - recon)
        Avg = np.mean(Diff)
        Sum = np.sum(Diff)
        print(f"The average difference for {comp} components is: {Avg} and the Sum is: {Sum}")
        Averages.append(Avg)
        Sums.append(Sum)

    return Averages, Sums
