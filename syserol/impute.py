""" Evaluate the ability of CP to impute data. """

import numpy as np
import tensorly as tl
from statsmodels.multivariate.pca import PCA
from .dataImport import createCube
from .tensor import perform_CMTF


def flatten_to_mat(tensor, matrix):
    """ Flatten a tensor and a matrix into just a matrix """
    n = tensor.shape[0]
    tMat = np.reshape(tensor, (n, -1))
    tMat = tMat[:, ~np.all(np.isnan(tMat), axis=0)]
    tMat = np.hstack((tMat, matrix))
    return tMat

def deletable(cube, emin=6):
    """ Return a binary cube, test that each chord has at least emin elements """
    emin = max(emin, 2)
    miss_cube = np.isfinite(cube)
    deletable_cube = np.ones_like(cube, dtype=int)
    if cube.ndim == 3:
        for d in range(3):
            js, ks = (np.sum(np.rollaxis(miss_cube, d), axis=0) < emin).nonzero()
            for ii in range(len(js)):
                np.rollaxis(deletable_cube, d)[:, js[ii], ks[ii]] = 0
    elif cube.ndim == 2:
        for d in range(2):
            js = (np.sum(np.rollaxis(miss_cube, d), axis=0) < emin).nonzero()
            for ii in range(len(js)):
                np.rollaxis(deletable_cube, d)[:, js[ii]] = 0
    return deletable_cube * miss_cube


def missingness(c):
    return 1 - np.sum(np.isfinite(c))/np.prod(c.shape)

def create_missing(cube, numSample):
    """ Remove numSample amount of values from cube while maintain the condition set by deletable()
    Absolute (realistic) upper bound: cube remove 93577 (~86000), glyCube remove 2450 (~2300), """

    missingCube = np.copy(cube)
    emin = 6
    while numSample > 0:
        delet = deletable(missingCube, emin=emin)
        idxs = np.argwhere(delet)
        # delete at most 5% of deletable item at a time
        numDel = min(np.sum(delet) // 20, numSample)
        if numDel <= 1:
            numDel = 1
            if emin > 2:
                emin -= 1
            else:
                raise RuntimeError("Cannot create this many missing values")
        #print(np.sum(delet) // 20, numSample, emin)
        if cube.ndim == 3:
            for (i, j, k) in idxs[np.random.choice(idxs.shape[0], numDel, replace=False)]:
                missingCube[i, j, k] = np.nan
        elif cube.ndim == 2:
            for (i, j) in idxs[np.random.choice(idxs.shape[0], numDel, replace=False)]:
                missingCube[i, j] = np.nan
        numSample -= numDel
    #print('Missingness', missingness(missingCube))
    return missingCube

def increase_missing(comps, PCAcompare=False):
    samples = np.array([1000, 5000, 12000, 20000, 28000, 36000, 44000, 52000, 60000, 68000, 76000, 80000, 82000, 84000, 86000, 89000, 90000])
    CMTFR2Xs = np.zeros(samples.shape)
    PCAR2Xs = np.zeros(samples.shape)
    missing = np.zeros(samples.shape)
    for ii, sample in enumerate(samples):
        print("Running sample: ", sample)
        CMTFR2X, PCAR2X, missingFrac = evaluate_missing(comps, numSample=sample, chords=False, PCAcompare=PCAcompare)
        CMTFR2Xs[ii] = CMTFR2X[-1]
        PCAR2Xs[ii] = PCAR2X[-1]
        missing[ii] = missingFrac

    return CMTFR2Xs, PCAR2Xs, missing


def evaluate_missing(comps, numSample=15, chords=True, PCAcompare=False):
    """ check differences between original and recon values for different number of components.
    chords: whether to leave out tensor chords or individual values """
    cube, glyCube = createCube()

    CMTFR2X = np.zeros(comps.shape)
    PCAR2X = np.zeros(comps.shape)

    if chords:
        missingCube = np.copy(cube)
        for _ in range(numSample):
            idxs = np.argwhere(np.isfinite(missingCube))
            i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
            missingCube[:, j, k] = np.nan
    else:
        missingCube = create_missing(np.copy(cube), numSample)

    missingFrac = np.isnan(missingCube).sum() / np.prod(cube.shape)
    imputeVals = np.copy(cube)
    imputeVals[np.isfinite(missingCube)] = np.nan

    if PCAcompare:
        missingMat = flatten_to_mat(missingCube, glyCube)
        imputeMat = np.copy(flatten_to_mat(cube, glyCube))
        imputeMat[np.isfinite(missingMat)] = np.nan

    for ii, nComp in enumerate(comps):
        # reconstruct with some values missing
        tensorR = tl.cp_to_tensor(perform_CMTF(missingCube, glyCube, nComp))
        tensorR[np.isfinite(missingCube)] = np.nan

        # Compare original Cube with reconstructed cube, which was created from the cube with imputed missing values
        CMTFR2X[ii] = 1.0 - np.nanvar(tensorR - imputeVals) / np.nanvar(imputeVals)

        if PCAcompare:
            outt = PCA(missingMat, ncomp=nComp, missing="fill-em", standardize=False, demean=False, normalize=False)
            recon = outt.scores @ outt.loadings.T
            PCAR2X[ii] = 1.0 - np.nanvar(recon - imputeMat) / np.nanvar(imputeMat)

    return CMTFR2X, PCAR2X, missingFrac
