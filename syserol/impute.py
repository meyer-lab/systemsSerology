""" Evaluate the ability of CP to impute data. """

import numpy as np
import tensorly as tl
from statsmodels.multivariate.pca import PCA
from .dataImport import createCube
from .tensor import perform_CMTF, calcR2X


def flatten_to_mat(tensor, matrix):
    """ Flatten a tensor and a matrix into just a matrix """
    n = tensor.shape[0]
    tMat = np.reshape(tensor, (n, -1))
    tMat = tMat[:, ~np.all(np.isnan(tMat), axis=0)]
    tMat = np.hstack((tMat, matrix))
    return tMat


def missingness(c):
    """ Calculate the missingness (NaN) in an array"""
    vTop, vBot = 0.0, 0.0
    for cc in c:
        vTop += np.sum(np.isfinite(cc))
        vBot += np.prod(cc.shape)
    return 1.0 - vTop / vBot


def gen_missing(cube, missing_num, emin=6):
    """ Generate a cube with missing values """
    choose_cube = np.isfinite(cube)
    fill_cube = np.zeros_like(cube, dtype=int)

    # Generate a bare minimum cube
    # fill each individual with emin elements
    for ii in range(cube.shape[0]):
        idxs = np.argwhere(choose_cube[ii, :])
        if len(idxs) <= 0:
            continue
        jk = idxs[np.random.choice(idxs.shape[0], emin, replace=False)]
        if cube.ndim == 3:
            fill_cube[ii, jk[:, 0], jk[:, 1]] = 1
            choose_cube[ii, jk[:, 0], jk[:, 1]] = 0
        elif cube.ndim == 2:
            fill_cube[ii, jk[:, 0]] = 1
            choose_cube[ii, jk[:, 0]] = 0

    # fill each non-empty chord with emin elements
    fill_feat = np.any(np.isfinite(cube), axis=0) * emin - np.sum(fill_cube, axis=0)
    for jk in np.argwhere(fill_feat > 0):
        idxs = np.argwhere(choose_cube[:, jk[0], jk[1]]) if cube.ndim == 3 else np.argwhere(choose_cube[:, jk[0]])
        if len(idxs) <= 0:
            continue
        iis = idxs[np.random.choice(idxs.shape[0], fill_feat[jk[0], jk[1]], replace=False)]
        if cube.ndim == 3:
            fill_cube[iis, jk[0], jk[1]] = 1
            choose_cube[iis, jk[0], jk[1]] = 0
        elif cube.ndim == 2:
            fill_cube[iis, jk[0]] = 1
            choose_cube[iis, jk[0]] = 0
    assert np.all((np.sum(fill_cube, axis=0) >= emin) == np.any(np.isfinite(cube), axis=0))

    # fill up the rest to the missing nums
    to_fill = np.sum(np.isfinite(cube)) - missing_num - np.sum(fill_cube)
    assert to_fill <= np.sum(choose_cube)
    assert to_fill > 0
    idxs = np.argwhere(choose_cube)
    ijk = idxs[np.random.choice(idxs.shape[0], to_fill, replace=False)]
    if cube.ndim == 3:
        fill_cube[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1
    elif cube.ndim == 2:
        fill_cube[ijk[:, 0], ijk[:, 1]] = 1

    gen_cube = np.copy(cube)
    gen_cube[fill_cube == 0] = np.nan
    return gen_cube


def increase_missing(comp):
    """ Generate excessive missing values and impute for Fig 3c  """
    cube, glyCube = createCube()
    samples = np.array([1000, 5000, 12000, 20000, 28000, 36000, 44000, 52000, 60000,
                        68000, 76000, 80000, 82000, 84000, 86000, 88000])
    CMTFR2Xs = np.zeros(samples.shape)
    PCAR2Xs = np.zeros(samples.shape)
    missing = np.zeros(samples.shape)

    for ii, sample in enumerate(samples):
        print("Running sample: ", sample)
        missCube = gen_missing(cube, int(sample * 0.979))
        missGlyCube = gen_missing(glyCube, int(sample * 0.021))

        CMTFR2X, PCAR2X = impute_accuracy(missCube, missGlyCube, np.array([comp]))
        CMTFR2Xs[ii] = CMTFR2X[-1]
        PCAR2Xs[ii] = PCAR2X[-1]
        missing[ii] = missingness([missCube, missGlyCube])

    return CMTFR2Xs, PCAR2Xs, missing


def evaluate_missing(comps, numSample=15, chords=True):
    """ Wrapper for chord loss or individual loss """
    cube, glyCube = createCube()
    if chords:
        missingCube = np.copy(cube)
        for _ in range(numSample):
            idxs = np.argwhere(np.isfinite(missingCube))
            i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
            missingCube[:, j, k] = np.nan
    else:
        missingCube = gen_missing(np.copy(cube), numSample)
    return impute_accuracy(missingCube, glyCube, comps, PCAcompare=(not chords))


def impute_accuracy(missingCube, missingGlyCube, comps, PCAcompare=True, ALS=True):
    """ Calculate the imputation R2X """
    cube, glyCube = createCube()
    CMTFR2X = np.zeros(comps.shape)
    PCAR2X = np.zeros(comps.shape)

    # compare artificially introduced missingness only
    imputeCube = np.copy(cube)
    imputeCube[np.isfinite(missingCube)] = np.nan
    imputeGlyCube = np.copy(glyCube)
    imputeGlyCube[np.isfinite(missingGlyCube)] = np.nan

    if PCAcompare:
        missingMat = flatten_to_mat(missingCube, missingGlyCube)
        imputeMat = np.copy(flatten_to_mat(cube, glyCube))
        imputeMat[np.isfinite(missingMat)] = np.nan

    for ii, nComp in enumerate(comps):
        # reconstruct with some values missing
        recon_cmtf = perform_CMTF(missingCube, missingGlyCube, nComp)
        CMTFR2X[ii] = calcR2X(recon_cmtf, tIn=imputeCube, mIn=imputeGlyCube)

        if PCAcompare:
            outt = PCA(missingMat, ncomp=nComp, missing="fill-em", standardize=False, demean=False, normalize=False)
            recon_pca = outt.scores @ outt.loadings.T
            PCAR2X[ii] = calcR2X(recon_pca, mIn=imputeMat)

    return CMTFR2X, PCAR2X
