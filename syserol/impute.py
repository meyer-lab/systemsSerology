""" Evaluate the ability of CP to impute data. """

import numpy as np
import tensorly as tl
from statsmodels.multivariate.pca import PCA
from .dataImport import createCube
from .tensor import perform_CMTF


def evaluate_missing(comps, numSample=15):
    """ check differences between original and recon values for different number of components. """
    # Evaluate ability to impute missing data using cube and tensor factorization
    cube, glyCube = createCube()
    np.random.seed(1)  # Avoid random variation in this output

    # Build flattened matrix from cube + glycube, for PCA imputation, below
    tMat = np.reshape(cube, (181, -1))
    tMat = tMat[:, ~np.all(np.isnan(tMat), axis=0)]
    tMat = np.hstack((tMat, glyCube))

    # TMTF IMPUTATION

    R2X_TMTF = np.zeros(comps.shape)
    missingCube = np.copy(cube)
    for _ in range(numSample):
        idxs = np.argwhere(np.isfinite(missingCube))
        _, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
        missingCube[:, j, k] = np.nan

    # Remove any original values that are not imputed, so as to only compare imputed values
    imputeVals = np.copy(cube)
    imputeVals[np.isfinite(missingCube)] = np.nan

    for ii, nComp in enumerate(comps):
        # reconstruct with some values missing
        tensorR = tl.cp_to_tensor(perform_CMTF(missingCube, glyCube, nComp)[0])
        tensorR[np.isfinite(missingCube)] = np.nan

        # Compare original Cube with reconstructed cube, which was created from the cube with imputed missing values
        R2X_TMTF[ii] = 1.0 - np.nanvar(tensorR - imputeVals) / np.nanvar(imputeVals)

    # PCA IMPUTATION
    # Evaluate ability to impute missing data using flat matrix and PCA
    imputeFlattened = np.copy(tMat)
    for _ in range(numSample):
        idxs = np.argwhere(np.isfinite(imputeFlattened))
        _, j= idxs[np.random.choice(idxs.shape[0], 1)][0]
        imputeFlattened[:, j] = np.nan

    # Remove any original values that are not imputed, so as to only compare imputed values
    imputeVals_matrix = np.copy(tMat)
    imputeVals_matrix[np.isfinite(imputeFlattened)] = np.nan

    R2X_PCA = np.zeros(comps.shape)
    for ii, cc in enumerate(comps):
        outt = PCA(tMat, ncomp=cc, missing="fill-em", standardize=False, demean=False, normalize=False)
        recon = outt.scores @ outt.loadings.T
        recon[np.isfinite(imputeFlattened)] = np.nan

        # Compare original matrix with reconstructed matrix, which was created from the matrix with imputed missing values
        R2X_PCA[ii] = 1.0 - np.nanvar(recon - imputeVals_matrix) / np.nanvar(imputeVals_matrix)

    return R2X_TMTF, R2X_PCA
