""" Regression methods using Factorized Data. """
import numpy as np
import tensorly as tl
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from tensorly.kruskal_tensor import kruskal_to_tensor
from syserol.tensor import perform_CMTF
from syserol.dataImport import createCube, importFunction, importGlycan


def test_predictions(function="ADCD"):
    """ Test correlation between original glyCube matrix and CMTF decomposed/reconstructed matrix"""
    cube, glyCube = createCube()
    _, mapped = importFunction()
    glycan, _ = importGlycan()
    corr = list()

    for comp in np.arange(1, 16):
        _, matrixFac, _ = perform_CMTF(cube, glyCube, comp)
        reconMatrix = kruskal_to_tensor(matrixFac)
        x = mapped[function]
        j = len(glycan) + x
        orig = list()
        recon = list()
        for i in range(len(glyCube)):
            if np.isfinite(glyCube[i, j]):
                orig.append(glyCube[i, j])
                recon.append(reconMatrix[i, j])
        corr.append(np.sqrt(r2_score(orig, recon)))
        print(f"Correlation for component {comp}: {np.sqrt(r2_score(orig, recon))}")

    return corr


def cross_validation():
    """ 10 Fold Cross Validation to Test Predictive Abilities"""
    cube, glyCube = createCube()
    _, mapped = importFunction()
    glycan, _ = importGlycan()

    X = glyCube
    matrix = np.zeros([181, 12])

    kf = KFold(n_splits=10, shuffle=True)  # split into 10 folds
    for train_index, test_index in kf.split(X):  # run cross validation
        copy = glyCube.copy()  # copy & restore original values at start of each cross validation fold
        matrix[test_index, 0:6] = copy[test_index, len(glycan) : len(glycan) + 6]  # store original value
        copy[test_index, len(glycan) : len(glycan) + 6] = np.nan  # artificially make the value NaN

        _, matrixFac, _ = perform_CMTF(cube, copy, 2)  # run decomposition on new matrix
        pred_matrix = tl.kruskal_to_tensor(matrixFac)
        matrix[test_index, 6:13] = pred_matrix[test_index, len(glycan) : len(glycan) + 6]  # store predicted values

    return matrix
