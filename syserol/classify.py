""" Regression methods using Factorized Data. """
from functools import reduce
import numpy as np
import pandas as pd
import tensorly as tl
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import SVC
from tensorly.kruskal_tensor import kruskal_to_tensor
from syserol.tensor import perform_CMTF
from syserol.dataImport import (
    functions,
    createCube,
    importFunction,
    importGlycan,
    load_file,
    importLuminex,
    importIGG,
    getAxes,
    importAlterDF,
)


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


def SVM_2class_predictions(subjects_matrix):
    """ Predict Subject Class with Support Vector Machines and Decomposed Tensor Data"""
    # Load Data
    classes = load_file("meta-subjects")
    classes = classes.replace(
        to_replace=["controller", "progressor", "viremic", "nonviremic"],
        value=[1, 0, 1, 0],
    )
    cp = np.array(classes["class.cp"])
    nv = np.array(classes["class.nv"])

    # Controller/Progressor classification
    Y = cp
    # Kernel = RBF
    # Run SVM classifier model
    clf = SVC(kernel="rbf")
    y_pred1 = cross_val_predict(clf, subjects_matrix, Y, cv=20)
    cp_accuracy = accuracy_score(Y, y_pred1)

    # Viremic/Nonviremic classification
    Y = nv
    # Kernel = RBF
    # Run SVM classifier model
    y_pred2 = cross_val_predict(clf, subjects_matrix, Y, cv=20)
    nv_accuracy = accuracy_score(Y, y_pred2)

    return cp_accuracy, nv_accuracy
