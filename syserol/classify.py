""" Regression methods using Factorized Data. """
from functools import reduce
import numpy as np
import pandas as pd
import tensorly as tl
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from tensorly.kruskal_tensor import kruskal_to_tensor
from scipy.stats import zscore
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


def two_way_classifications():
    """ Predict classifications of subjects by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP) - Alter methods"""
    # Import Luminex, Luminex-IGG, Subject group pairs, and Glycan into DF
    df = importLuminex()
    lum = df.pivot(index="subject", columns="variable", values="value")
    subj = load_file("meta-subjects")
    igg = importIGG()

    igg = igg.pivot(index="subject", columns="variable", values="value")
    data_frames = [lum, subj, igg]
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on=["subject"], how="inner"),
        data_frames,
    )
    df_merged = df_merged.dropna()

    # Subset, Z score
    df_class = df_merged[["class.cp", "class.nv"]]
    df_variables = df_merged.drop(
        ["subject", "class.etuv", "class.cp", "class.nv"], axis=1
    )
    df_variables = df_variables.apply(zscore)

    # Predict Controller vs. Progressor
    Y1 = df_class["class.cp"]
    Y1 = (Y1 == "controller").astype(int)  # controllers are 1s, progressors are 0s
    X1 = df_variables

    Y_pred1 = cross_val_predict(LogisticRegressionCV(), X1, Y1)
    model1 = LogisticRegressionCV().fit(X1, Y1)

    print(model1.coef_)
    confusionCvP = confusion_matrix(Y1, Y_pred1)
    accuracyCvP = accuracy_score(Y1, Y_pred1)
    print(f"Confusion Matrix Controller vs. Progressor: {confusionCvP} \n")

    # Predict Viremic vs. Nonviremic
    Y2 = df_class["class.nv"]
    Y2 = (Y2 == "viremic").astype(int)  # viremic = 1, nonviremic = 0
    X2 = df_variables

    Y_pred2 = cross_val_predict(LogisticRegressionCV(), X2, Y2)
    model2 = LogisticRegressionCV().fit(X2, Y2)

    print(model2.coef_)
    confusionVvN = confusion_matrix(Y2, Y_pred2)
    accuracyVvN = accuracy_score(Y2, Y_pred2)
    print(f"Confusion Matrix Viremic vs. Nonviremic: {confusionVvN} \n")

    return accuracyCvP, accuracyVvN, confusionCvP, confusionVvN
