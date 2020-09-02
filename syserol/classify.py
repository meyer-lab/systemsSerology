""" Regression methods using Factorized Data. """
from functools import reduce
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from scipy.stats import zscore
from syserol.dataImport import (
    load_file,
    importLuminex,
    importIGG,
    importAlterDF
)


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


def logistic_2class_predictions(subjects_matrix):
    """ Predict Subject Class with Decomposed Tensor Data and Logistic Regression"""
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
    regr = LogisticRegressionCV(penalty="elasticnet", solver="saga", l1_ratios=[0.0, 0.1, 0.5, 0.9, 1.0])
    regr.fit(subjects_matrix, Y)
    Y_pred1 = cross_val_predict(LogisticRegression(penalty="elasticnet", solver="saga", C=regr.C_[0], l1_ratio=regr.l1_ratio_[0]), subjects_matrix, Y)
    cp_accuracy = accuracy_score(Y, Y_pred1)

    # Viremic/Nonviremic classification
    Y = nv
    regr = LogisticRegressionCV(penalty="elasticnet", solver="saga", l1_ratios=[0.0, 0.1, 0.5, 0.9, 1.0])
    regr.fit(subjects_matrix, Y)
    Y_pred2 = cross_val_predict(LogisticRegression(penalty="elasticnet", solver="saga", C=regr.C_[0], l1_ratio=regr.l1_ratio_[0]), subjects_matrix, Y)
    nv_accuracy = accuracy_score(Y, Y_pred2)

    return cp_accuracy, nv_accuracy


def two_way_classifications():
    """ Predict classifications of subjects by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP) - Alter methods"""
    # Import Luminex, Luminex-IGG, Subject group pairs into DF
    df = importAlterDF()
    subj = load_file("meta-subjects")
    df_merged = df.merge(subj, on="subject", how="inner")
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
