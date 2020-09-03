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
)


def getClassY(df):
    """ Extract Ys for classification. """
    Y1 = (df["class.cp"] == "controller").astype(int)  # controllers are 1s, progressors are 0s
    Y2 = (df["class.nv"] == "viremic").astype(int)  # viremic = 1, nonviremic = 0
    return Y1, Y2


def class_predictions(subjects_matrix, methodLR):
    """ Predict Subject Class with Decomposed Tensor Data """
    # Load Data
    cp, nv = getClassY(load_file("meta-subjects"))

    # Controller/Progressor classification
    _, cp_accuracy, _ = ClassifyHelper(subjects_matrix, cp, methodLR)

    # Viremic/Nonviremic classification
    _, nv_accuracy, _ = ClassifyHelper(subjects_matrix, nv, methodLR)

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
    Y1, Y2 = getClassY(df_class)

    # Predict Controller vs. Progressor
    _, accuracyCvP, confusionCvP = ClassifyHelper(df_variables, Y1, True)

    # Predict Viremic vs. Nonviremic
    _, accuracyVvN, confusionVvN = ClassifyHelper(df_variables, Y2, True)

    return accuracyCvP, accuracyVvN, confusionCvP, confusionVvN


def ClassifyHelper(X, Y, methodLR):
    """ Function with common Logistic regression methods. """
    if methodLR is True:
        regr = LogisticRegressionCV(n_jobs=-1)
        regr.fit(X, Y)
        clf = LogisticRegression(C=regr.C_[0])
    else:
        clf = SVC(kernel="rbf")

    Y_pred = cross_val_predict(clf, X, Y, cv=30, n_jobs=-1)
    confusion = confusion_matrix(Y, Y_pred)
    accuracy = accuracy_score(Y, Y_pred)
    return Y_pred, accuracy, confusion
