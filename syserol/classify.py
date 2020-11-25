""" Regression methods using Factorized Data. """
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from scipy.stats import zscore
from syserol.dataImport import (
    load_file,
    importAlterDF,
)


def getClassY(df):
    """ Extract Ys for classification. """
    Y1 = (df["class.cp"] == "controller").astype(int)  # controllers are 1s, progressors are 0s
    Y2 = (df["class.nv"] == "viremic").astype(int)  # viremic = 1, nonviremic = 0
    return Y1, Y2


def class_predictions(X):
    """ Predict Subject Class with Decomposed Tensor Data """
    # Load Data
    cp, nv = getClassY(load_file("meta-subjects"))

    # Controller/Progressor classification
    _, cp_accuracy, coef_cp = ClassifySVC(X, cp)

    # Viremic/Nonviremic classification
    _, nv_accuracy, coef_nv = ClassifySVC(X, nv)

    return cp_accuracy, nv_accuracy, coef_cp, coef_nv


def two_way_classifications():
    """ Predict classifications of subjects by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP) - Alter methods"""
    df_merged = importAlterDF(function=False, subjects=True)

    # Subset, Z score
    X = df_merged.drop(
        ["subject", "class.etuv", "class.cp", "class.nv"], axis=1
    )
    X = X.apply(zscore)
    Y1, Y2 = getClassY(df_merged)

    # Predict Controller vs. Progressor
    _, accuracyCvP, _ = ClassifyHelper(X, Y1)

    # Predict Viremic vs. Nonviremic
    _, accuracyVvN, _ = ClassifyHelper(X, Y2)

    return accuracyCvP, accuracyVvN


def ClassifyHelper(X, Y):
    """ Function with common Logistic regression methods. """
    regr = LogisticRegressionCV(n_jobs=-1, cv=40, max_iter=1000).fit(X, Y)
    clf = LogisticRegression(C=regr.C_[0], max_iter=1000).fit(X, Y)
    coef = clf.coef_
    Y_pred = cross_val_predict(clf, X, Y, cv=40, n_jobs=-1)
    return Y_pred, accuracy_score(Y, Y_pred), coef


def ClassifySVC(X, Y):
    if X.shape[1] < 2:
        return np.zeros(Y.shape[0]), 0, np.zeros[X.shape[1]]
    Cs = np.logspace(-4, 4, num=50)
    kernels = ['linear', 'rbf']
    skf = StratifiedKFold(n_splits=10)
    comp = X.shape[1]
    values_all = []
    for C in Cs:
        for kernel in kernels:
            for i in range(0, comp - 1):
                for j in range(i + 1, comp):
                    folds = []
                    Y_preds = []
                    for train, test in skf.split(X, Y):
                        double = np.vstack((X[train, i], X[train, j])).T
                        clf = SVC(kernel=kernel, C=C).fit(double, Y[train])
                        test_double = np.vstack((X[test, i], X[test, j])).T
                        Y_pred = clf.predict(test_double)
                        Y_preds.append(Y_pred)
                        score = accuracy_score(Y[test], Y_pred)
                        folds.append([score])
                    values_all.append([i, j, np.mean(folds), folds, Y_preds, kernel, C])
    df_comp = pd.DataFrame(values_all)
    df_comp.columns = ["First", "Second", "Score", "All_scores", "Y_pred", "kernel", "C"]
    df_comp = df_comp.sort_values(by="Score", ascending=False)
    components = np.zeros(comp)
    components[df_comp.iloc[0, 0]] = 1
    components[df_comp.iloc[0, 1]] = 1
    return df_comp.iloc[0, 4], df_comp.iloc[0, 2], components
