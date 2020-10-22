""" Regression methods using Factorized Data. """
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
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
    _, cp_accuracy = ClassifyHelper(X, cp)

    # Viremic/Nonviremic classification
    _, nv_accuracy = ClassifyHelper(X, nv)

    return cp_accuracy, nv_accuracy


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
    _, accuracyCvP = ClassifyHelper(X, Y1)

    # Predict Viremic vs. Nonviremic
    _, accuracyVvN = ClassifyHelper(X, Y2)

    return accuracyCvP, accuracyVvN


def ClassifyHelper(X, Y):
    """ Function with common Logistic regression methods. """
    regr = LogisticRegressionCV(n_jobs=-1, cv=40, max_iter=1000).fit(X, Y)
    clf = LogisticRegression(C=regr.C_[0], max_iter=1000).fit(X, Y)
    Y_pred = cross_val_predict(clf, X, Y, cv=40, n_jobs=-1)
    return Y_pred, accuracy_score(Y, Y_pred)
