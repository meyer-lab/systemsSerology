""" Regression methods using Factorized Data. """
from sklearn.metrics import accuracy_score
from scipy.stats import zscore
from syserol.dataImport import (
    load_file,
    importAlterDF,
)
from glmnet import LogitNet

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
    _, cp_accuracy, coef_cp = ClassifyHelper(X, cp)

    # Viremic/Nonviremic classification
    _, nv_accuracy, coef_nv = ClassifyHelper(X, nv)

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
    scores = []
    for _ in range(100):
        clf = LogitNet(alpha=.8, n_splits=10, n_jobs=25, scoring="mean_squared_error").fit(X, Y)
        Y_pred = clf.predict(X)
        scores.append([Y_pred, accuracy_score(Y, Y_pred), clf.coef_])
    scores.sort(key=lambda x: x[1])
    return scores[49][0], scores[49][1], scores[49][2]
