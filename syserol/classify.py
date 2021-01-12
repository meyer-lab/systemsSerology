""" Regression methods using Factorized Data. """
from sklearn.metrics import accuracy_score
from .dataImport import load_file, importAlterDF, selectAlter
from .regression import RegressionHelper


def getClassPred(X, df):
    """ Extract Ys for classification. """
    Y1 = (df["class.cp"] == "controller").astype(int)  # controllers are 1s, progressors are 0s
    Y2 = (df["class.nv"] == "viremic").astype(int)  # viremic = 1, nonviremic = 0

    return RegressionHelper(X, Y1, classify=True), RegressionHelper(X, Y2, classify=True), Y1, Y2


def class_predictions(X):
    """ Predict Subject Class with Decomposed Tensor Data """
    # Load Data
    cp, nv, Y_cp, Y_nv = getClassPred(X, load_file("meta-subjects"))

    accuracies = {}
    accuracies["cp_all"] = accuracy_score(*selectAlter(Y_cp, cp[0]))
    accuracies["nv_all"] = accuracy_score(*selectAlter(Y_nv, nv[0]))
    return accuracies, cp[1], nv[1]


def two_way_classifications():
    """ Predict classifications of subjects by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP) - Alter methods"""
    df = importAlterDF(function=False, subjects=True)

    # Subset, Z score
    X = df.drop(["subject", "class.etuv", "class.cp", "class.nv"], axis=1)
    cp, nv, Y1, Y2 = getClassPred(X, df)
    return accuracy_score(Y1, cp[0]), accuracy_score(Y2, nv[0])
